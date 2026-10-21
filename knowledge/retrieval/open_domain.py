from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse

from knowledge.sources.catalog import KnowledgeDocument

_STOP = {
    "qual", "quais", "como", "quando", "onde", "porque", "por", "para", "com", "uma", "uns",
    "das", "dos", "que", "the", "what", "when", "where", "why", "does", "this", "that",
    "with", "from", "about", "explain", "tell", "have", "into", "your", "than", "then",
}

_DOMAIN_TERMS = {
    "networking": {"bgp", "ospf", "mpls", "router", "roteador", "rede", "redes", "network", "networking", "ethernet", "vlan", "ipv6", "rfc", "ietf", "switch", "route", "rota"},
    "physics": {"physics", "fisica", "quantum", "quantica", "gravitacao", "gravity", "energia", "particula", "particle", "relatividade", "relativity", "thermodynamics", "termodinamica", "eletromagnetismo"},
    "medicine": {"medicina", "medical", "saude", "health", "disease", "doenca", "tratamento", "treatment", "sintoma", "symptom", "drug", "medicamento", "diagnostico", "diagnosis"},
    "law": {"lei", "legal", "law", "regulation", "regulacao", "regulamento", "direito", "statute", "jurisprudencia", "court", "tribunal"},
    "chemistry": {"chemistry", "quimica", "chemical", "quimico", "molecule", "molecula", "reaction", "reacao", "element", "elemento", "periodic", "periodica"},
    "mathematics": {"math", "mathematics", "matematica", "algebra", "calculus", "calculo", "theorem", "teorema", "geometry", "geometria", "probability", "probabilidade"},
    "astronomy": {"astronomy", "astronomia", "planet", "planeta", "galaxy", "galaxia", "star", "estrela", "blackhole", "exoplanet", "exoplaneta", "cosmology", "cosmologia"},
    "history": {"history", "historia", "historical", "historico", "war", "guerra", "empire", "imperio", "civilization", "civilizacao"},
    "computing": {"software", "programming", "programacao", "python", "javascript", "api", "database", "banco", "linux", "compiler", "compilador", "algorithm", "algoritmo"},
}

_DOMAIN_AUTHORITIES = {
    "networking": ("rfc-editor.org", "datatracker.ietf.org", "ietf.org", "cisco.com", "juniper.net", "huawei.com", "mikrotik.com", "arista.com", "nokia.com"),
    "physics": ("nasa.gov", "esa.int", "cern.ch", "aps.org", "arxiv.org", "physicsworld.com"),
    "medicine": ("who.int", "nih.gov", "ncbi.nlm.nih.gov", "cdc.gov", "mayoclinic.org", "nhs.uk"),
    "law": ("gov.br", "planalto.gov.br", "supremecourt.gov", "uscourts.gov", "eur-lex.europa.eu", "legislation.gov.uk"),
    "chemistry": ("acs.org", "rsc.org", "pubchem.ncbi.nlm.nih.gov", "nist.gov", "chemrxiv.org"),
    "mathematics": ("ams.org", "mathworld.wolfram.com", "arxiv.org", "maa.org"),
    "astronomy": ("nasa.gov", "esa.int", "iau.org", "adsabs.harvard.edu"),
    "history": ("loc.gov", "archives.gov", "nationalarchives.gov.uk", "history.state.gov", "britannica.com"),
    "computing": ("python.org", "developer.mozilla.org", "docs.github.com", "kernel.org", "learn.microsoft.com"),
    "general": ("britannica.com", "arxiv.org"),
}


def terms(text: str) -> set[str]:
    normalized = "".join(c for c in unicodedata.normalize("NFD", text.casefold()) if unicodedata.category(c) != "Mn")
    return {word for word in re.findall(r"[\w-]{3,}", normalized) if word not in _STOP}


def classify_domain(query: str) -> str:
    query_terms = terms(query)
    scores = {domain: len(query_terms & vocab) for domain, vocab in _DOMAIN_TERMS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "general"


def assess_source(url: str, domain: str) -> Dict[str, Any]:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    markers = _DOMAIN_AUTHORITIES.get(domain, _DOMAIN_AUTHORITIES["general"])
    institutional = host.endswith((".gov", ".edu", ".gov.br"))
    primary = institutional or any(host == marker or host.endswith("." + marker) for marker in markers)
    if "wikipedia.org" in host:
        label, authority = "Encyclopedia", 0.45
    elif primary:
        label, authority = "Institutional or primary source", 1.0
    elif host.endswith((".org", ".int")):
        label, authority = "Organization", 0.72
    else:
        label, authority = "Web source", 0.55
    return {"host": host, "source_type": label, "is_primary_source": primary, "authority_score": authority}


def synthesize_evidence(query: str, sources: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract and rank claims from full pages; no unsupported facts are generated."""
    query_terms = terms(query)
    candidates: List[Dict[str, Any]] = []
    for source in sources:
        source_url = str(source.get("url", ""))
        body = str(source.get("content") or source.get("snippet") or "")
        source_terms = terms(f"{source.get('title', '')} {body}")
        relevance = len(query_terms & source_terms) / max(1, len(query_terms))
        if relevance < 0.08:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
            sentence = re.sub(r"\s+", " ", sentence).strip(" -•\t")
            if not 45 <= len(sentence) <= 700:
                continue
            sentence_terms = terms(sentence)
            sentence_overlap = len(query_terms & sentence_terms)
            if not sentence_overlap:
                continue
            authority = float(source.get("authority_score", 0.5))
            score = sentence_overlap / max(1, len(query_terms)) * 0.75 + authority * 0.25
            candidates.append({
                "text": sentence, "url": source_url,
                "title": source.get("title", "Consulted source"),
                "score": score, "relevance": relevance,
                "authority_score": authority,
                "is_primary_source": bool(source.get("is_primary_source", False)),
                "full_page": bool(source.get("full_page", False)),
            })

    candidates.sort(key=lambda item: (item["score"], item["relevance"], item["authority_score"]), reverse=True)
    selected: List[Dict[str, Any]] = []
    used_urls: set[str] = set()
    per_url_count: Counter[str] = Counter()
    seen_sentences: set[str] = set()
    for item in candidates:
        normalized = re.sub(r"\W+", " ", item["text"].casefold()).strip()
        if normalized in seen_sentences or per_url_count[item["url"]] >= 2:
            continue
        if len(used_urls) >= 3 and item["url"] not in used_urls:
            continue
        selected.append(item)
        seen_sentences.add(normalized)
        used_urls.add(item["url"])
        per_url_count[item["url"]] += 1
        if len(selected) >= 4:
            break

    numeric_claims: Dict[str, set[str]] = {}
    for item in selected:
        numbers = set(re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?:\s?(?:%|km/s|m/s|GHz|MHz|km|m|kg|N|J|eV|K|°C|°F))?", item["text"], re.I))
        if numbers:
            numeric_claims.setdefault(item["url"], set()).update(numbers)
    number_sets = list(numeric_claims.values())
    conflict = len(number_sets) > 1 and any(
        left.isdisjoint(right)
        for index, left in enumerate(number_sets)
        for right in number_sets[index + 1:]
    )

    by_source: Dict[str, List[set[str]]] = {}
    for item in selected:
        by_source.setdefault(item["url"], []).append(terms(item["text"]))
    corroboration_scores: List[float] = []
    for source_url, claims in by_source.items():
        other_claims = [claim for other_url, other in by_source.items() if other_url != source_url for claim in other]
        for claim in claims:
            if other_claims:
                corroboration_scores.append(max((len(claim & other) / max(1, len(claim | other)) for other in other_claims), default=0.0))
    corroboration = round(sum(corroboration_scores) / len(corroboration_scores), 3) if corroboration_scores else 0.0
    answer = " ".join(item["text"] for item in selected[:3])
    return {
        "answer": answer,
        "claims": selected,
        "source_count": len({item["url"] for item in candidates}),
        "full_page_count": len({item["url"] for item in candidates if item["full_page"]}),
        "primary_source_count": len({item["url"] for item in candidates if item["is_primary_source"]}),
        "corroboration": corroboration,
        "possible_numeric_conflict": conflict,
        "coverage": round(max((item["relevance"] for item in candidates), default=0.0), 3),
    }


class OpenDomainKnowledgeStore:
    """Durable local cache of sanitized source documents and anonymous quality metrics."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, payload TEXT NOT NULL, saved_at TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS research_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, domain TEXT NOT NULL, metrics TEXT NOT NULL, created_at TEXT NOT NULL)")
            connection.commit()

    def save_document(self, document: KnowledgeDocument) -> None:
        payload = json.dumps(document.to_dict(), ensure_ascii=False)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("INSERT OR REPLACE INTO documents(doc_id, payload, saved_at) VALUES (?, ?, ?)", (document.doc_id, payload, datetime.now(timezone.utc).isoformat()))
            connection.execute("DELETE FROM documents WHERE rowid NOT IN (SELECT rowid FROM documents ORDER BY saved_at DESC LIMIT 5000)")
            connection.commit()

    def load_documents(self, limit: int = 300) -> List[KnowledgeDocument]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute("SELECT payload FROM documents ORDER BY saved_at DESC LIMIT ?", (limit,)).fetchall()
        output: List[KnowledgeDocument] = []
        for (payload,) in rows:
            try:
                output.append(KnowledgeDocument(**json.loads(payload)))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return output

    def record_metrics(self, domain: str, metrics: Dict[str, Any]) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("INSERT INTO research_metrics(domain, metrics, created_at) VALUES (?, ?, ?)", (domain, json.dumps(metrics, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))
            connection.execute("DELETE FROM research_metrics WHERE id NOT IN (SELECT id FROM research_metrics ORDER BY id DESC LIMIT 10000)")
            connection.commit()

    def quality_summary(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute("SELECT domain, metrics FROM research_metrics ORDER BY id DESC LIMIT 10000").fetchall()
            doc_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for domain, payload in rows:
            try:
                grouped.setdefault(domain, []).append(json.loads(payload))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        by_domain: Dict[str, Dict[str, Any]] = {}
        for domain, items in grouped.items():
            count = len(items)
            by_domain[domain] = {
                "queries": count,
                "mean_source_count": round(sum(float(item.get("source_count", 0)) for item in items) / count, 2),
                "mean_full_page_count": round(sum(float(item.get("full_page_count", 0)) for item in items) / count, 2),
                "mean_primary_source_count": round(sum(float(item.get("primary_source_count", 0)) for item in items) / count, 2),
                "mean_claim_count": round(sum(float(item.get("claim_count", 0)) for item in items) / count, 2),
                "mean_coverage": round(sum(float(item.get("coverage", 0)) for item in items) / count, 3),
                "mean_corroboration": round(sum(float(item.get("corroboration", 0)) for item in items) / count, 3),
                "possible_conflict_rate": round(sum(bool(item.get("possible_numeric_conflict")) for item in items) / count, 3),
            }
        return {"indexed_documents": doc_count, "queries_recorded": len(rows), "by_domain": by_domain}
