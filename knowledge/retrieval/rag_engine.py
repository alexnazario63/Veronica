"""
Motor RAG Híbrido (Lexical BM25 + Embeddings Neurais Próprios da Veronica).
Prioriza fontes oficiais, detecta documentação obsoleta ou conflitante,
exclui documentos em quarentena de segurança e gera citações rastreáveis
com os 12 metadados obrigatórios (Seção 4).
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional, Sequence
import numpy as np

import unicodedata
from core.tokenizer.tokenizer import TechnicalTokenizer
from core.transformer.model import VeronicaTransformer
from knowledge.sources.catalog import KnowledgeDocument, OFFICIAL_INITIAL_DOCUMENTS

WORD_SPLIT_RE = re.compile(r"[a-zA-Z0-9À-ÿ_\-/\.]+")

STOPWORDS_PT_EN = {
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "os", "as", "ao", "aos", "por", "para",
    "com", "sem", "sob", "sobre", "que", "qual", "quais", "como", "onde",
    "quando", "quem", "porque", "se", "ou", "mas", "mais", "muito", "isso",
    "isto", "esse", "este", "esta", "sua", "seu", "sao", "ser", "foi", "ter",
    "tem", "pode", "deve", "sabe", "me", "te", "ele", "ela", "eles", "elas",
    "fale", "explique", "diga", "mostrar", "mostre", "existe", "existem",
    "the", "of", "in", "to", "for", "and", "or", "is", "are", "on", "with",
    "by", "at", "from", "what", "how", "why", "which", "who",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _tokenize_for_bm25(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in WORD_SPLIT_RE.findall(text):
        norm = _strip_accents(raw.lower()).strip(".-_/")
        if len(norm) > 1 and norm not in STOPWORDS_PT_EN:
            tokens.append(norm)
    return tokens


class HybridRAGEngine:
    """
    Motor de Recuperação Documental Especializado da Veronica.
    """

    def __init__(
        self,
        model: Optional[VeronicaTransformer] = None,
        tokenizer: Optional[TechnicalTokenizer] = None,
        initial_docs: Optional[Sequence[KnowledgeDocument]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or TechnicalTokenizer()
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.doc_embeddings: Dict[str, np.ndarray] = {}
        self.avg_doc_len: float = 1.0

        docs_to_load = list(initial_docs) if initial_docs is not None else OFFICIAL_INITIAL_DOCUMENTS
        for doc in docs_to_load:
            self.add_document(doc, recompute_stats=False)
        self._recompute_index_stats()

    def _compute_doc_embedding(self, doc: KnowledgeDocument) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        combined = f"{doc.vendor} {doc.os_name} {doc.version_build} {doc.feature} {doc.title} {doc.content}"
        ids = self.tokenizer.encode(combined, add_bos=True, max_length=self.model.config.max_seq_len)
        return self.model.encode_sequence_embedding(ids)

    def _recompute_index_stats(self) -> None:
        self.doc_freqs.clear()
        total_len = 0
        active_count = 0
        for doc_id, toks in self.doc_tokens.items():
            if self.documents[doc_id].security_quarantined:
                continue
            active_count += 1
            total_len += len(toks)
            for unique_t in set(toks):
                self.doc_freqs[unique_t] += 1
        self.avg_doc_len = float(total_len) / max(active_count, 1)

    def add_document(self, doc: KnowledgeDocument, recompute_stats: bool = True) -> None:
        self.documents[doc.doc_id] = doc
        searchable_text = (
            f"{doc.title} {doc.vendor} {doc.family_model} {doc.os_name} "
            f"{doc.version_build} {doc.feature} {doc.content} {' '.join(doc.extracted_commands)}"
        )
        self.doc_tokens[doc.doc_id] = _tokenize_for_bm25(searchable_text)
        emb = self._compute_doc_embedding(doc)
        if emb is not None:
            self.doc_embeddings[doc.doc_id] = emb
        if recompute_stats:
            self._recompute_index_stats()

    def refresh_neural_embeddings(self) -> None:
        if self.model is None:
            return
        for doc_id, doc in self.documents.items():
            emb = self._compute_doc_embedding(doc)
            if emb is not None:
                self.doc_embeddings[doc_id] = emb

    def _bm25_score(
        self,
        query_terms: List[str],
        doc_id: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        toks = self.doc_tokens.get(doc_id, [])
        if not toks:
            return 0.0
        tf_counter = Counter(toks)
        doc_len = len(toks)
        n_docs = max(len(self.documents), 1)
        score = 0.0

        for q in query_terms:
            tf = tf_counter.get(q, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(q, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (doc_len / max(self.avg_doc_len, 1.0)))
            score += idf * ((tf * (k1 + 1.0)) / denom)
        return score

    def search(
        self,
        query: str,
        vendor_filter: Optional[str] = None,
        os_filter: Optional[str] = None,
        include_obsolete: bool = True,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        query_terms = _tokenize_for_bm25(query)
        unique_q_terms = list(dict.fromkeys(query_terms))
        query_emb: Optional[np.ndarray] = None
        if self.model is not None:
            q_ids = self.tokenizer.encode(query, add_bos=True, max_length=self.model.config.max_seq_len)
            query_emb = self.model.encode_sequence_embedding(q_ids)

        scored_results: List[Dict[str, Any]] = []
        obsolete_warnings: List[str] = []

        for doc_id, doc in self.documents.items():
            if doc.security_quarantined:
                continue
            if vendor_filter and doc.vendor.lower() != vendor_filter.lower():
                continue
            if os_filter and os_filter.lower() not in doc.os_name.lower():
                continue
            if doc.is_obsolete and not include_obsolete:
                continue

            bm25 = self._bm25_score(query_terms, doc_id)
            if bm25 <= 0.0 and not vendor_filter:
                continue

            doc_tok_set = set(self.doc_tokens.get(doc_id, []))
            matched_terms = [t for t in unique_q_terms if t in doc_tok_set]
            missing_terms = [t for t in unique_q_terms if t not in doc_tok_set]
            coverage = (len(matched_terms) / len(unique_q_terms)) if unique_q_terms else 1.0

            vec_sim = 0.0
            if query_emb is not None and doc_id in self.doc_embeddings:
                vec_sim = float(np.dot(query_emb, self.doc_embeddings[doc_id]))
                vec_sim = max(0.0, vec_sim)

            official_boost = 1.15 if doc.is_official else 0.90
            obsolete_penalty = 0.65 if doc.is_obsolete else 1.0
            coverage_weight = (0.25 + 0.75 * (coverage ** 2)) if (len(unique_q_terms) >= 2 and not vendor_filter) else 1.0
            combined_score = (
                (bm25 + 0.4 * vec_sim)
                * doc.reliability_score
                * official_boost
                * obsolete_penalty
                * coverage_weight
            )

            if combined_score > 0.05:
                if doc.is_obsolete:
                    obsolete_warnings.append(
                        f"Documento '{doc.doc_id}' ({doc.os_name} {doc.version_build}) é OBSOLETO; "
                        f"substituído por '{doc.superseded_by}'."
                    )
                scored_results.append(
                    {
                        "score": round(combined_score, 4),
                        "bm25_score": round(bm25, 4),
                        "neural_embedding_similarity": round(vec_sim, 4),
                        "query_coverage": round(coverage, 4),
                        "matched_terms": matched_terms,
                        "missing_terms": missing_terms,
                        "document": doc.to_dict(),
                        "citation": {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "vendor": doc.vendor,
                            "family_model": doc.family_model,
                            "os_name": doc.os_name,
                            "version_build": doc.version_build,
                            "feature": doc.feature,
                            "original_source": doc.original_source,
                            "source_uri": doc.source_uri,
                            "page_section": doc.page_section,
                            "published_date": doc.published_date,
                            "consulted_date": doc.consulted_date,
                            "reliability_score": doc.reliability_score,
                            "applicability_restrictions": doc.applicability_restrictions,
                            "is_official": doc.is_official,
                            "is_obsolete": doc.is_obsolete,
                        },
                    }
                )

        scored_results.sort(key=lambda r: float(r["score"]), reverse=True)
        top_results = scored_results[:top_k]

        # Detecta conflitos de versão/sintaxe entre os resultados recuperados
        conflicts: List[str] = []
        by_vendor_os: Dict[str, List[Dict[str, Any]]] = {}
        for item in top_results:
            d = item["document"]
            key = f"{d['vendor']}:{d['os_name']}"
            by_vendor_os.setdefault(key, []).append(d)

        for key, docs_group in by_vendor_os.items():
            versions = {d["version_build"] for d in docs_group}
            if len(versions) > 1:
                conflicts.append(
                    f"Múltiplas versões de {key} recuperadas ({', '.join(sorted(versions))}). "
                    f"Não presumir compatibilidade direta de CLI entre versões distintas."
                )

        return {
            "query": query,
            "total_matches": len(top_results),
            "results": top_results,
            "obsolete_warnings": obsolete_warnings,
            "version_conflicts": conflicts,
        }
