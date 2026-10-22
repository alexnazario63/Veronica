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
        # Inverted postings keep updates incremental. Optional native indexes are
        # used when installed; the Python path remains dependency-free.
        self.postings: Dict[str, set[str]] = {}
        self._vector_index: Any = None
        self._bm25s_index: Any = None
        self._bm25s_ids: List[str] = []
        self._vector_ids: Dict[int, str] = {}
        self._next_vector_id = 0

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

    def add_document(self, doc: KnowledgeDocument, recompute_stats: bool = True, compute_embedding: bool = True) -> None:
        old_tokens = self.doc_tokens.get(doc.doc_id, ())
        for token in set(old_tokens):
            self.postings.get(token, set()).discard(doc.doc_id)
        self.documents[doc.doc_id] = doc
        self._bm25s_index = None  # lazily rebuild after incremental updates
        searchable_text = (
            f"{doc.title} {doc.vendor} {doc.family_model} {doc.os_name} "
            f"{doc.version_build} {doc.feature} {doc.content} {' '.join(doc.extracted_commands)}"
        )
        self.doc_tokens[doc.doc_id] = _tokenize_for_bm25(searchable_text)
        for token in set(self.doc_tokens[doc.doc_id]):
            self.postings.setdefault(token, set()).add(doc.doc_id)
        emb = self._compute_doc_embedding(doc) if compute_embedding else None
        if emb is not None:
            self.doc_embeddings[doc.doc_id] = emb
            self._index_embedding(doc.doc_id, emb)
        else:
            self.doc_embeddings.pop(doc.doc_id, None)
        if recompute_stats:
            self._recompute_index_stats()
            self.rebuild_bm25s_index()

    def rebuild_bm25s_index(self) -> None:
        """Reconstrói ou atualiza o índice esparso BM25S em tempo real."""
        try:
            import bm25s
            self._bm25s_ids = [
                doc_id for doc_id, doc in self.documents.items()
                if not doc.security_quarantined
            ]
            if self._bm25s_ids:
                corpus = [" ".join(self.doc_tokens.get(doc_id, ())) for doc_id in self._bm25s_ids]
                retriever = bm25s.BM25()
                retriever.index(bm25s.tokenize(corpus))
                self._bm25s_index = retriever
        except Exception:
            self._bm25s_index = None

    def index_document_in_realtime(
        self,
        doc: KnowledgeDocument,
        compute_embedding: bool = True,
    ) -> None:
        """Indexa dinamicamente um novo documento no BM25S e no índice vetorial em tempo real."""
        self.add_document(doc, recompute_stats=True, compute_embedding=compute_embedding)
        self.rebuild_bm25s_index()

    def _index_embedding(self, doc_id: str, embedding: np.ndarray) -> None:
        try:
            from usearch.index import Index
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if self._vector_index is None:
                self._vector_index = Index(ndim=len(vector), metric="cos", dtype="f32")
            label = self._next_vector_id
            self._next_vector_id += 1
            self._vector_ids[label] = doc_id
            self._vector_index.add(label, vector)
        except (ImportError, TypeError, ValueError):
            # Small installations and unsupported embedding shapes use exact
            # cosine scoring below.
            self._vector_index = None

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

    def _candidate_ids(self, query_terms: List[str]) -> set[str]:
        found: set[str] = set()
        for term in query_terms:
            found.update(self.postings.get(term, ()))
        return found

    def _bm25s_ranking(self, query: str, eligible: set[str]) -> List[str]:
        """Use bm25s' sparse index when available; retain postings BM25 fallback."""
        try:
            import bm25s
            if self._bm25s_index is None:
                self._bm25s_ids = [
                    doc_id for doc_id, doc in self.documents.items()
                    if not doc.security_quarantined
                ]
                corpus = [" ".join(self.doc_tokens.get(doc_id, ())) for doc_id in self._bm25s_ids]
                retriever = bm25s.BM25()
                retriever.index(bm25s.tokenize(corpus))
                self._bm25s_index = retriever
            query_tokens = bm25s.tokenize([query])
            indices, _scores = self._bm25s_index.retrieve(query_tokens, k=max(1, len(self._bm25s_ids)))
            order = indices[0].tolist() if hasattr(indices[0], "tolist") else list(indices[0])
            return [self._bm25s_ids[int(i)] for i in order if int(i) < len(self._bm25s_ids)
                    and self._bm25s_ids[int(i)] in eligible]
        except (ImportError, AttributeError, TypeError, ValueError, IndexError):
            return sorted(self._candidate_ids(_tokenize_for_bm25(query)) & eligible,
                           key=lambda d: self._bm25_score(_tokenize_for_bm25(query), d), reverse=True)

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

        # Reciprocal Rank Fusion avoids mixing incomparable BM25 and cosine
        # scales, and lets lexical and neural retrieval contribute independently.
        eligible = {
            doc_id for doc_id, doc in self.documents.items()
            if not doc.security_quarantined
            and (not vendor_filter or doc.vendor.lower() == vendor_filter.lower())
            and (not os_filter or os_filter.lower() in doc.os_name.lower())
            and (include_obsolete or not doc.is_obsolete)
        }
        lexical = self._bm25s_ranking(query, eligible)
        vector_scores: Dict[str, float] = {}
        if query_emb is not None:
            if self._vector_index is not None:
                try:
                    hits = self._vector_index.search(np.asarray(query_emb, dtype=np.float32), min(len(eligible), max(top_k * 8, 32)))
                    for label, distance in zip(hits.keys, hits.distances):
                        doc_id = self._vector_ids.get(int(label))
                        if doc_id in eligible:
                            vector_scores[doc_id] = 1.0 - float(distance)
                except (AttributeError, TypeError, ValueError):
                    vector_scores = {}
            if not vector_scores:
                for doc_id in eligible:
                    emb = self.doc_embeddings.get(doc_id)
                    if emb is not None:
                        vector_scores[doc_id] = float(np.dot(query_emb, emb))
        semantic = sorted(vector_scores, key=vector_scores.get, reverse=True)
        fused: Counter[str] = Counter()
        rrf_k = 60
        for ranking in (lexical, semantic):
            for rank, doc_id in enumerate(ranking, 1):
                fused[doc_id] += 1.0 / (rrf_k + rank)
        # Preserve vendor-filter behavior: a filtered vendor can still yield
        # useful candidates when query wording does not overlap its metadata.
        if vendor_filter and not fused:
            fused.update({doc_id: 1.0 / (rrf_k + 1) for doc_id in eligible})

        scored_results: List[Dict[str, Any]] = []
        obsolete_warnings: List[str] = []

        for doc_id in fused:
            doc = self.documents[doc_id]

            bm25 = self._bm25_score(query_terms, doc_id)

            doc_tok_set = set(self.doc_tokens.get(doc_id, []))
            matched_terms = [t for t in unique_q_terms if t in doc_tok_set]
            missing_terms = [t for t in unique_q_terms if t not in doc_tok_set]
            coverage = (len(matched_terms) / len(unique_q_terms)) if unique_q_terms else 1.0

            vec_sim = max(0.0, vector_scores.get(doc_id, 0.0))

            official_boost = 1.15 if doc.is_official else 0.90
            obsolete_penalty = 0.65 if doc.is_obsolete else 1.0
            coverage_weight = (0.25 + 0.75 * (coverage ** 2)) if (len(unique_q_terms) >= 2 and not vendor_filter) else 1.0
            combined_score = (
                fused[doc_id]
                * doc.reliability_score
                * official_boost
                * obsolete_penalty
                * coverage_weight
            )

            # RRF scores are naturally small (about 1/61 for a top-ranked hit),
            # so a legacy raw-BM25 cutoff silently removed every valid result.
            if combined_score > 0.0:
                if doc.is_obsolete:
                    obsolete_warnings.append(
                        f"Documento '{doc.doc_id}' ({doc.os_name} {doc.version_build}) é OBSOLETO; "
                        f"substituído por '{doc.superseded_by}'."
                    )
                scored_results.append(
                    {
                        "score": round(combined_score, 4),
                        "rrf_score": round(fused[doc_id], 6),
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
