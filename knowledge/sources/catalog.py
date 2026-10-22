"""
Motor de Catálogo Dinâmico e Persistente da Veronica (Seção 4 e 6).
Substitui catálogos estáticos por armazenamento dinâmico apoiado em SQLite
e no BinaryMachineMemoryStore (VRNC_MEM v2.0), expandindo metadados para suportar
qualquer organização, linguagem, framework e domínio do conhecimento
(Python, Rust, TypeScript, Linux, IEEE, IETF, Mathematics, DataScience, etc.),
além de fabricantes de rede (Cisco, Huawei, Juniper, Arista, etc.).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np



@dataclass
class KnowledgeDocument:
    """Documento de conhecimento canônico com 12 metadados obrigatórios e campos extensíveis."""
    doc_id: str
    title: str
    content: str
    # 12 Metadados Obrigatórios (Seção 4)
    vendor: str
    family_model: str
    os_name: str
    version_build: str
    feature: str
    original_source: str
    source_uri: str
    page_section: str
    published_date: str
    consulted_date: str
    reliability_score: float
    applicability_restrictions: str
    # Campos extraídos automaticamente
    extracted_commands: List[str] = field(default_factory=list)
    extracted_tables: List[str] = field(default_factory=list)
    is_official: bool = True
    is_obsolete: bool = False
    superseded_by: str = ""
    security_quarantined: bool = False
    quarantine_reason: str = ""
    domain: str = "routing"
    # Metadados dinâmicos expandidos para qualquer organização, linguagem, framework ou domínio
    organization: str = ""
    language_or_framework: str = ""
    topic_category: str = "general"
    metadata_extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeDocument:
        fields_set = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in data.items():
            if k in fields_set:
                kwargs[k] = v
            else:
                extra[k] = v
        if extra:
            meta = kwargs.get("metadata_extra") or {}
            meta.update(extra)
            kwargs["metadata_extra"] = meta
        return cls(**kwargs)


class DynamicDomainRegistry(dict):
    """
    Registro dinâmico e extensível de domínios do conhecimento.
    Suporta redes e qualquer domínio de ciência e engenharia de software.
    """

    def register(
        self,
        domain_id: str,
        title: str,
        topics: List[str],
        section_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        norm_id = domain_id.strip().lower()
        spec = {
            "section_number": section_number if section_number is not None else len(self) + 1,
            "title": title,
            "topics": list(topics),
            "metadata": metadata or {},
        }
        self[norm_id] = spec
        return spec


def _load_seed_payload() -> Dict[str, Any]:
    seed_path = Path(__file__).parent / "catalog_seed.json"
    if seed_path.exists():
        try:
            with open(seed_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


_SEED_PAYLOAD = _load_seed_payload()

# Inicializa os 7 domínios de redes dinâmicos a partir do seed
NETWORKING_KNOWLEDGE_DOMAINS: DynamicDomainRegistry = DynamicDomainRegistry()
for _dom_id, _dom_spec in _SEED_PAYLOAD.get("domains", {}).items():
    NETWORKING_KNOWLEDGE_DOMAINS[_dom_id] = _dom_spec

# Alias genérico para todos os domínios
KNOWLEDGE_DOMAINS: DynamicDomainRegistry = NETWORKING_KNOWLEDGE_DOMAINS


class DynamicKnowledgeCatalog:
    """
    Motor de Catálogo Dinâmico e Persistente da Veronica.
    - Persistência apoiada em SQLite (`storage_dir / "knowledge_catalog.sqlite3"`).
    - Indexação semântica acelerada com `BinaryMachineMemoryStore`.
    - Metadados abertos para qualquer organização, linguagem, framework e domínio do conhecimento.
    - Desacoplamento total de dados e código (zero strings de manuais no código fonte).
    """

    _default_instance: Optional[DynamicKnowledgeCatalog] = None

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
        vector_dim: int = 48,
        auto_seed: bool = True,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else Path("runtime_data")
        if db_path is not None:
            self.db_path = Path(db_path) if str(db_path) != ":memory:" else ":memory:"
        else:
            self.db_path = self.storage_dir / "knowledge_catalog.sqlite3"

        if isinstance(self.db_path, Path):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.vector_dim = vector_dim
        self._catalog_mem_system = type("_CatalogMem", (), {"semantic_nodes": {}})()
        self._binary_store: Any = None
        self._doc_memory_ids: Dict[str, str] = {}
        self._memory_id_to_doc_id: Dict[str, str] = {}
        self._in_memory_docs: Dict[str, KnowledgeDocument] = {}

        self._init_sqlite()
        if auto_seed and self._is_empty():
            self._seed_from_source(_SEED_PAYLOAD)
        self._load_cache()

    @property
    def binary_store(self) -> Any:
        if self._binary_store is None:
            try:
                from core.development.memory_system import BinaryMachineMemoryStore
                self._binary_store = BinaryMachineMemoryStore(
                    memory_system=self._catalog_mem_system,
                    vector_dim=self.vector_dim,
                )
            except Exception:
                self._binary_store = None
        return self._binary_store

    @classmethod
    def get_default(cls) -> DynamicKnowledgeCatalog:
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path if isinstance(self.db_path, str) else str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    vendor TEXT,
                    family_model TEXT,
                    os_name TEXT,
                    version_build TEXT,
                    feature TEXT,
                    original_source TEXT,
                    source_uri TEXT,
                    page_section TEXT,
                    published_date TEXT,
                    consulted_date TEXT,
                    reliability_score REAL,
                    applicability_restrictions TEXT,
                    extracted_commands TEXT,
                    extracted_tables TEXT,
                    is_official INTEGER,
                    is_obsolete INTEGER,
                    superseded_by TEXT,
                    security_quarantined INTEGER,
                    quarantine_reason TEXT,
                    domain TEXT,
                    organization TEXT,
                    language_or_framework TEXT,
                    topic_category TEXT,
                    metadata_extra TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS domains (
                    domain_id TEXT PRIMARY KEY,
                    section_number INTEGER,
                    title TEXT NOT NULL,
                    topics TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_domain ON documents(domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_vendor ON documents(vendor)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_org ON documents(organization)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_lang ON documents(language_or_framework)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_official ON documents(is_official)")
            conn.commit()

    def _is_empty(self) -> bool:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM documents").fetchone()
            return bool(row["cnt"] == 0) if row else True

    def _seed_from_source(self, payload: Dict[str, Any]) -> None:
        raw_docs = payload.get("documents", [])
        for d in raw_docs:
            doc = KnowledgeDocument.from_dict(d)
            self.add_document(doc, sync_binary=False)
        for dom_id, spec in payload.get("domains", {}).items():
            self.register_domain(
                domain_id=dom_id,
                title=spec.get("title", dom_id),
                topics=spec.get("topics", []),
                section_number=spec.get("section_number"),
                metadata=spec.get("metadata"),
            )
        self._sync_all_to_binary_store()

    def _load_cache(self) -> None:
        self._in_memory_docs.clear()
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
            for r in rows:
                doc = self._row_to_doc(r)
                self._in_memory_docs[doc.doc_id] = doc
        self._sync_all_to_binary_store()

    def _row_to_doc(self, r: sqlite3.Row) -> KnowledgeDocument:
        extra = json.loads(r["metadata_extra"] or "{}")
        commands = json.loads(r["extracted_commands"] or "[]")
        tables = json.loads(r["extracted_tables"] or "[]")
        return KnowledgeDocument(
            doc_id=r["doc_id"],
            title=r["title"],
            content=r["content"],
            vendor=r["vendor"] or "",
            family_model=r["family_model"] or "",
            os_name=r["os_name"] or "",
            version_build=r["version_build"] or "",
            feature=r["feature"] or "",
            original_source=r["original_source"] or "",
            source_uri=r["source_uri"] or "",
            page_section=r["page_section"] or "",
            published_date=r["published_date"] or "",
            consulted_date=r["consulted_date"] or "",
            reliability_score=float(r["reliability_score"] or 0.0),
            applicability_restrictions=r["applicability_restrictions"] or "",
            extracted_commands=commands,
            extracted_tables=tables,
            is_official=bool(r["is_official"]),
            is_obsolete=bool(r["is_obsolete"]),
            superseded_by=r["superseded_by"] or "",
            security_quarantined=bool(r["security_quarantined"]),
            quarantine_reason=r["quarantine_reason"] or "",
            domain=r["domain"] or "routing",
            organization=r["organization"] or "",
            language_or_framework=r["language_or_framework"] or "",
            topic_category=r["topic_category"] or "general",
            metadata_extra=extra,
        )

    def _sync_all_to_binary_store(self) -> None:
        concept_ids: List[str] = []
        self._catalog_mem_system.semantic_nodes.clear()
        for doc in self._in_memory_docs.values():
            cid = f"DOC:{doc.doc_id}"
            concept_ids.append(cid)
            self._doc_memory_ids[doc.doc_id] = cid
            self._memory_id_to_doc_id[cid] = doc.doc_id
            self._catalog_mem_system.semantic_nodes[cid] = True

        store = self.binary_store
        if store is not None and concept_ids:
            store._ensure_knn_index()

    def add_document(self, doc: KnowledgeDocument, sync_binary: bool = True) -> KnowledgeDocument:
        """Adiciona ou atualiza documento no banco SQLite e no BinaryMachineMemoryStore."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, title, content, vendor, family_model, os_name, version_build,
                    feature, original_source, source_uri, page_section, published_date,
                    consulted_date, reliability_score, applicability_restrictions,
                    extracted_commands, extracted_tables, is_official, is_obsolete,
                    superseded_by, security_quarantined, quarantine_reason, domain,
                    organization, language_or_framework, topic_category, metadata_extra,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title,
                    content=excluded.content,
                    vendor=excluded.vendor,
                    family_model=excluded.family_model,
                    os_name=excluded.os_name,
                    version_build=excluded.version_build,
                    feature=excluded.feature,
                    original_source=excluded.original_source,
                    source_uri=excluded.source_uri,
                    page_section=excluded.page_section,
                    published_date=excluded.published_date,
                    consulted_date=excluded.consulted_date,
                    reliability_score=excluded.reliability_score,
                    applicability_restrictions=excluded.applicability_restrictions,
                    extracted_commands=excluded.extracted_commands,
                    extracted_tables=excluded.extracted_tables,
                    is_official=excluded.is_official,
                    is_obsolete=excluded.is_obsolete,
                    superseded_by=excluded.superseded_by,
                    security_quarantined=excluded.security_quarantined,
                    quarantine_reason=excluded.quarantine_reason,
                    domain=excluded.domain,
                    organization=excluded.organization,
                    language_or_framework=excluded.language_or_framework,
                    topic_category=excluded.topic_category,
                    metadata_extra=excluded.metadata_extra,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    doc.doc_id,
                    doc.title,
                    doc.content,
                    doc.vendor,
                    doc.family_model,
                    doc.os_name,
                    doc.version_build,
                    doc.feature,
                    doc.original_source,
                    doc.source_uri,
                    doc.page_section,
                    doc.published_date,
                    doc.consulted_date,
                    doc.reliability_score,
                    doc.applicability_restrictions,
                    json.dumps(doc.extracted_commands, ensure_ascii=False),
                    json.dumps(doc.extracted_tables, ensure_ascii=False),
                    1 if doc.is_official else 0,
                    1 if doc.is_obsolete else 0,
                    doc.superseded_by,
                    1 if doc.security_quarantined else 0,
                    doc.quarantine_reason,
                    doc.domain,
                    doc.organization,
                    doc.language_or_framework,
                    doc.topic_category,
                    json.dumps(doc.metadata_extra, ensure_ascii=False),
                ),
            )
            conn.commit()

        self._in_memory_docs[doc.doc_id] = doc
        if sync_binary:
            self._sync_all_to_binary_store()
            self._update_global_official_documents()
        return doc

    def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        if doc_id in self._in_memory_docs:
            return self._in_memory_docs[doc_id]
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if row:
                doc = self._row_to_doc(row)
                self._in_memory_docs[doc.doc_id] = doc
                return doc
        return None

    def delete_document(self, doc_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
            deleted = bool(cursor.rowcount > 0)
        self._in_memory_docs.pop(doc_id, None)
        self._sync_all_to_binary_store()
        self._update_global_official_documents()
        return deleted

    def list_documents(
        self,
        domain: Optional[str] = None,
        vendor: Optional[str] = None,
        organization: Optional[str] = None,
        language_or_framework: Optional[str] = None,
        is_official: Optional[bool] = None,
        include_quarantined: bool = False,
        limit: int = 1000,
    ) -> List[KnowledgeDocument]:
        """Consulta flexível com suporte a filtros dinâmicos."""
        results: List[KnowledgeDocument] = []
        for doc in self._in_memory_docs.values():
            if not include_quarantined and doc.security_quarantined:
                continue
            if domain and doc.domain.lower() != domain.strip().lower():
                continue
            if vendor and doc.vendor.lower() != vendor.strip().lower():
                continue
            if organization and doc.organization.lower() != organization.strip().lower():
                continue
            if language_or_framework and doc.language_or_framework.lower() != language_or_framework.strip().lower():
                continue
            if is_official is not None and doc.is_official != is_official:
                continue
            results.append(doc)
            if len(results) >= limit:
                break
        return results

    def list_official_documents(self) -> List[KnowledgeDocument]:
        return self.list_documents(is_official=True, include_quarantined=False)

    def register_domain(
        self,
        domain_id: str,
        title: str,
        topics: List[str],
        section_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        norm_id = domain_id.strip().lower()
        NETWORKING_KNOWLEDGE_DOMAINS.register(
            domain_id=norm_id,
            title=title,
            topics=topics,
            section_number=section_number,
            metadata=metadata,
        )
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO domains (domain_id, section_number, title, topics, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(domain_id) DO UPDATE SET
                    section_number=excluded.section_number,
                    title=excluded.title,
                    topics=excluded.topics,
                    metadata=excluded.metadata
                """,
                (
                    norm_id,
                    section_number or len(NETWORKING_KNOWLEDGE_DOMAINS),
                    title,
                    json.dumps(topics, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            conn.commit()

    def search_knn(
        self, query_or_vector: Union[str, np.ndarray], k: int = 5
    ) -> List[Tuple[KnowledgeDocument, float]]:
        """Busca semântica acelerada KNN via BinaryMachineMemoryStore."""
        store = self.binary_store
        if store is None:
            return []
        if isinstance(query_or_vector, str):
            vec = store._deterministic_concept_vector(query_or_vector)
        else:
            vec = query_or_vector
        raw_hits = store.fast_knn_lookup(vec, k=k)
        results: List[Tuple[KnowledgeDocument, float]] = []
        for cid, score in raw_hits:
            doc_id = self._memory_id_to_doc_id.get(cid)
            if doc_id and doc_id in self._in_memory_docs:
                results.append((self._in_memory_docs[doc_id], score))
        return results

    def verify_coverage(self, domain_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Valida dinamicamente a cobertura temática dos domínios especificados."""
        target_domains = domain_keys or list(NETWORKING_KNOWLEDGE_DOMAINS.keys())
        coverage: Dict[str, Any] = {}
        all_covered = True

        for domain_key in target_domains:
            norm_key = domain_key.strip().lower()
            spec = NETWORKING_KNOWLEDGE_DOMAINS.get(norm_key, {
                "section_number": 0,
                "title": norm_key.title(),
                "topics": [],
            })
            docs = self.list_documents(domain=norm_key, is_official=True)
            combined_text = " ".join(
                f"{d.title} {d.feature} {d.content}" for d in docs
            ).lower()
            missing_topics = [
                topic for topic in spec.get("topics", []) if topic.lower() not in combined_text
            ]
            domain_ok = len(docs) > 0 and len(missing_topics) == 0
            if not domain_ok:
                all_covered = False
            coverage[norm_key] = {
                "section_number": spec.get("section_number", 0),
                "title": spec.get("title", norm_key),
                "document_count": len(docs),
                "topics_required": spec.get("topics", []),
                "missing_topics": missing_topics,
                "fully_covered": domain_ok,
            }

        return {
            "total_domains": len(target_domains),
            "all_7_domains_covered": all_covered and len(target_domains) == 7,
            "domains": coverage,
        }

    def _update_global_official_documents(self) -> None:
        global OFFICIAL_INITIAL_DOCUMENTS
        OFFICIAL_INITIAL_DOCUMENTS.clear()
        OFFICIAL_INITIAL_DOCUMENTS.extend(self.list_official_documents())


# Alias canônico
PersistentKnowledgeCatalogEngine = DynamicKnowledgeCatalog

# Instância padrão gerenciada
_DEFAULT_CATALOG = DynamicKnowledgeCatalog.get_default()

# Lista oficial dinâmica e sincronizada (sem centenas de strings hardcoded!)
OFFICIAL_INITIAL_DOCUMENTS: List[KnowledgeDocument] = list(_DEFAULT_CATALOG.list_official_documents())


def get_documents_by_domain(domain: str) -> List[KnowledgeDocument]:
    """Retorna todos os documentos oficiais associados ao domínio solicitado."""
    return _DEFAULT_CATALOG.list_documents(domain=domain, is_official=True)


def verify_all_7_domains_coverage() -> Dict[str, Any]:
    """Valida cobertura temática dos 7 domínios de redes da Seção 6 no catálogo persistente."""
    return _DEFAULT_CATALOG.verify_coverage(["fundamentals", "routing", "mpls_services", "infrastructure", "security", "operations", "systems"])
