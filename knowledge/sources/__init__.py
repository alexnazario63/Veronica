"""Pacote knowledge.sources da Veronica."""

from knowledge.sources.catalog import (
    KnowledgeDocument,
    NETWORKING_KNOWLEDGE_DOMAINS,
    KNOWLEDGE_DOMAINS,
    OFFICIAL_INITIAL_DOCUMENTS,
    DynamicDomainRegistry,
    DynamicKnowledgeCatalog,
    PersistentKnowledgeCatalogEngine,
    get_documents_by_domain,
    verify_all_7_domains_coverage,
)

__all__ = [
    "KnowledgeDocument",
    "NETWORKING_KNOWLEDGE_DOMAINS",
    "KNOWLEDGE_DOMAINS",
    "OFFICIAL_INITIAL_DOCUMENTS",
    "DynamicDomainRegistry",
    "DynamicKnowledgeCatalog",
    "PersistentKnowledgeCatalogEngine",
    "get_documents_by_domain",
    "verify_all_7_domains_coverage",
]
