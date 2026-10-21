"""Pacote knowledge — Sistema de Conhecimento Especializado da Veronica."""

from knowledge.sources import (
    KnowledgeDocument,
    NETWORKING_KNOWLEDGE_DOMAINS,
    OFFICIAL_INITIAL_DOCUMENTS,
    get_documents_by_domain,
    verify_all_7_domains_coverage,
)
from knowledge.ingestion import (
    scrub_secrets,
    detect_prompt_injection,
    extract_tables_and_commands,
    extract_docx_content,
    extract_pdf_content,
    normalize_cli_config_content,
    normalize_format_content,
    chunk_large_document,
    OfficialDocumentationSearchConnector,
    KnowledgeIngestor,
    SanitizedUntrustedText,
    sanitize_untrusted_text,
    sanitize_device_output,
)
from knowledge.retrieval import HybridRAGEngine

__all__ = [
    "KnowledgeDocument",
    "NETWORKING_KNOWLEDGE_DOMAINS",
    "OFFICIAL_INITIAL_DOCUMENTS",
    "get_documents_by_domain",
    "verify_all_7_domains_coverage",
    "scrub_secrets",
    "detect_prompt_injection",
    "extract_tables_and_commands",
    "extract_docx_content",
    "extract_pdf_content",
    "normalize_cli_config_content",
    "normalize_format_content",
    "chunk_large_document",
    "OfficialDocumentationSearchConnector",
    "KnowledgeIngestor",
    "SanitizedUntrustedText",
    "sanitize_untrusted_text",
    "sanitize_device_output",
    "HybridRAGEngine",
]

