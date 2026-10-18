"""Pacote knowledge.ingestion da Veronica."""

from knowledge.ingestion.ingestor import (
    scrub_secrets,
    detect_prompt_injection,
    extract_tables_and_commands,
    extract_docx_content,
    extract_pdf_content,
    normalize_cli_config_content,
    normalize_format_content,
    chunk_large_document,
    OfficialDocumentationSearchConnector,
    search_live_web_and_rfc,
    KnowledgeIngestor,
)

__all__ = [
    "scrub_secrets",
    "detect_prompt_injection",
    "extract_tables_and_commands",
    "extract_docx_content",
    "extract_pdf_content",
    "normalize_cli_config_content",
    "normalize_format_content",
    "chunk_large_document",
    "OfficialDocumentationSearchConnector",
    "search_live_web_and_rfc",
    "KnowledgeIngestor",
]

