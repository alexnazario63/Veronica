"""Pacote networking.protocols da Veronica."""

from networking.protocols.analyzers import (
    PROTOCOL_DOMAIN_REGISTRY,
    analyze_protocol_anomalies,
    evaluate_all_7_protocol_domains,
)

__all__ = [
    "PROTOCOL_DOMAIN_REGISTRY",
    "analyze_protocol_anomalies",
    "evaluate_all_7_protocol_domains",
]

