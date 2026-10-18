"""Pacote networking — Engenharia, Protocolos, Topologia e Diagnóstico de Redes da Veronica."""

from networking.vendors import (
    PlatformMatrixEntry,
    COMPATIBILITY_MATRIX,
    BaseVendorAdapter,
    DigitalTwinNetworkAdapter,
)
from networking.topology import create_default_network_inventory, NetworkTopologyGraph
from networking.protocols import analyze_protocol_anomalies
from networking.diagnostics import NetworkDiagnosticEngine

__all__ = [
    "PlatformMatrixEntry",
    "COMPATIBILITY_MATRIX",
    "BaseVendorAdapter",
    "DigitalTwinNetworkAdapter",
    "create_default_network_inventory",
    "NetworkTopologyGraph",
    "analyze_protocol_anomalies",
    "NetworkDiagnosticEngine",
]
