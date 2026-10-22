"""Pacote networking.diagnostics da Veronica."""

from networking.diagnostics.engine import NetworkDiagnosticEngine
from networking.diagnostics.fault_injector import (
    FaultInjector,
    SyntheticFaultInjector,
    SyntheticFaultScenario,
    InjectedFaultRecord,
)

__all__ = [
    "NetworkDiagnosticEngine",
    "FaultInjector",
    "SyntheticFaultInjector",
    "SyntheticFaultScenario",
    "InjectedFaultRecord",
]
