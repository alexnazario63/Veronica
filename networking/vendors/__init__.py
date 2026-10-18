"""Pacote networking.vendors da Veronica."""

from networking.vendors.adapters import (
    PlatformMatrixEntry,
    COMPATIBILITY_MATRIX,
    BaseVendorAdapter,
    DigitalTwinNetworkAdapter,
)

__all__ = [
    "PlatformMatrixEntry",
    "COMPATIBILITY_MATRIX",
    "BaseVendorAdapter",
    "DigitalTwinNetworkAdapter",
]
