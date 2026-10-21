"""Pacote networking.vendors da Veronica."""

from networking.vendors.adapters import (
    PlatformMatrixEntry,
    COMPATIBILITY_MATRIX,
    BaseVendorAdapter,
    DigitalTwinNetworkAdapter,
)
from networking.vendors.models import BgpPeerState, InterfaceState, OpticalLaneState, RouteEntry
from networking.vendors.async_adapters import (
    AsyncVendorAdapter, HuaweiVRPAdapter, CiscoIOSXEAdapter, CiscoIOSXRAdapter,
    CiscoNXOSAdapter, JuniperJunosAdapter, MikroTikRouterOSAdapter, NokiaSROSAdapter,
    AristaEOSAdapter, FortinetFortiOSAdapter, audit_hierarchical_config, normalize_cli_bundle,
)

__all__ = [
    "PlatformMatrixEntry",
    "COMPATIBILITY_MATRIX",
    "BaseVendorAdapter",
    "DigitalTwinNetworkAdapter",
    "InterfaceState", "BgpPeerState", "OpticalLaneState", "RouteEntry",
    "AsyncVendorAdapter", "HuaweiVRPAdapter", "CiscoIOSXEAdapter", "CiscoIOSXRAdapter",
    "CiscoNXOSAdapter", "JuniperJunosAdapter", "MikroTikRouterOSAdapter", "NokiaSROSAdapter",
    "AristaEOSAdapter", "FortinetFortiOSAdapter", "audit_hierarchical_config", "normalize_cli_bundle",
]
