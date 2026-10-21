"""Canonical, vendor-neutral operational state models (OpenConfig inspired)."""

from __future__ import annotations

from datetime import datetime
from ipaddress import IPv4Address, IPv6Address, IPv4Network, IPv6Network, ip_address, ip_network
from typing import Literal, cast

from pydantic import ConfigDict, Field, IPvAnyAddress, IPvAnyNetwork, BaseModel


class CanonicalModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InterfaceState(CanonicalModel):
    device_id: str
    name: str
    admin_status: Literal["UP", "DOWN", "UNKNOWN"] = "UNKNOWN"
    oper_status: Literal["UP", "DOWN", "DORMANT", "UNKNOWN"] = "UNKNOWN"
    addresses: tuple[IPvAnyAddress | IPvAnyNetwork, ...] = ()
    mtu: int | None = Field(default=None, gt=0)
    in_octets: int | None = Field(default=None, ge=0)
    out_octets: int | None = Field(default=None, ge=0)
    in_errors: int | None = Field(default=None, ge=0)
    out_errors: int | None = Field(default=None, ge=0)
    observed_at: datetime | None = None


class BgpPeerState(CanonicalModel):
    device_id: str
    peer_address: IPv4Address | IPv6Address
    remote_as: int | None = Field(default=None, ge=1, le=4294967295)
    local_as: int | None = Field(default=None, ge=1, le=4294967295)
    state: str
    uptime_seconds: int | None = Field(default=None, ge=0)
    prefixes_received: int | None = Field(default=None, ge=0)
    last_error: str | None = None
    observed_at: datetime | None = None


class OpticalLaneState(CanonicalModel):
    device_id: str
    interface: str
    lane: int = Field(ge=0)
    rx_power_dbm: float | None = None
    tx_power_dbm: float | None = None
    rx_low_threshold_dbm: float | None = None
    tx_low_threshold_dbm: float | None = None
    wavelength_nm: float | None = Field(default=None, gt=0)
    alarms: tuple[str, ...] = ()
    observed_at: datetime | None = None


class RouteEntry(CanonicalModel):
    device_id: str
    prefix: IPv4Network | IPv6Network
    next_hop: IPv4Address | IPv6Address | None = None
    protocol: str
    preference: int | None = None
    metric: int | None = None
    vrf: str = "default"
    active: bool = True
    observed_at: datetime | None = None


def canonicalize_operational_state(device_id: str, state: dict[str, object]) -> dict[str, list[CanonicalModel]]:
    """Normalize common legacy/vendor-neutral collection keys into canonical records."""
    interfaces: list[CanonicalModel] = []
    for raw in state.get("interfaces", []):  # type: ignore[union-attr]
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        status = str(raw.get("status", "UNKNOWN")).upper()
        protocol = str(raw.get("protocol", status)).upper()
        addresses: list[IPvAnyAddress | IPvAnyNetwork] = []
        raw_address = raw.get("ip")
        if isinstance(raw_address, str) and raw_address:
            try:
                if "/" in raw_address:
                    addresses.append(ip_network(raw_address, strict=False))
                else:
                    addresses.append(ip_address(raw_address))
            except ValueError:
                pass
        interfaces.append(InterfaceState(device_id=device_id, name=str(raw["name"]),
            admin_status=cast(Literal["UP", "DOWN", "UNKNOWN"], status if status in {"UP", "DOWN"} else "UNKNOWN"),
            oper_status=cast(Literal["UP", "DOWN", "DORMANT", "UNKNOWN"], protocol if protocol in {"UP", "DOWN", "DORMANT"} else "UNKNOWN"),
            addresses=tuple(addresses), mtu=raw.get("mtu") if isinstance(raw.get("mtu"), int) else None,
            in_errors=raw.get("in_errors") if isinstance(raw.get("in_errors"), int) else None,
            out_errors=raw.get("out_errors") if isinstance(raw.get("out_errors"), int) else None))
    peers: list[CanonicalModel] = []
    for raw in state.get("bgp_peers", []):  # type: ignore[union-attr]
        if isinstance(raw, dict) and raw.get("peer_ip"):
            peers.append(BgpPeerState(device_id=device_id, peer_address=raw["peer_ip"],
                remote_as=raw.get("remote_as"), state=str(raw.get("state", "UNKNOWN")),
                prefixes_received=raw.get("prefixes_received"), last_error=raw.get("last_error")))
    optics: list[CanonicalModel] = []
    for raw in state.get("optical_transceivers", []):  # type: ignore[union-attr]
        if isinstance(raw, dict) and raw.get("interface"):
            optics.append(OpticalLaneState(device_id=device_id, interface=str(raw["interface"]),
                lane=int(raw.get("lane", 0)), rx_power_dbm=raw.get("rx_power_dbm"),
                tx_power_dbm=raw.get("tx_power_dbm"),
                rx_low_threshold_dbm=raw.get("rx_low_threshold_dbm"),
                wavelength_nm=raw.get("wavelength_nm")))
    return {"interfaces": interfaces, "bgp_peers": peers, "optical_lanes": optics, "routes": []}
