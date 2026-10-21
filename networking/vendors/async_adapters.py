"""Credential-injected async vendor adapter with read-only cascade defaults.

This module deliberately does not create transport sessions or discover secrets.
Callers inject protocol clients after obtaining credentials through their own vault.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Mapping, Protocol, Sequence


class AsyncTransport(Protocol):
    async def execute(self, operation: str, device_id: str, **kwargs: Any) -> Any: ...


class AsyncVendorAdapter(ABC):
    """Sixteen adapter operations; live reads cascade gNMI -> NETCONF -> SSH."""

    vendor: str = "generic"
    os_family: str = "generic"
    ntc_platform: str | None = None

    def __init__(
        self,
        *,
        gnmi: AsyncTransport | None = None,
        netconf: AsyncTransport | None = None,
        ssh: AsyncTransport | None = None,
    ) -> None:
        self._transports = (("gnmi", gnmi), ("netconf", netconf), ("ssh", ssh))

    async def _read(self, operation: str, device_id: str, **kwargs: Any) -> Any:
        errors: list[str] = []
        for protocol, client in self._transports:
            if client is None:
                continue
            try:
                result = await client.execute(operation, device_id, **kwargs)
                return {"transport": protocol, "data": result}
            except (TimeoutError, ConnectionError, OSError) as exc:
                errors.append(f"{protocol}: {type(exc).__name__}: {exc}")
        detail = "; ".join(errors) if errors else "no transport clients configured"
        raise ConnectionError(f"Read-only operation {operation!r} unavailable for {device_id}: {detail}")

    async def discover_device(self, device_id: str) -> Any:
        return await self._read("discover_device", device_id)

    async def get_capabilities(self, device_id: str) -> Any:
        return await self._read("get_capabilities", device_id)

    async def get_version(self, device_id: str) -> Any:
        return await self._read("get_version", device_id)

    async def get_running_config(self, device_id: str) -> Any:
        return await self._read("get_running_config", device_id)

    async def get_operational_state(self, device_id: str) -> Any:
        return await self._read("get_operational_state", device_id)

    async def get_topology_neighbors(self, device_id: str) -> Any:
        return await self._read("get_topology_neighbors", device_id)

    async def collect_diagnostics(self, device_id: str, focus: str = "all") -> Any:
        result = await self._read("collect_diagnostics", device_id, focus=focus, safety="read_only")
        return normalize_cli_bundle(result, ntc_platform=self.ntc_platform)

    async def validate_commands(self, device_id: str, commands: Sequence[str]) -> Any:
        return await self._read("validate_commands", device_id, commands=list(commands))

    async def generate_candidate(self, device_id: str, proposed_commands: Sequence[str]) -> Any:
        # Planning is local and does not apply candidate configuration to a device.
        return {"device_id": device_id, "commands": list(proposed_commands), "applied": False}

    async def compare_configurations(self, running_cfg: str, candidate_cfg: str) -> Any:
        import difflib

        return {"diff": "".join(difflib.unified_diff(
            running_cfg.splitlines(True), candidate_cfg.splitlines(True),
            fromfile="baseline", tofile="candidate")), "applied": False}

    async def check_preconditions(self, device_id: str, commands: Sequence[str]) -> Any:
        state = await self.get_operational_state(device_id)
        return {"device_id": device_id, "commands": list(commands), "state_snapshot": state, "write_authorized": False}

    async def create_backup(self, device_id: str) -> Any:
        # Returns an in-memory read snapshot only; it does not trigger device-side backup commands.
        return await self.get_running_config(device_id)

    async def prepare_rollback(self, device_id: str, backup: Mapping[str, Any], commands: Sequence[str]) -> Any:
        return {"device_id": device_id, "backup": dict(backup), "commands": list(commands), "prepared_only": True}

    async def apply_change(self, device_id: str, commands: Sequence[str], approval_token: str | None = None,
                           expected_config_hash: str | None = None) -> Any:
        raise PermissionError("Device writes are disabled in networking adapters; require the safety change-control executor.")

    async def verify_change(self, device_id: str, baseline_state: Mapping[str, Any], post_state: Mapping[str, Any]) -> Any:
        return {"device_id": device_id, "baseline": dict(baseline_state), "post_state": dict(post_state), "compared": True}

    async def rollback_change(self, device_id: str, rollback_plan: Mapping[str, Any]) -> Any:
        raise PermissionError("Device rollback execution is disabled in networking adapters; require the safety change-control executor.")


class _VendorFamilyAdapter(AsyncVendorAdapter):
    pass


class HuaweiVRPAdapter(_VendorFamilyAdapter):
    vendor, os_family = "Huawei", "VRP"
    ntc_platform = "huawei_vrp"


class CiscoIOSXEAdapter(_VendorFamilyAdapter):
    vendor, os_family = "Cisco", "IOS-XE"
    ntc_platform = "cisco_ios"


class CiscoIOSXRAdapter(_VendorFamilyAdapter):
    vendor, os_family = "Cisco", "IOS-XR"
    ntc_platform = "cisco_xr"


class CiscoNXOSAdapter(_VendorFamilyAdapter):
    vendor, os_family = "Cisco", "NX-OS"
    ntc_platform = "cisco_nxos"


class JuniperJunosAdapter(_VendorFamilyAdapter):
    vendor, os_family = "Juniper", "Junos"
    ntc_platform = "juniper_junos"


class MikroTikRouterOSAdapter(_VendorFamilyAdapter):
    vendor, os_family = "MikroTik", "RouterOS"
    ntc_platform = "mikrotik_routeros"


class NokiaSROSAdapter(_VendorFamilyAdapter):
    vendor, os_family = "Nokia", "SR OS"
    ntc_platform = "nokia_sros"


class AristaEOSAdapter(_VendorFamilyAdapter):
    vendor, os_family = "Arista", "EOS"
    ntc_platform = "arista_eos"


class FortinetFortiOSAdapter(_VendorFamilyAdapter):
    vendor, os_family = "Fortinet", "FortiOS"
    ntc_platform = "fortinet"


def normalize_cli_bundle(bundle: Any, *, ntc_platform: str | None = None,
                         ttp_template: str | None = None) -> Any:
    """Best-effort structured parsing; retain raw evidence when optional parsers lack templates."""
    if not isinstance(bundle, Mapping):
        return bundle
    data = bundle.get("data")
    if not isinstance(data, Mapping):
        return bundle
    outputs = data.get("cli_outputs")
    if not isinstance(outputs, Mapping):
        return bundle
    parsed: dict[str, Any] = {}
    for command, output in outputs.items():
        if not isinstance(output, str):
            parsed[str(command)] = output
            continue
        value: Any = None
        if ttp_template:
            try:
                from ttp import ttp

                parser = ttp(data=output, template=ttp_template)
                parser.parse()
                value = parser.result()
            except ImportError:
                pass
            except Exception:
                value = None
        if value is None and ntc_platform:
            try:
                from ntc_templates.parse import parse_output

                value = parse_output(platform=ntc_platform, command=str(command), data=output)
            except (ImportError, ValueError, OSError):
                value = None
        parsed[str(command)] = value if value is not None else {"raw": output, "parsed": False}
    return {**bundle, "data": {**data, "parsed_cli_outputs": parsed}}


def audit_hierarchical_config(config: str) -> dict[str, Any]:
    """Return parser-backed hierarchy when ciscoconfparse2 is installed, else raw lines."""
    try:
        from ciscoconfparse2 import CiscoConfParse

        parsed = CiscoConfParse(config.splitlines(), syntax="ios")
        # ciscoconfparse2 exposes ConfigObjs at runtime, but its current type
        # stubs omit the property.
        config_objects = getattr(parsed, "ConfigObjs", config.splitlines())
        return {"available": True, "syntax": "ios", "lines": [str(line) for line in config_objects]}
    except ImportError:
        return {"available": False, "reason": "ciscoconfparse2 is not installed", "lines": config.splitlines()}
