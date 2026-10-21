"""Deterministic safety invariants for post-change operational state."""

from __future__ import annotations

from typing import Any


def compare_operational_invariants(
    baseline: dict[str, Any], post: dict[str, Any], *, max_cpu_increase: float = 20.0
) -> dict[str, Any]:
    """Compare available BGP, route, optical/error, CPU and service invariants.

    Missing optional telemetry is reported as unavailable; it is not invented.
    Required checks (BGP when present, CPU when present, service health) fail closed.
    """
    checks: dict[str, dict[str, Any]] = {}

    def established(state: dict[str, Any]) -> int:
        return sum(
            str(peer.get("state", "")).lower() == "established"
            for peer in state.get("bgp_peers", [])
        )

    if "bgp_peers" in baseline:
        before, after = established(baseline), established(post)
        checks["bgp_established"] = {"passed": after >= before, "baseline": before, "post": after}

    for key in ("route_count", "routes"):
        if key in baseline:
            before = len(baseline[key]) if isinstance(baseline[key], (list, tuple, set)) else baseline[key]
            after_value = post.get(key)
            after = len(after_value) if isinstance(after_value, (list, tuple, set)) else after_value
            checks["routes"] = {
                "passed": after is not None and after >= before,
                "baseline": before,
                "post": after,
            }
            break

    def error_count(state: dict[str, Any]) -> int:
        return sum(int(interface.get("crc_errors", 0)) for interface in state.get("interfaces", []))

    if "interfaces" in baseline:
        before, after = error_count(baseline), error_count(post)
        checks["interface_errors"] = {"passed": after <= before, "baseline": before, "post": after}

    if "optical_transceivers" in baseline:
        alarms_before = sum(
            "ALARM" in str(lane.get("status", "")).upper()
            for lane in baseline.get("optical_transceivers", [])
        )
        alarms_after = sum(
            "ALARM" in str(lane.get("status", "")).upper()
            for lane in post.get("optical_transceivers", [])
        )
        checks["optical_alarms"] = {
            "passed": alarms_after <= alarms_before,
            "baseline": alarms_before,
            "post": alarms_after,
        }

    if "cpu_percent" in baseline:
        before, after = float(baseline["cpu_percent"]), float(post.get("cpu_percent", 100.0))
        checks["cpu"] = {
            "passed": after < 95.0 and after <= before + max_cpu_increase,
            "baseline": before,
            "post": after,
        }

    checks["services"] = {
        "passed": bool(post.get("services_healthy", False)),
        "post": bool(post.get("services_healthy", False)),
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    return {"passed": not failed, "checks": checks, "failed_invariants": failed}

