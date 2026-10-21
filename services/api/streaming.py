"""Transport helpers for the API's SSE and WebSocket streams."""

from __future__ import annotations

import json
from typing import Any

try:  # orjson is an optional runtime accelerator in development environments.
    import orjson
except ImportError:  # pragma: no cover - exercised only in minimal installations
    orjson = None


def json_bytes(value: Any) -> bytes:
    """Serialize API payloads, including numpy scalar/array values."""
    if orjson is not None:
        option = getattr(orjson, "OPT_SERIALIZE_NUMPY", 0)
        try:
            return orjson.dumps(value, option=option)
        except (TypeError, ValueError):
            pass
    return json.dumps(value, ensure_ascii=False, default=_json_default).encode("utf-8")


def sse_frame(value: Any, event: str | None = None) -> str:
    payload = json_bytes(value).decode("utf-8").replace("\r", "").replace("\n", "\\n")
    event_line = f"event: {event}\n" if event else ""
    return f"{event_line}data: {payload}\n\n"


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
