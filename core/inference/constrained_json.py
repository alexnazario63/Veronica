"""Incremental JSON syntax and lightweight JSON Schema constrained decoding."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List


_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")


def _json_prefix_is_possible(text: str) -> bool:
    """Check whether text is a valid prefix of a JSON value using a small FSM."""
    stack: List[Dict[str, str]] = []
    root = "value"
    in_string = False
    escaped = False
    string_role = "value"
    primitive = ""
    i = 0

    def value_done() -> None:
        nonlocal root
        if stack:
            stack[-1]["state"] = "comma_or_end"
        else:
            root = "done"

    while i < len(text):
        ch = text[i]
        if in_string:
            if escaped:
                if ch not in '"\\/bfnrtu':
                    return False
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                if string_role == "key":
                    if not stack or stack[-1]["kind"] != "object" or stack[-1]["state"] not in {"key", "key_or_end"}:
                        return False
                    stack[-1]["state"] = "colon"
                else:
                    value_done()
            elif ord(ch) < 0x20:
                return False
            i += 1
            continue
        if primitive:
            if ch in ",]} \t\r\n":
                if primitive in ("true", "false", "null") or _NUMBER.fullmatch(primitive):
                    primitive = ""
                    value_done()
                    continue  # process delimiter again
                return False
            primitive += ch
            if not ("true".startswith(primitive) or "false".startswith(primitive)
                    or "null".startswith(primitive) or re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?\d*)?", primitive)):
                return False
            i += 1
            continue

        state = stack[-1]["state"] if stack else root
        if ch in " \t\r\n":
            i += 1
            continue
        if state == "done":
            return False
        if state == "colon":
            if ch != ":":
                return False
            stack[-1]["state"] = "value"
        elif state == "comma_or_end":
            kind = stack[-1]["kind"]
            if ch == ",":
                stack[-1]["state"] = "key" if kind == "object" else "value"
            elif (ch == "}" and kind == "object") or (ch == "]" and kind == "array"):
                stack.pop()
                value_done()
            else:
                return False
        elif state in ("value", "value_or_end", "key", "key_or_end"):
            if state == "key_or_end" and ch == "}":
                stack.pop()
                value_done()
            elif state == "value_or_end" and ch == "]":
                stack.pop()
                value_done()
            elif state in ("key", "key_or_end") and ch == '"':
                in_string, string_role = True, "key"
            elif state in ("value", "value_or_end") and ch == '"':
                in_string, string_role = True, "value"
            elif state in ("value", "value_or_end") and ch == "{":
                stack.append({"kind": "object", "state": "key_or_end"})
            elif state in ("value", "value_or_end") and ch == "[":
                stack.append({"kind": "array", "state": "value_or_end"})
            elif state in ("value", "value_or_end") and ch in "-0123456789tfn":
                primitive = ch
                if not ("true".startswith(primitive) or "false".startswith(primitive)
                        or "null".startswith(primitive) or re.fullmatch(r"-?(?:0|[1-9]\d*)?", primitive)):
                    return False
            else:
                return False
        else:
            return False
        i += 1
    if in_string or escaped:
        return True
    if primitive:
        return ("true".startswith(primitive) or "false".startswith(primitive)
                or "null".startswith(primitive)
                or re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+-]?\d*)?", primitive) is not None)
    return root == "value" or bool(stack) or root == "done"


def _schema_matches(value: Any, schema: Dict[str, Any]) -> bool:
    expected = schema.get("type")
    type_map = {"object": dict, "array": list, "string": str, "integer": int,
                "number": (int, float), "boolean": bool, "null": type(None)}
    if expected in type_map:
        cls = type_map[expected]
        if not isinstance(value, cls) or (expected == "integer" and isinstance(value, bool)):
            return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        if any(key not in value for key in schema.get("required", [])):
            return False
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
        return all(k not in props or _schema_matches(v, props[k]) for k, v in value.items())
    if isinstance(value, list) and "items" in schema:
        return all(_schema_matches(item, schema["items"]) for item in value)
    return True


class ConstrainedJSONDecoder:
    """Masks tokens that cannot continue JSON or violate the supplied schema."""

    def __init__(self, tokenizer: Any, schema: Dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.schema = schema

    def allows(self, prefix: str) -> bool:
        if not _json_prefix_is_possible(prefix):
            return False
        try:
            value = json.loads(prefix)
        except (json.JSONDecodeError, ValueError):
            return True
        return _schema_matches(value, self.schema)

    def mask_logits(self, logits: Any, prefix: str = "") -> Any:
        import numpy as np
        scores = np.asarray(logits).copy()
        for token_id, token in self.tokenizer.id_to_token.items():
            if token.startswith("[") and token.endswith("]"):
                if token != "[/TOOL_CALL]" or not self.allows(prefix):
                    scores[token_id] = -np.inf
                continue
            extension = token
            if not self.allows(prefix + extension):
                scores[token_id] = -np.inf
        if not np.isfinite(scores).any():
            raise ValueError("No vocabulary token can continue valid JSON for the requested schema")
        return scores
