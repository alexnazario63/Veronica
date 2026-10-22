"""Incremental JSON syntax, lightweight JSON Schema, and Agent Grammar constrained decoding.

Supports automatic grammatical transitions between:
- Thought blocks ([THOUGHT] ... [/THOUGHT])
- Tool calls ([TOOL_CALL] ... [/TOOL_CALL] validated against tool input_schema)
- Artifact patch generation ([ARTIFACT] ... [/ARTIFACT] for ArtifactStore)
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")

DEFAULT_ARTIFACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["version"],
    "properties": {
        "id": {"type": "string"},
        "artifact_id": {"type": "string"},
        "version": {"type": "integer"},
        "type": {"type": "string"},
        "patch": {"type": "object"},
        "delta": {"type": "string"},
        "content": {"type": "string"},
        "baseline_content": {"type": "string"},
        "proposed_content": {"type": "string"},
        "blast_radius_graph": {"type": "object"},
        "sha256_hash": {"type": "string"},
    },
}


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
    if not schema:
        return True
    expected = schema.get("type")
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
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
    """
    Decodificador autoregressivo com restrições gramaticais e de esquema JSON.
    Controla transições automáticas entre blocos [THOUGHT], [TOOL_CALL] e [ARTIFACT].
    """

    def __init__(
        self,
        schema_or_tokenizer: Any,
        tokenizer_or_schema: Any = None,
        tools: Optional[Dict[str, Any]] = None,
        artifact_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(schema_or_tokenizer, dict):
            self.schema: Dict[str, Any] = schema_or_tokenizer
            self.tokenizer: Any = tokenizer_or_schema
        else:
            self.tokenizer = schema_or_tokenizer
            self.schema = tokenizer_or_schema if isinstance(tokenizer_or_schema, dict) else {}

        self.tools: Dict[str, Any] = tools or {}
        self.artifact_schema: Dict[str, Any] = artifact_schema or DEFAULT_ARTIFACT_SCHEMA

    def get_phase(self, text: str) -> str:
        """
        Determina o estado gramatical do fluxo do agente:
        - 'THOUGHT': dentro de [THOUGHT] ... [/THOUGHT]
        - 'ACTION': dentro de [TOOL_CALL] ... [/TOOL_CALL]
        - 'ARTIFACT': dentro de [ARTIFACT] ... [/ARTIFACT]
        - 'ANSWER': fora de blocos especiais
        """
        t_open = text.rfind("[THOUGHT]")
        t_close = text.rfind("[/THOUGHT]")
        c_open = text.rfind("[TOOL_CALL]")
        c_close = text.rfind("[/TOOL_CALL]")
        a_open = text.rfind("[ARTIFACT]")
        a_close = text.rfind("[/ARTIFACT]")

        last_open = max(t_open, c_open, a_open)
        if last_open == -1:
            return "ANSWER"
        if last_open == t_open and t_open > t_close:
            return "THOUGHT"
        if last_open == c_open and c_open > c_close:
            return "ACTION"
        if last_open == a_open and a_open > a_close:
            return "ARTIFACT"
        return "ANSWER"

    def _get_tool_schema(self, inner_json: str) -> Dict[str, Any]:
        """Extrai o schema apropriado com base no nome da ferramenta em decodificação."""
        match = re.search(r'"(?:name|tool)"\s*:\s*"([^"]+)"', inner_json)
        if match:
            tool_name = match.group(1)
            if tool_name in self.tools:
                tool_def = self.tools[tool_name]
                arg_schema = (
                    tool_def.get("input_schema", tool_def)
                    if isinstance(tool_def, dict)
                    else getattr(tool_def, "input_schema", {})
                )
                return {
                    "type": "object",
                    "required": ["arguments"],
                    "properties": {
                        "name": {"type": "string"},
                        "tool": {"type": "string"},
                        "arguments": arg_schema,
                        "parameters": arg_schema,
                    },
                }

        tool_names = list(self.tools.keys())
        name_prop: Dict[str, Any] = {"type": "string"}
        if tool_names:
            name_prop["enum"] = tool_names
        return {
            "type": "object",
            "properties": {
                "name": name_prop,
                "tool": name_prop,
                "arguments": {"type": "object"},
                "parameters": {"type": "object"},
            },
        }

    def allows(self, prefix: str) -> bool:
        """Verifica se o prefixo de texto mantém sintaxe e restrições de esquema válidas."""
        # Se contiver tags de ação/artefato, avalia a seção interna aberta
        c_open = prefix.rfind("[TOOL_CALL]")
        c_close = prefix.rfind("[/TOOL_CALL]")
        if c_open != -1 and c_open > c_close:
            inner = prefix[c_open + len("[TOOL_CALL]"):]
            schema = self._get_tool_schema(inner)
            return self._allows_json_prefix(inner, schema)

        a_open = prefix.rfind("[ARTIFACT]")
        a_close = prefix.rfind("[/ARTIFACT]")
        if a_open != -1 and a_open > a_close:
            inner = prefix[a_open + len("[ARTIFACT]"):]
            return self._allows_json_prefix(inner, self.artifact_schema)

        # Prefixo JSON puro ou raiz
        return self._allows_json_prefix(prefix, self.schema)

    def _allows_json_prefix(self, prefix: str, schema: Dict[str, Any]) -> bool:
        if not _json_prefix_is_possible(prefix):
            return False
        try:
            value = json.loads(prefix)
        except (json.JSONDecodeError, ValueError):
            return True
        return _schema_matches(value, schema)

    def mask_logits(self, logits: Any, prefix: str = "") -> Any:
        """Mascara tokens do vocabulário que geram JSON sintaticamente ou esquematicamente inválido."""
        scores = np.asarray(logits).copy()
        if self.tokenizer is None:
            return scores

        vsize = len(scores)
        for token_id, token in getattr(self.tokenizer, "id_to_token", {}).items():
            if token_id >= vsize:
                continue
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

    def filter_logits(self, prefix_tokens: List[int], logits: np.ndarray) -> np.ndarray:
        """
        Filtra os logits autoregressivos conforme a gramática do agente e o esquema ativo.
        Assinatura conforme INTERFACES.md Seção 1.4.
        """
        scores = np.asarray(logits, dtype=np.float64).copy()
        if self.tokenizer is None:
            return scores

        vsize = len(scores)
        prefix_text = self.tokenizer.decode(prefix_tokens, skip_special_tokens=False) if hasattr(self.tokenizer, "decode") else ""
        phase = self.get_phase(prefix_text)

        if phase == "ACTION":
            c_open = prefix_text.rfind("[TOOL_CALL]")
            inner = prefix_text[c_open + len("[TOOL_CALL]"):]
            schema = self._get_tool_schema(inner)
            return self._mask_block_tokens(scores, prefix_text, inner, schema, close_token="[/TOOL_CALL]")

        elif phase == "ARTIFACT":
            a_open = prefix_text.rfind("[ARTIFACT]")
            inner = prefix_text[a_open + len("[ARTIFACT]"):]
            return self._mask_block_tokens(scores, prefix_text, inner, self.artifact_schema, close_token="[/ARTIFACT]")

        elif phase == "THOUGHT":
            # Dentro do pensamento, bloqueia abertura precipitada de novos blocos antes de [/THOUGHT]
            for tid, token in getattr(self.tokenizer, "id_to_token", {}).items():
                if tid < vsize and token in ("[TOOL_CALL]", "[ARTIFACT]"):
                    scores[tid] = -np.inf
            return scores

        else:
            # Fora de blocos especiais: permite texto natural ou tags de transição
            if self.schema and (prefix_text.strip().startswith("{") or prefix_text.strip().startswith("[")):
                return self.mask_logits(scores, prefix_text)
            return scores

    def _mask_block_tokens(
        self,
        scores: np.ndarray,
        full_prefix: str,
        inner_json: str,
        schema: Dict[str, Any],
        close_token: str,
    ) -> np.ndarray:
        vsize = len(scores)
        for tid, token in getattr(self.tokenizer, "id_to_token", {}).items():
            if tid >= vsize:
                continue
            if token == close_token:
                # Permite fechar o bloco apenas se o JSON interno for um valor completo e válido
                try:
                    val = json.loads(inner_json)
                    if not _schema_matches(val, schema):
                        scores[tid] = -np.inf
                except Exception:
                    scores[tid] = -np.inf
                continue
            if token.startswith("[") and token.endswith("]"):
                scores[tid] = -np.inf
                continue
            if not self._allows_json_prefix(inner_json + token, schema):
                scores[tid] = -np.inf
        return scores

    def is_valid_terminal(self, tokens: List[int]) -> bool:
        """
        Verifica se a sequência de tokens está em um estado terminal válido.
        Assinatura conforme INTERFACES.md Seção 1.4.
        """
        if self.tokenizer is None:
            return True
        text = self.tokenizer.decode(tokens, skip_special_tokens=False) if hasattr(self.tokenizer, "decode") else ""
        phase = self.get_phase(text)
        if phase in ("ACTION", "ARTIFACT", "THOUGHT"):
            return False

        # Valida blocos de ferramentas fechados
        for match in re.finditer(r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]", text, re.DOTALL):
            content = match.group(1).strip()
            try:
                val = json.loads(content)
                name = val.get("name") or val.get("tool")
                if not name:
                    return False
                if self.tools and name in self.tools:
                    tool = self.tools[name]
                    arg_schema = (
                        tool.get("input_schema", tool)
                        if isinstance(tool, dict)
                        else getattr(tool, "input_schema", {})
                    )
                    args = val.get("arguments") or val.get("parameters") or {}
                    if not _schema_matches(args, arg_schema):
                        return False
            except Exception:
                return False

        # Valida blocos de artefato fechados
        for match in re.finditer(r"\[ARTIFACT\](.*?)\[/ARTIFACT\]", text, re.DOTALL):
            content = match.group(1).strip()
            try:
                val = json.loads(content)
                if not _schema_matches(val, self.artifact_schema):
                    return False
            except Exception:
                return False

        # Valida JSON puro se o esquema tiver sido configurado diretamente
        if self.schema and not ("[" in text and "]" in text):
            try:
                val = json.loads(text.strip())
                return _schema_matches(val, self.schema)
            except Exception:
                return False

        return True

    def create_artifact_patch(
        self,
        artifact_id: str,
        version: int,
        patch: Dict[str, Any],
        patch_type: str = "json_patch",
    ) -> Dict[str, Any]:
        """Cria e valida um payload de patch estruturado para o ArtifactStore."""
        payload: Dict[str, Any] = {
            "id": str(artifact_id),
            "artifact_id": str(artifact_id),
            "version": int(version),
            "type": patch_type,
            "patch": patch,
        }
        return payload

    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Valida argumentos de ferramenta contra o input_schema registrado."""
        if tool_name not in self.tools:
            return True
        tool = self.tools[tool_name]
        arg_schema = (
            tool.get("input_schema", tool)
            if isinstance(tool, dict)
            else getattr(tool, "input_schema", {})
        )
        return _schema_matches(arguments, arg_schema)
