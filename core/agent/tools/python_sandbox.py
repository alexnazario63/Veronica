"""Python Sandbox Tool for safe dynamic execution in isolated subprocess."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from typing import Any, Dict, Mapping, Optional, Union

from core.contracts import ToolSafetyTier


class PythonSandboxTool:
    """Executa código Python dinâmico em subprocesso isolado com timeout e captura completa."""

    name: str = "python_sandbox"
    description: str = (
        "Executes dynamic Python code in an isolated subprocess with timeout (10s), "
        "capturing stdout, stderr, exception trace, and return values. "
        "Used for mathematical calculations, algorithms, hypothesis validation, simulations, and unit tests."
    )
    tier: ToolSafetyTier = ToolSafetyTier.TIER_0_READ_ONLY
    is_safe_read_only: bool = True
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code or script to execute in sandbox",
            },
            "timeout": {
                "type": "number",
                "description": "Execution timeout in seconds (default 10.0s)",
            },
        },
        "required": ["code"],
    }
    output_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error", "timeout"]},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "returncode": {"type": "integer"},
            "output": {"type": "string"},
            "result": {"type": "any"},
            "execution_time_ms": {"type": "number"},
            "error": {"type": "string"},
        },
    }

    def __init__(
        self,
        name: str = "python_sandbox",
        timeout_seconds: float = 10.0,
        tier: Union[ToolSafetyTier, str] = ToolSafetyTier.TIER_0_READ_ONLY,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        if isinstance(tier, ToolSafetyTier):
            self.tier = tier
        elif isinstance(tier, str) and hasattr(ToolSafetyTier, tier):
            self.tier = getattr(ToolSafetyTier, tier)
        else:
            self.tier = ToolSafetyTier.TIER_0_READ_ONLY

        if description:
            self.description = description
        self.default_timeout = max(1.0, float(timeout_seconds))
        self.handler = self.execute
        self.is_safe_read_only = True

    def execute(
        self,
        params: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executa o código fornecido em subprocesso isolado com medição de tempo e limites estritos."""
        code = params.get("code") or params.get("script") or ""
        if not isinstance(code, str) or not code.strip():
            return {
                "status": "error",
                "stdout": "",
                "stderr": "",
                "returncode": 1,
                "output": "",
                "result": None,
                "execution_time_ms": 0.0,
                "error": "No Python code provided in parameters.",
            }

        timeout = float(params.get("timeout") or self.default_timeout)

        # Prepara wrapper seguro para capturar retorno de última expressão se houver
        wrapper_script = (
            "import sys, ast, traceback\n"
            "_source = " + repr(code) + "\n"
            "try:\n"
            "    _parsed = ast.parse(_source)\n"
            "    _last_expr = None\n"
            "    if _parsed.body and isinstance(_parsed.body[-1], ast.Expr):\n"
            "        _last_expr = _parsed.body.pop()\n"
            "    _scope = {'__name__': '__main__'}\n"
            "    if _parsed.body:\n"
            "        _compiled = compile(ast.Module(body=_parsed.body, type_ignores=[]), '<sandbox>', 'exec')\n"
            "        exec(_compiled, _scope)\n"
            "    if _last_expr is not None:\n"
            "        _val = eval(compile(ast.Expression(body=_last_expr.value), '<sandbox>', 'eval'), _scope)\n"
            "        if _val is not None:\n"
            "            print(f'__VERONICA_RESULT__:{repr(_val)}')\n"
            "except Exception:\n"
            "    traceback.print_exc(file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )

        t_start = time.perf_counter()
        try:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.run(
                [sys.executable, "-c", wrapper_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

            raw_stdout = proc.stdout or ""
            raw_stderr = proc.stderr or ""
            result_val = None
            clean_stdout_lines = []

            for line in raw_stdout.splitlines():
                if line.startswith("__VERONICA_RESULT__:"):
                    result_val = line[len("__VERONICA_RESULT__:") :]
                else:
                    clean_stdout_lines.append(line)

            clean_stdout = "\n".join(clean_stdout_lines).strip()
            output_display = clean_stdout or (f"Result: {result_val}" if result_val else "")

            if proc.returncode == 0:
                return {
                    "status": "success",
                    "stdout": clean_stdout,
                    "stderr": raw_stderr.strip(),
                    "returncode": proc.returncode,
                    "output": output_display,
                    "result": result_val,
                    "execution_time_ms": duration_ms,
                    "error": None,
                }
            else:
                return {
                    "status": "error",
                    "stdout": clean_stdout,
                    "stderr": raw_stderr.strip(),
                    "returncode": proc.returncode,
                    "output": output_display,
                    "result": None,
                    "execution_time_ms": duration_ms,
                    "error": raw_stderr.strip() or f"Process exited with non-zero code {proc.returncode}",
                }

        except subprocess.TimeoutExpired:
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "returncode": -1,
                "output": "",
                "result": None,
                "execution_time_ms": duration_ms,
                "error": f"Execution timed out after {timeout} seconds (exceeded deterministic boundary)",
            }
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
            return {
                "status": "error",
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
                "output": "",
                "result": None,
                "execution_time_ms": duration_ms,
                "error": f"Execution exception: {type(exc).__name__}: {exc}",
            }
