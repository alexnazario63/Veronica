"""Codebase Inspector Tool for syntax analysis, AST inspection, file reading, and symbol search."""
from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Union

from core.contracts import ToolSafetyTier
from core.agent.tools.base import AgentTool


class CodebaseInspectorTool(AgentTool):
    """Inspeciona sintaxe via AST, lê arquivos do repositório, busca símbolos e apoia refatoração."""

    def __init__(
        self,
        name: str = "codebase_inspector",
        tier: Union[ToolSafetyTier, str] = ToolSafetyTier.TIER_0_READ_ONLY,
        description: Optional[str] = None,
        workspace_root: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> None:
        desc = description or (
            "Inspects codebase, analyzes syntax trees using python ast, validates syntax across languages, "
            "reads repository files, and searches symbols to support surgical code generation and refactoring."
        )
        super().__init__(
            name=name,
            tier=tier,
            description=desc,
            input_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["validate_syntax", "inspect_ast", "read_file", "search_symbols", "list_files"],
                        "description": "Inspection action to execute",
                    },
                    "code": {"type": "string", "description": "Source code snippet for syntax validation or AST parsing"},
                    "language": {"type": "string", "description": "Programming language (python, typescript, rust, go, etc.)"},
                    "file_path": {"type": "string", "description": "Relative or absolute file path to read/inspect"},
                    "query": {"type": "string", "description": "Symbol name or search query"},
                    "start_line": {"type": "integer", "description": "Starting line number (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Ending line number (inclusive)"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ok", "error"]},
                    "action": {"type": "string"},
                    "result": {"type": "any"},
                    "valid": {"type": "boolean"},
                    "error": {"type": "string"},
                },
            },
            is_safe_read_only=True,
            **kwargs,
        )
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        self.handler = self.execute

    def validate_syntax(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Valida a correção sintática do código fornecido."""
        lang = (language or "python").lower().strip()
        if lang in ("python", "py"):
            try:
                tree = ast.parse(code)
                node_count = sum(1 for _ in ast.walk(tree))
                return {
                    "valid": True,
                    "language": "python",
                    "ast_nodes": node_count,
                    "error": None,
                    "message": "Python syntax parsed successfully via ast.parse.",
                }
            except SyntaxError as err:
                return {
                    "valid": False,
                    "language": "python",
                    "error": str(err),
                    "lineno": err.lineno,
                    "offset": err.offset,
                    "line_text": err.text.strip() if err.text else None,
                    "message": f"Syntax error at line {err.lineno}: {err.msg}",
                }
            except Exception as exc:
                return {
                    "valid": False,
                    "language": "python",
                    "error": str(exc),
                    "message": f"Parse exception: {type(exc).__name__}: {exc}",
                }

        # Verificação estrutural para TypeScript/JavaScript/Rust/C++/Go/SQL
        stack: List[str] = []
        matching = {")": "(", "}": "{", "]": "["}
        lines = code.splitlines()
        for idx, line in enumerate(lines, 1):
            in_str = False
            str_char = ""
            for ch in line:
                if ch in ("'", '"', '`') and not in_str:
                    in_str = True
                    str_char = ch
                elif in_str and ch == str_char:
                    in_str = False
                elif not in_str:
                    if ch in ("(", "{", "["):
                        stack.append(ch)
                    elif ch in (")", "}", "]"):
                        if not stack or stack[-1] != matching[ch]:
                            return {
                                "valid": False,
                                "language": lang,
                                "error": f"Mismatched bracket '{ch}' at line {idx}",
                                "lineno": idx,
                                "message": f"Delimiter mismatch '{ch}' at line {idx}",
                            }
                        stack.pop()

        if stack:
            return {
                "valid": False,
                "language": lang,
                "error": f"Unclosed brackets remaining: {', '.join(stack)}",
                "message": "Unclosed brackets detected in code block.",
            }

        return {
            "valid": True,
            "language": lang,
            "lines_count": len(lines),
            "error": None,
            "message": f"{lang.title()} code block delimiters validated successfully.",
        }

    def inspect_ast(self, code: str) -> Dict[str, Any]:
        """Analisa a árvore sintática do Python e extrai classes, funções, imports e complexidade."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {
                "valid": False,
                "error": str(exc),
                "lineno": exc.lineno,
                "classes": [],
                "functions": [],
                "imports": [],
            }

        classes: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
        imports: List[Dict[str, Any]] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                doc = ast.get_docstring(node) or ""
                classes.append(
                    {
                        "name": node.name,
                        "lineno": node.lineno,
                        "methods": methods,
                        "docstring": doc[:120],
                    }
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                doc = ast.get_docstring(node) or ""
                functions.append(
                    {
                        "name": node.name,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "args": args,
                        "lineno": node.lineno,
                        "docstring": doc[:120],
                    }
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"module": alias.name, "alias": alias.asname})
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append(
                        {"module": node.module or "", "name": alias.name, "alias": alias.asname}
                    )

        total_nodes = sum(1 for _ in ast.walk(tree))
        return {
            "valid": True,
            "total_nodes": total_nodes,
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "summary": f"AST com {len(classes)} classes, {len(functions)} funções e {len(imports)} imports.",
        }

    def read_file(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: int = -1,
    ) -> Dict[str, Any]:
        """Lê arquivo do repositório respeitando o workspace."""
        path = Path(file_path)
        if not path.is_absolute():
            path = (self.workspace_root / path).resolve()

        if not path.exists():
            return {"status": "error", "error": f"File not found: {file_path}", "lines": []}
        if not path.is_file():
            return {"status": "error", "error": f"Path is not a file: {file_path}", "lines": []}

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            all_lines = content.splitlines()
            s_idx = max(0, start_line - 1)
            e_idx = len(all_lines) if end_line < 1 else min(len(all_lines), end_line)
            selected_lines = all_lines[s_idx:e_idx]
            return {
                "status": "ok",
                "file_path": str(path),
                "total_lines": len(all_lines),
                "start_line": s_idx + 1,
                "end_line": e_idx,
                "content": "\n".join(selected_lines),
                "lines_count": len(selected_lines),
            }
        except Exception as exc:
            return {"status": "error", "error": f"Error reading file: {exc}", "lines": []}

    def search_symbols(
        self,
        query: str,
        root_dir: Optional[str] = None,
        max_results: int = 20,
    ) -> Dict[str, Any]:
        """Busca definições de classes, funções e símbolos nos arquivos do repositório."""
        target_root = Path(root_dir).resolve() if root_dir else self.workspace_root
        pattern = re.compile(
            rf"^\s*(?:async\s+def|def|class|function|fn|interface|type)\s+([a-zA-Z0-9_]*{re.escape(query)}[a-zA-Z0-9_]*)",
            re.IGNORECASE,
        )

        matches: List[Dict[str, Any]] = []
        extensions = {".py", ".ts", ".js", ".rs", ".go", ".cpp", ".c", ".sql"}

        for p in target_root.rglob("*"):
            if len(matches) >= max_results:
                break
            if p.suffix.lower() not in extensions:
                continue
            if any(skip in p.parts for skip in (".venv", "venv", "__pycache__", ".git", "node_modules")):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
                for lineno, line in enumerate(text.splitlines(), 1):
                    m = pattern.search(line)
                    if m:
                        rel = p.relative_to(self.workspace_root) if p.is_relative_to(self.workspace_root) else p
                        matches.append(
                            {
                                "file": str(rel),
                                "line": lineno,
                                "symbol": m.group(1),
                                "declaration": line.strip(),
                            }
                        )
                        if len(matches) >= max_results:
                            break
            except Exception:
                continue

        return {
            "status": "ok",
            "query": query,
            "matches_count": len(matches),
            "matches": matches,
        }

    def list_files(self, pattern: str = "*", directory: str = ".") -> Dict[str, Any]:
        """Lista arquivos no diretório especificado filtrando por padrão glob."""
        target_dir = (self.workspace_root / directory).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return {"status": "error", "error": f"Directory not found: {directory}", "files": []}

        results = []
        for p in target_dir.glob(pattern):
            if any(skip in p.parts for skip in (".venv", "venv", "__pycache__", ".git")):
                continue
            rel = p.relative_to(self.workspace_root) if p.is_relative_to(self.workspace_root) else p
            results.append(str(rel))

        return {"status": "ok", "directory": str(directory), "count": len(results), "files": sorted(results[:50])}

    def execute(
        self,
        params: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Roteia a ação solicitada para a funcionalidade correspondente."""
        action = str(params.get("action") or params.get("operation") or "").strip().lower()
        code = str(params.get("code") or params.get("content") or "")
        file_path = str(params.get("file_path") or params.get("path") or "")
        query = str(params.get("query") or params.get("symbol") or "")
        language = str(params.get("language") or "python")

        if action in ("validate_syntax", "check_syntax"):
            return self.validate_syntax(code, language=language)

        if action in ("inspect_ast", "parse_ast"):
            if not code and file_path:
                rf = self.read_file(file_path)
                code = rf.get("content", "")
            return self.inspect_ast(code)

        if action in ("read_file", "view_file"):
            start_l = int(params.get("start_line", 1))
            end_l = int(params.get("end_line", -1))
            return self.read_file(file_path, start_line=start_l, end_line=end_l)

        if action in ("search_symbols", "find_symbol"):
            return self.search_symbols(query)

        if action in ("list_files", "glob"):
            return self.list_files(
                pattern=str(params.get("pattern", "*")),
                directory=str(params.get("directory", ".")),
            )

        # Fallbacks automáticos intuitivos
        if code:
            return self.validate_syntax(code, language=language)
        if file_path:
            return self.read_file(file_path)
        if query:
            return self.search_symbols(query)

        return {
            "status": "error",
            "error": "No action, code, file_path, or query specified for CodebaseInspectorTool.",
        }
