"""Unit and integration tests for universal cognitive ReAct orchestrator and codex tools suite."""
from __future__ import annotations

import asyncio
import json
import pytest

from core.agent import (
    AgentTool,
    BGPProtocolAgent,
    CodexAndSoftwareAgent,
    CodebaseInspectorTool,
    DataAnalysisAgent,
    DomainSubAgent,
    DynamicWebAndDocsSearchTool,
    GeneralKnowledgeAgent,
    MathAndScienceAgent,
    MPLSAndTrafficAgent,
    NetworkOperationsTool,
    OpticalAndL1Agent,
    PythonSandboxTool,
    ReActOrchestrator,
    SecurityAndACLAgent,
)
from core.contracts import ToolSafetyTier


# ============================================================================
# 1. Testes da Suite de Ferramentas Estilo Codex
# ============================================================================

def test_python_sandbox_tool_execution_and_math():
    """Valida execução de cálculos matemáticos e retorno de expressões no PythonSandboxTool."""
    sandbox = PythonSandboxTool(timeout_seconds=5.0)
    assert sandbox.name == "python_sandbox"
    assert sandbox.is_safe_read_only is True

    # Teste de cálculo aritmético com valor de retorno
    result = sandbox.execute({"code": "a = 21\nb = 2\na * b"})
    assert result["status"] == "success"
    assert result["result"] == "42"
    assert result["returncode"] == 0
    assert result["execution_time_ms"] > 0

    # Teste com stdout explícito
    res_print = sandbox.execute({"code": "print('Veronica Universal Codex')"})
    assert res_print["status"] == "success"
    assert "Veronica Universal Codex" in res_print["stdout"]

    # Teste de erro sintático/exceção isolada
    res_err = sandbox.execute({"code": "1 / 0"})
    assert res_err["status"] == "error"
    assert "ZeroDivisionError" in res_err["stderr"]


def test_python_sandbox_tool_timeout():
    """Valida que loops infinitos ou scripts lentos sofrem timeout determinístico de 10s."""
    sandbox = PythonSandboxTool(timeout_seconds=1.5)
    result = sandbox.execute({"code": "import time\ntime.sleep(5)"})
    assert result["status"] == "timeout"
    assert "timed out" in result["error"].lower()


def test_codebase_inspector_tool():
    """Valida análise sintática via AST, inspeção de símbolos e leitura de arquivos."""
    inspector = CodebaseInspectorTool()
    assert inspector.name == "codebase_inspector"
    assert inspector.is_safe_read_only is True

    # 1. Validação de sintaxe Python correta
    valid_code = "def compute_hash(val: str) -> str:\n    return val.strip().lower()\n"
    res_val = inspector.execute({"action": "validate_syntax", "code": valid_code, "language": "python"})
    assert res_val["valid"] is True
    assert res_val["ast_nodes"] > 0

    # 2. Validação de sintaxe Python com erro
    invalid_code = "def broken(\n    return 42"
    res_inval = inspector.execute({"action": "validate_syntax", "code": invalid_code, "language": "python"})
    assert res_inval["valid"] is False
    assert res_inval["lineno"] is not None

    # 3. Inspeção estrutural de AST
    module_code = (
        "class RouterConfig:\n"
        "    def apply(self):\n"
        "        pass\n\n"
        "def main():\n"
        "    return 0\n"
    )
    ast_res = inspector.execute({"action": "inspect_ast", "code": module_code})
    assert ast_res["valid"] is True
    assert any(c["name"] == "RouterConfig" for c in ast_res["classes"])
    assert any(f["name"] == "main" for f in ast_res["functions"])

    # 4. Leitura segura de arquivo
    file_res = inspector.execute({"action": "read_file", "file_path": "core/agent/__init__.py", "start_line": 1, "end_line": 5})
    assert file_res["status"] == "ok"
    assert "ReActOrchestrator" in file_res["content"] or "AgentTool" in file_res["content"]


def test_dynamic_web_and_docs_search_tool():
    """Valida buscas técnicas de RFCs e padrões da indústria."""
    search_tool = DynamicWebAndDocsSearchTool()
    assert search_tool.name == "dynamic_web_and_docs_search"
    assert search_tool.is_safe_read_only is True

    # Busca por BGP RFC 4271
    res = search_tool.execute({"query": "RFC 4271 BGP"})
    assert res["status"] == "ok"
    assert res["count"] >= 1
    titles = [r["title"] for r in res["results"]]
    assert any("4271" in t for t in titles)

    # Busca por MPLS VPN
    res_mpls = search_tool.execute({"query": "RFC 4364 MPLS VPN"})
    assert any("4364" in r["title"] for r in res_mpls["results"])


def test_network_operations_tool_and_commit_safety():
    """Valida diagnóstico de rede e inclusão determinística de commit confirmed/trial 120."""
    net_tool = NetworkOperationsTool()
    assert net_tool.name == "network_operations"

    # Diagnóstico nominal
    diag = net_tool.execute({"action": "diagnose", "device_id": "NE8000-CORE-01"})
    assert diag["status"] == "ok"
    assert diag["device_id"] == "NE8000-CORE-01"

    # Garantia determinística de commit confirmed/trial 120
    raw_cmds = ["system-view", "bgp 65001", "peer 10.0.0.1 timer keepalive 30 hold 90"]
    # Para Huawei/VRP:
    safe_vrp = net_tool.ensure_commit_confirmed_safety(raw_cmds, vendor="Huawei")
    assert "commit trial 120" in safe_vrp

    # Para Cisco/Outros:
    safe_cisco = net_tool.ensure_commit_confirmed_safety(raw_cmds, vendor="Cisco")
    assert "commit confirmed 120" in safe_cisco

    # Comandos somente leitura não devem receber commit confirmed
    ro_cmds = ["display bgp peer", "display current-configuration"]
    safe_ro = net_tool.ensure_commit_confirmed_safety(ro_cmds, vendor="Huawei")
    assert not any("commit" in c for c in safe_ro)


# ============================================================================
# 2. Testes de Roteamento Cognitivo Universal
# ============================================================================

def test_universal_cognitive_routing():
    """Valida que o ReActOrchestrator roteia para o domínio cognitivo correto sem viés de rede."""
    orchestrator = ReActOrchestrator()

    # 1. Programação / Codex
    agents_codex = orchestrator._route_to_specialized_agents("Escreva uma função em Python para calcular a sequência de Fibonacci")
    assert agents_codex[0].name == "CodexAndSoftwareAgent"
    assert agents_codex[0].domain == "software.engineering_and_codex"

    # 2. Matemática / Ciência
    agents_math = orchestrator._route_to_specialized_agents("Calcule a matriz jacobiana e verifique a convergência do gradiente com gradcheck")
    assert agents_math[0].name == "MathAndScienceAgent"
    assert agents_math[0].domain == "science.mathematics_and_physics"

    # 3. Análise de Dados
    agents_data = orchestrator._route_to_specialized_agents("Faça uma análise estatística e agregação no dataframe de telemetria")
    assert agents_data[0].name == "DataAnalysisAgent"
    assert agents_data[0].domain == "data.analytics_and_science"

    # 4. Conhecimento Geral e Técnico
    agents_gen = orchestrator._route_to_specialized_agents("Explique a especificação e o padrão do protocolo IETF RFC 8200 IPv6")
    assert agents_gen[0].name in ("GeneralKnowledgeAgent", "CodexAndSoftwareAgent")

    # 5. Engenharia de Redes Multi-Vendor
    agents_bgp = orchestrator._route_to_specialized_agents("Investigar sessão BGP em estado Active no roteador de borda")
    assert agents_bgp[0].name == "BGPProtocolAgent"


# ============================================================================
# 3. Testes de Auto-Reflexão, Verificação de Código e Geração de Artefato
# ============================================================================

@pytest.mark.asyncio
async def test_solve_programming_task_with_code_verification():
    """Valida auto-reflexão, validação sintática ast.parse e emissão de workspace_artifact de código."""
    orchestrator = ReActOrchestrator()

    events = []
    async for event in orchestrator.solve("Escreva uma função Python para calcular o fatorial de um número"):
        events.append(event)

    event_types = [e["type"] for e in events]
    assert "thought_step" in event_types
    assert "workspace_artifact" in event_types
    assert "final_answer" in event_types

    # Valida o thought step de auto-reflexão de código
    reflection_thoughts = [
        e for e in events if e.get("type") == "thought_step" and e.get("stage") == "CODE_SELF_REFLECTION"
    ]
    assert len(reflection_thoughts) >= 1
    assert "ast.parse" in reflection_thoughts[0]["content"]

    # Valida o artefato emitido para o Workspace
    code_artifacts = [
        e for e in events if e.get("type") == "workspace_artifact" and e.get("kind") == "code_workspace"
    ]
    assert len(code_artifacts) == 1
    art = code_artifacts[0]
    assert art["language"] == "python"
    assert "def " in art["code"]

    payload = json.loads(art["content"])
    assert payload["syntax_validated"] is True
    assert payload["status"] == "READY_FOR_WORKSPACE"


# ============================================================================
# 4. Testes de Preservação de Contrato (INTERFACES.md)
# ============================================================================

def test_contract_run_step_preservation():
    """Valida contrato formal de run_step(question, history) de acordo com INTERFACES.md."""
    orchestrator = ReActOrchestrator()
    step_out = orchestrator.run_step("Como implementar uma pilha thread-safe em Rust?")

    assert "thought" in step_out
    assert "action" in step_out
    assert "params" in step_out
    assert "agent" in step_out
    assert "cognitive_context" in step_out
    assert step_out["agent"] == "CodexAndSoftwareAgent"


def test_agent_tool_subclasses_contract():
    """Valida contrato formal de AgentTool em todas as ferramentas da suite."""
    tools = [
        PythonSandboxTool(),
        CodebaseInspectorTool(),
        DynamicWebAndDocsSearchTool(),
        NetworkOperationsTool(),
    ]
    for t in tools:
        assert isinstance(t, AgentTool)
        assert hasattr(t, "name")
        assert hasattr(t, "description")
        assert hasattr(t, "input_schema")
        assert hasattr(t, "output_schema")
        assert hasattr(t, "is_safe_read_only")
        assert callable(getattr(t, "execute"))
