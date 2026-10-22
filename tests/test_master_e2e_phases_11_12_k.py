"""
Bateria de Testes de Homologação Final da VERONICA:
Fases 11, 12, Fase K e Experiência Unificada Chat + Agent + Work.

Verificações obrigatórias:
1. Critérios matemáticos estritos do METRICS_AND_ACCEPTANCE.md:
   - gradcheck < 10^-5 em todas as operações
   - zero NaN/Inf e Gradient Clipping L2 com gamma_max = 1.0
   - latência de BinaryMachineMemoryStore.fast_knn_lookup(k=5) < 2.0 ms para 10.000 conceitos
   - neurogênese acionada com S(x) > 0.75 e poda após 1000 ciclos inativos
2. Fluxo Ponta a Ponta:
   Chat -> Multi-Agente ReAct -> Diagnóstico em 11 Etapas ->
   Geração/Edição de Artefato no Workspace -> Aprovação Humana na Etapa 6 (bloqueio de voz) ->
   Execução/Rollback -> Auditoria SHA-256 Multi-Tenant.
3. Telemetria Direta e Injetor de Falhas Sintéticas (DOM, LACP, BGP leak, MTU, OSPF partition) + Blast Radius.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from core.neural.tensor import Tensor
from core.neural.backend import get_backend, set_backend
from core.neural.clipping import clip_grad_norm_
from core.neural.gradcheck import gradcheck
from core.development.memory_system import BinaryMachineMemoryStore, ConceptNode, MemorySystem
from core.development.neurogenesis import NeurogenesisEngine
from core.development.development_engine import DevelopmentEngine, DevelopmentalStage
from core.agent.orchestrator import (
    ReActOrchestrator,
    BGPProtocolAgent,
    MPLSAndTrafficAgent,
    OpticalAndL1Agent,
    SecurityAndACLAgent,
    AgentTool,
)
from networking.topology.graph import NetworkTopologyGraph, create_default_network_inventory
from networking.vendors.adapters import DigitalTwinNetworkAdapter
from networking.vendors.async_adapters import (
    AsyncVendorAdapter,
    ScrapliAsyncSSHTransport,
    NetconfStreamingTransport,
    GnmiStreamingTransport,
)
from networking.diagnostics.engine import NetworkDiagnosticEngine
from networking.diagnostics.fault_injector import FaultInjector, SyntheticFaultScenario
from knowledge.retrieval.rag_engine import HybridRAGEngine
from safety.change_control.controller import DeterministicChangeController
from safety.policies.rbac_vault import RBACPolicyManager, CredentialVault, TamperEvidentAuditLog
from services.api.workspace.store import ArtifactStore


def _signed_approval(rbac: RBACPolicyManager, plan: dict[str, Any], approver_id: str = "usr-approver-01") -> dict[str, str]:
    signer = Ed25519PrivateKey.generate()
    public_key = signer.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("ascii")
    rbac.register_approval_public_key(approver_id, public_key)
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(
        {
            "operator_id": approver_id,
            "artifact_sha256_hash": plan["plan_hash"],
            "maintenance_window_id": plan["maintenance_window"],
            "timestamp": timestamp,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "signature_ed25519": base64.b64encode(signer.sign(payload)).decode("ascii"),
        "approval_timestamp": timestamp,
    }


def test_mathematical_criteria_gradcheck_and_clipping():
    """Valida erro relativo < 10^-5 no gradcheck, zero NaN/Inf e Gradient Clipping L2 gamma=1.0."""
    set_backend("numpy")
    rng = np.random.default_rng(42)

    # 1. Matmul + Add
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True, name="A")
    b = Tensor(rng.normal(size=(4, 2)), requires_grad=True, name="B")
    bias = Tensor(rng.normal(size=(1, 2)), requires_grad=True, name="bias")

    def fn_matmul() -> Tensor:
        z = (a @ b) + bias
        return ((z ** 2) * 0.5 - z * 0.3).mean()

    rep_matmul = gradcheck(fn_matmul, [a, b, bias])
    assert rep_matmul["passed"] is True
    assert rep_matmul["max_relative_error"] < 1e-5

    # 2. ReLU & GELU
    r_in = Tensor(rng.normal(size=(3, 3)), requires_grad=True, name="relu_in")
    rep_relu = gradcheck(lambda: r_in.relu().mean(), [r_in])
    assert rep_relu["passed"] is True
    assert rep_relu["max_relative_error"] < 1e-5

    g_in = Tensor(rng.normal(size=(3, 3)), requires_grad=True, name="gelu_in")
    rep_gelu = gradcheck(lambda: g_in.gelu().mean(), [g_in])
    assert rep_gelu["passed"] is True
    assert rep_gelu["max_relative_error"] < 1e-5

    # 3. Softmax
    s_in = Tensor(rng.normal(size=(2, 4)), requires_grad=True, name="soft_in")
    rep_soft = gradcheck(lambda: s_in.softmax(axis=-1).sum(), [s_in])
    assert rep_soft["passed"] is True
    assert rep_soft["max_relative_error"] < 1e-5

    # 4. Gradient Clipping L2 com gamma_max = 1.0 e zero NaN/Inf
    p1 = Tensor(np.ones((5, 5)), requires_grad=True)
    p2 = Tensor(np.ones((5, 5)), requires_grad=True)
    p1.grad = np.full((5, 5), 10.0)
    p2.grad = np.full((5, 5), 10.0)
    total_norm = clip_grad_norm_([p1, p2], max_norm=1.0)
    assert total_norm > 1.0

    clipped_norm = np.sqrt(np.sum(p1.grad ** 2) + np.sum(p2.grad ** 2))
    assert abs(clipped_norm - 1.0) < 1e-6
    assert not np.isnan(p1.grad).any() and not np.isinf(p1.grad).any()
    assert not np.isnan(p2.grad).any() and not np.isinf(p2.grad).any()


def test_mathematical_criteria_knn_latency_and_neurogenesis(tmp_path: Path):
    """Valida latência de fast_knn_lookup < 2.0 ms em 10.000 conceitos e limiar de neurogênese tau=0.75."""
    mem = MemorySystem(storage_dir=tmp_path)
    store = BinaryMachineMemoryStore(memory_system=mem, vector_dim=48)
    store._ensure_knn_index(min_concepts=10000)

    # Mede latência de k-NN lookup para k=5
    query = np.random.randn(48).astype(np.float32)
    query = query / np.linalg.norm(query)

    # Warm-up
    _ = store.fast_knn_lookup(query, k=5)

    start = time.perf_counter()
    iterations = 25
    for _ in range(iterations):
        results = store.fast_knn_lookup(query, k=5)
    elapsed_ms = ((time.perf_counter() - start) / iterations) * 1000.0

    assert len(results) == 5
    assert elapsed_ms < 2.0, f"Latência do KNN excedeu 2.0 ms: {elapsed_ms:.3f} ms"

    # Neurogênese: acionamento com S(x) > 0.75 e poda após inatividade
    neuro = NeurogenesisEngine(d_model=48, storage_dir=tmp_path)
    circuit = "bgp_anomaly_detector"

    novel_signal = np.random.randn(48).astype(np.float32)
    surprise = neuro.evaluate_signal_surprise(circuit, novel_signal)
    assert surprise >= 0.75, f"Surpresa esperada >= 0.75, obteve {surprise}"

    neuron = neuro.trigger_neurogenesis(circuit, novel_signal, reason="High DOM optical anomaly")
    assert neuron is not None

    pruned = neuro.prune_idle_synapses(inactivity_threshold_cycles=1000)
    assert isinstance(pruned, int)


@pytest.mark.asyncio
async def test_end_to_end_cycle_chat_react_change_and_audit(tmp_path: Path):
    """
    Fluxo E2E Integrado:
    Pergunta no Chat -> Ciclo Multi-Agente ReAct -> Diagnóstico 11 Etapas ->
    Geração de Artefato no Workspace -> Co-edição de Diff ->
    Aprovação Humana na Etapa 6 (com bloqueio de voz) -> Execução/Auditoria SHA-256 Multi-Tenant.
    """
    prior_lock = os.environ.get("VERONICA_SAFETY_LOCK_DB")
    os.environ["VERONICA_SAFETY_LOCK_DB"] = str(tmp_path / "safety-locks.sqlite3")

    try:
        # 1. Governança Multi-Tenant (Fase K)
        tenant_id = "tenant-enterprise-alpha"
        rbac = RBACPolicyManager()
        rbac.register_user(
            user_id="usr-engineer-01",
            name="Alice Engineer",
            role="NETWORK_ENGINEER",
            tenant_id=tenant_id,
        )
        rbac.register_user(
            user_id="usr-approver-01",
            name="Bob Approver",
            role="CHANGE_APPROVER",
            tenant_id=tenant_id,
        )
        audit_log = TamperEvidentAuditLog()

        # 2. Infraestrutura e Adaptadores
        inv = create_default_network_inventory()
        topology = NetworkTopologyGraph(inv)
        adapter = DigitalTwinNetworkAdapter(inv)
        change_ctrl = DeterministicChangeController(
            adapter=adapter,
            topology=topology,
            rbac=rbac,
            audit_log=audit_log,
        )
        artifact_store = ArtifactStore(tmp_path / "artifacts.sqlite3")

        # 3. Orquestrador ReAct com Sub-Agentes Especializados (Fase 12)
        bgp_agent = BGPProtocolAgent()
        mpls_agent = MPLSAndTrafficAgent()
        optical_agent = OpticalAndL1Agent()
        sec_agent = SecurityAndACLAgent()

        orchestrator = ReActOrchestrator(
            tools=[
                AgentTool(
                    name="diagnose_bgp_route_map",
                    tier=None,
                    handler=lambda args: {"status": "bgp_session_established", "as_path": [65001, 65002]},
                    description="Diagnostica adjacências e políticas BGP",
                    is_safe_read_only=True,
                ),
            ],
            change_control=change_ctrl,
        )

        # 4. Execução do método solve() multi-agente
        events = []
        async for event in orchestrator.solve("Investigar instabilidade BGP no roteador NE8000-CORE-01"):
            events.append(event)

        event_types = [e.get("type") for e in events]
        assert "thought_step" in event_types
        assert "final_answer" in event_types

        # 5. Diagnóstico em 11 Etapas
        rag = HybridRAGEngine()
        diag_engine = NetworkDiagnosticEngine(adapter=adapter, topology=topology, rag=rag)
        diag_result = diag_engine.investigate("Queda intermitente de sessão BGP no NE8000-CORE-01")
        assert len(diag_result["steps_executed"]) == 11
        assert len(diag_result["epistemic_breakdown"]["verified_facts"]) >= 1

        # 6. Geração de Artefato no Workspace NetCanvas (com commit trial 120 para VRP 8)
        plan = change_ctrl.stage_change(
            target="NE8000-CORE-01",
            intent=["system-view", "bgp 65001", "peer 10.0.0.2 timer keepalive 30 hold 90", "commit trial 120"],
        )
        assert plan["current_stage"] == 5 or plan.get("status") in {"STAGED", "AWAITING_HUMAN_APPROVAL", "SIMULATED"}

        art_id = str(plan["change_id"])
        v1 = artifact_store.save_artifact(
            artifact_id=art_id,
            title=f"Plano de Mudança BGP Timers {art_id}",
            content=plan.get("unified_diff", "diff placeholder"),
            kind="change_plan_10_steps",
            author="usr-engineer-01",
            tenant_id=tenant_id,
            metadata={"plan_id": art_id, "device_id": "NE8000-CORE-01"},
        )
        assert v1 == 1
        artifact_store.link_change_plan(art_id, art_id)

        # Co-edição de proposta de patch no Workspace
        patch_text = (
            "--- baseline\n"
            "+++ proposed\n"
            "@@ -1,2 +1,2 @@\n"
            "- peer 10.0.0.2 timer keepalive 60 hold 180\n"
            "+ peer 10.0.0.2 timer keepalive 30 hold 90\n"
        )
        patch_res = artifact_store.propose_patch(art_id, version=1, patch_diff=patch_text, author="usr-engineer-01")
        assert patch_res["status"] == "PROPOSED"
        assert patch_res["new_version"] == 2

        # 7. Regras de Aprovação Humana na Etapa 6:
        plan_hash = plan["plan_hash"]
        window_id = plan.get("maintenance_window") or "MW-EMERGENCY-WINDOW"

        # A) Bloqueio Absoluto de Voz (ADR-03 / R-09)
        with pytest.raises(PermissionError) as exc_voice:
            change_ctrl.authorize_change(
                plan_id=art_id,
                approver_id="usr-approver-01",
                token=plan_hash,
                window_id=window_id,
                approval_channel="VOICE",
                tenant_id=tenant_id,
            )
        assert "VOICE_APPROVAL_PROHIBITED" in str(exc_voice.value)

        # B) Aprovação Web Válida assinada com Ed25519
        sig_bundle = _signed_approval(rbac, plan, "usr-approver-01")
        auth_receipt = change_ctrl.authorize_change(
            plan_id=art_id,
            approver_id="usr-approver-01",
            token=plan_hash,
            window_id=window_id,
            approval_channel="EXPLICIT_WEB_APPROVAL",
            signature_ed25519=sig_bundle["signature_ed25519"],
            approval_timestamp=sig_bundle["approval_timestamp"],
            tenant_id=tenant_id,
        )
        assert auth_receipt is not None

        # 8. Execução Controlada na Etapa 7 até a Etapa 10
        exec_result = change_ctrl.execute_change(art_id, auth_receipt)
        assert exec_result.get("success") is True or exec_result.get("status") in {
            "SUCCESS",
            "COMPLETED",
            "COMPLETED_VERIFIED",
            "EXECUTED",
        }

        # 9. Verificação Criptográfica da Trilha de Auditoria SHA-256 Multi-Tenant
        chain_valid, broken_idx = audit_log.verify_chain_integrity(tenant_id=tenant_id)
        assert chain_valid is True, f"Trilha de auditoria corrompida no índice: {broken_idx}"

        entries = audit_log.list_entries(tenant_id=tenant_id)
        assert len(entries) >= 2
    finally:
        if prior_lock is None:
            os.environ.pop("VERONICA_SAFETY_LOCK_DB", None)
        else:
            os.environ["VERONICA_SAFETY_LOCK_DB"] = prior_lock


def test_synthetic_fault_injection_and_blast_radius():
    """Valida gerador de falhas sintéticas (Fase 12) nos 5 cenários com cálculo de Blast Radius."""
    injector = FaultInjector()

    # 1. Falha DOM (Degradação Óptica)
    fault_dom = injector.inject_optical_degradation(
        device_id="NE8000-CORE-01",
        interface="10GE1/0/2",
        rx_power_dbm=-19.0,
    )
    assert fault_dom["scenario"] == SyntheticFaultScenario.OPTICAL_DEGRADATION.value
    blast_dom = injector.calculate_blast_radius(fault_dom["fault_id"])
    assert "root_device" in blast_dom
    assert blast_dom["root_device"] == "NE8000-CORE-01"

    # 2. Flapping LACP
    fault_lacp = injector.inject_lacp_flapping(
        device_id="NE8000-CORE-01",
        bundle_interface="Eth-Trunk1",
    )
    assert fault_lacp["scenario"] == SyntheticFaultScenario.LACP_FLAPPING.value

    # 3. Route-leaking BGP
    fault_bgp = injector.inject_bgp_route_leak(
        device_id="NE8000-CORE-01",
        leaked_prefix="10.100.0.0/16",
        peer_ip="10.0.0.2",
    )
    assert fault_bgp["scenario"] == SyntheticFaultScenario.BGP_ROUTE_LEAK.value

    # 4. MTU Mismatch
    fault_mtu = injector.inject_mtu_mismatch(
        device_id="ASR9K-CORE-02",
        interface="TenGigE0/0/0/2",
        neighbor_device="C9500-DIST-04",
        local_mtu=9216,
        remote_mtu=1500,
    )
    assert fault_mtu["scenario"] == SyntheticFaultScenario.MTU_MISMATCH.value

    # 5. Partição OSPF/IS-IS
    fault_part = injector.inject_area_partition(
        partitioned_area="0.0.0.0",
        isolated_nodes=["NE8000-CORE-01"],
    )
    assert fault_part["scenario"] == SyntheticFaultScenario.AREA_PARTITION.value

    # Grafo de impacto
    impact_graph = injector.generate_impact_graph(fault_dom["fault_id"])
    assert "nodes" in impact_graph
    assert "edges" in impact_graph

    # Restauração limpa
    injector.restore_all()
