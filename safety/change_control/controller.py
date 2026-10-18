"""
Controlador Determinístico de Mudanças Seguras da Veronica (Seção 8).
Independente do modelo neural, obriga que toda alteração percorra rigorosamente
as 10 etapas:
1. Descoberta
2. Baseline
3. Análise de impacto
4. Planejamento
5. Simulação
6. Aprovação humana explícita (proibida aprovação implícita ou por comando de voz)
7. Execução controlada (com verificação de concorrência e escopo)
8. Pós-validação
9. Rollback automático em caso de falha
10. Auditoria imutável SHA-256
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional
import uuid

from networking.vendors.adapters import BaseVendorAdapter
from networking.topology.graph import NetworkTopologyGraph
from safety.policies.rbac_vault import RBACPolicyManager, TamperEvidentAuditLog
from safety.rollback.manager import RollbackOrchestrator


class DeterministicChangeController:
    """
    Motor determinístico de controle de mudanças em 10 etapas (Seção 8).
    """

    def __init__(
        self,
        adapter: BaseVendorAdapter,
        topology: NetworkTopologyGraph,
        rbac: RBACPolicyManager,
        audit_log: TamperEvidentAuditLog,
    ) -> None:
        self.adapter = adapter
        self.topology = topology
        self.rbac = rbac
        self.audit_log = audit_log
        self.rollback_orchestrator = RollbackOrchestrator(adapter, audit_log)
        self.change_records: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _compute_plan_hash(device_id: str, base_hash: str, commands: List[str]) -> str:
        payload = json.dumps(
            {"device_id": device_id, "base_hash": base_hash, "commands": commands},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def create_change_plan_stages_1_to_5(
        self,
        requester_user_id: str,
        device_id: str,
        title: str,
        proposed_commands: List[str],
        maintenance_window: str = "2026-10-18 02:00-04:00 BRT",
        origin_channel: str = "TEXT_UI",
    ) -> Dict[str, Any]:
        """
        Executa as Etapas 1 a 5 (Descoberta, Baseline, Impacto, Planejamento e Simulação)
        e estaciona o plano no estado `AWAITING_HUMAN_APPROVAL` (Etapa 6).
        """
        self.rbac.require_permission(requester_user_id, "change:propose")

        change_id = f"CHG-{uuid.uuid4().hex[:8].upper()}"

        # Etapa 3 (pré-check de topologia): se o impacto não puder ser avaliado, bloqueia imediatamente
        impact = self.topology.analyze_impact(device_id, proposed_commands)
        if not impact.get("can_evaluate", False):
            self.audit_log.record(
                actor=requester_user_id,
                action="CHANGE_BLOCKED_UNKNOWN_IMPACT",
                target=device_id,
                status="BLOCKED",
                details={"change_id": change_id, "reason": impact.get("reason")},
            )
            raise RuntimeError(
                f"IMPACT_EVALUATION_IMPOSSIBLE: Mudança bloqueada porque o impacto não pôde ser avaliado ({impact.get('reason')})."
            )

        # Etapa 1 — Descoberta
        discovery = {
            "identity": self.adapter.discover_device(device_id),
            "version": self.adapter.get_version(device_id),
            "capabilities": self.adapter.get_capabilities(device_id),
            "neighbors": self.adapter.get_topology_neighbors(device_id),
        }

        # Etapa 2 — Baseline
        running_cfg = self.adapter.get_running_config(device_id)
        baseline_state = self.adapter.get_operational_state(device_id)
        backup = self.adapter.create_backup(device_id)
        baseline = {
            "config_hash": running_cfg["config_hash"],
            "backup_id": backup["backup_id"],
            "operational_state": baseline_state,
            "collected_at": running_cfg["collected_at"],
        }

        # Etapa 4 — Planejamento e Validação Sintática/Pré-condições
        validation = self.adapter.validate_commands(device_id, proposed_commands)
        preconditions = self.adapter.check_preconditions(device_id, proposed_commands)

        if not validation["all_valid"] or not preconditions["passed"]:
            self.audit_log.record(
                actor=requester_user_id,
                action="CHANGE_PLAN_REJECTED",
                target=device_id,
                status="REJECTED",
                details={
                    "change_id": change_id,
                    "rejected_commands": validation["rejected_commands"],
                    "blocking_reasons": preconditions["blocking_reasons"],
                },
            )
            raise ValueError(
                f"SAFE_CHANGE_GUARD_BLOCKED: Comandos inválidos ou pré-condições violadas. "
                f"Rejeitados={validation['rejected_commands']}; Motivos={preconditions['blocking_reasons']}"
            )

        rollback_plan = self.adapter.prepare_rollback(device_id, backup, proposed_commands)
        plan_hash = self._compute_plan_hash(
            device_id, running_cfg["config_hash"], validation["valid_commands"]
        )

        # Etapa 5 — Simulação em Digital Twin (Candidate + Diff)
        candidate = self.adapter.generate_candidate(device_id, validation["valid_commands"])
        diff = self.adapter.compare_configurations(
            running_cfg["running_config"], candidate["candidate_config"]
        )

        record: Dict[str, Any] = {
            "change_id": change_id,
            "title": title,
            "device_id": device_id,
            "requester_user_id": requester_user_id,
            "origin_channel": origin_channel,
            "maintenance_window": maintenance_window,
            "status": "AWAITING_HUMAN_APPROVAL",
            "current_stage": 6,
            "plan_hash": plan_hash,
            "stages": {
                "stage_1_discovery": discovery,
                "stage_2_baseline": baseline,
                "stage_3_impact_analysis": impact,
                "stage_4_plan": {
                    "commands": validation["valid_commands"],
                    "preconditions": preconditions,
                    "rollback_plan": rollback_plan,
                    "expected_outcome": "Convergência limpa de protocolo sem perda de serviços adjacentes.",
                },
                "stage_5_simulation": {
                    "candidate_hash": candidate["candidate_hash"],
                    "diff": diff,
                    "syntax_validated": True,
                },
                "stage_6_approval": None,
                "stage_7_execution": None,
                "stage_8_post_validation": None,
                "stage_9_rollback": None,
                "stage_10_audit": [],
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        audit_entry = self.audit_log.record(
            actor=requester_user_id,
            action="CHANGE_PLANNED_AND_SIMULATED",
            target=device_id,
            status="AWAITING_HUMAN_APPROVAL",
            details={
                "change_id": change_id,
                "plan_hash": plan_hash,
                "baseline_config_hash": baseline["config_hash"],
                "risk_level": impact["risk_level"],
                "commands": validation["valid_commands"],
            },
        )
        record["stages"]["stage_10_audit"].append(audit_entry["entry_id"])
        self.change_records[change_id] = record
        return record

    def approve_change_stage_6(
        self,
        change_id: str,
        approver_user_id: str,
        confirmed_plan_hash: str,
        confirmed_maintenance_window: str,
        approval_channel: str = "EXPLICIT_WEB_APPROVAL",
    ) -> Dict[str, Any]:
        """
        Etapa 6 — Aprovação Humana Explícita.
        Bloqueia aprovações originadas por canal de voz (`VOICE`) conforme Seção 16.8!
        """
        if approval_channel.upper() == "VOICE":
            self.audit_log.record(
                actor=approver_user_id,
                action="VOICE_APPROVAL_BLOCKED",
                target=change_id,
                status="DENIED",
                details={"reason": "O modo de voz não pode autorizar alterações em equipamentos."},
            )
            raise PermissionError(
                "VOICE_APPROVAL_PROHIBITED: O modo de voz não pode autorizar alterações em equipamentos. "
                "Exige aprovação humana formal vinculada ao plano."
            )

        self.rbac.require_permission(approver_user_id, "change:approve")

        if change_id not in self.change_records:
            raise KeyError(f"Plano de mudança '{change_id}' não encontrado.")
        rec = self.change_records[change_id]

        if confirmed_plan_hash != rec["plan_hash"]:
            raise ValueError(
                f"PLAN_HASH_MISMATCH: Hash informado ({confirmed_plan_hash}) difere do hash do plano ({rec['plan_hash']})."
            )
        if confirmed_maintenance_window != rec["maintenance_window"]:
            raise ValueError("MAINTENANCE_WINDOW_MISMATCH: Janela de manutenção divergente.")

        approval_token = f"APPR-{change_id}-{uuid.uuid4().hex[:10].upper()}"
        approval_data = {
            "approved": True,
            "approver_user_id": approver_user_id,
            "approval_token": approval_token,
            "plan_hash": rec["plan_hash"],
            "baseline_config_hash": rec["stages"]["stage_2_baseline"]["config_hash"],
            "maintenance_window": confirmed_maintenance_window,
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        rec["stages"]["stage_6_approval"] = approval_data
        rec["status"] = "APPROVED_READY_FOR_EXECUTION"
        rec["current_stage"] = 7

        audit_entry = self.audit_log.record(
            actor=approver_user_id,
            action="CHANGE_APPROVED",
            target=rec["device_id"],
            status="APPROVED",
            details={
                "change_id": change_id,
                "plan_hash": rec["plan_hash"],
                "maintenance_window": confirmed_maintenance_window,
            },
        )
        rec["stages"]["stage_10_audit"].append(audit_entry["entry_id"])
        return rec

    def execute_change_stages_7_to_10(
        self,
        change_id: str,
        executor_user_id: str,
        approval_token: str,
    ) -> Dict[str, Any]:
        """
        Executa as Etapas 7 (Execução Controlada), 8 (Pós-validação),
        9 (Rollback automático se pós-validação falhar) e 10 (Auditoria final).
        """
        self.rbac.require_permission(executor_user_id, "change:execute")

        if change_id not in self.change_records:
            raise KeyError(f"Plano de mudança '{change_id}' não encontrado.")
        rec = self.change_records[change_id]
        approval = rec["stages"].get("stage_6_approval")

        if not approval or not approval.get("approved") or approval.get("approval_token") != approval_token:
            self.audit_log.record(
                actor=executor_user_id,
                action="UNAUTHORIZED_EXECUTION_ATTEMPT",
                target=rec["device_id"],
                status="BLOCKED",
                details={"change_id": change_id},
            )
            raise PermissionError(
                "UNAUTHORIZED_EXECUTION_BLOCKED: Tentativa de execução sem token de aprovação válido."
            )

        device_id = rec["device_id"]
        commands = rec["stages"]["stage_4_plan"]["commands"]
        expected_hash = rec["stages"]["stage_2_baseline"]["config_hash"]
        baseline_state = rec["stages"]["stage_2_baseline"]["operational_state"]
        rollback_plan = rec["stages"]["stage_4_plan"]["rollback_plan"]

        # Etapa 7 — Execução controlada (verifica concorrência pelo expected_config_hash)
        try:
            exec_res = self.adapter.apply_change(
                device_id=device_id,
                commands=commands,
                approval_token=approval_token,
                expected_config_hash=expected_hash,
            )
        except Exception as exc:
            rec["status"] = "ABORTED_PRE_EXECUTION"
            self.audit_log.record(
                actor=executor_user_id,
                action="CHANGE_EXECUTION_ABORTED",
                target=device_id,
                status="ABORTED",
                details={"change_id": change_id, "error": str(exc)},
            )
            raise

        rec["stages"]["stage_7_execution"] = exec_res
        rec["current_stage"] = 8

        # Etapa 8 — Pós-validação
        post_state = self.adapter.get_operational_state(device_id)
        verification = self.adapter.verify_change(
            device_id=device_id,
            baseline_state=baseline_state,
            post_state=post_state,
        )
        rec["stages"]["stage_8_post_validation"] = verification

        # Etapa 9 — Rollback automático se a pós-validação falhar
        if not verification["passed"]:
            rec["current_stage"] = 9
            rb_res = self.rollback_orchestrator.trigger_rollback(
                change_id=change_id,
                device_id=device_id,
                rollback_plan=rollback_plan,
                reason=verification["summary"],
                actor=executor_user_id,
            )
            rec["stages"]["stage_9_rollback"] = rb_res
            rec["status"] = "ROLLED_BACK_AFTER_POSTCHECK_FAILURE"
        else:
            rec["stages"]["stage_9_rollback"] = {"triggered": False, "reason": "Pós-validação aprovada."}
            rec["status"] = "COMPLETED_VERIFIED"

        rec["current_stage"] = 10
        audit_entry = self.audit_log.record(
            actor=executor_user_id,
            action="CHANGE_LIFECYCLE_FINISHED",
            target=device_id,
            status=rec["status"],
            details={
                "change_id": change_id,
                "post_validation_passed": verification["passed"],
                "rollback_triggered": not verification["passed"],
            },
        )
        rec["stages"]["stage_10_audit"].append(audit_entry["entry_id"])
        return rec
