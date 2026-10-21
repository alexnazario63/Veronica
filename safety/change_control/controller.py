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
from typing import Any, Dict, List
import uuid
import base64

from networking.vendors.adapters import BaseVendorAdapter
from networking.topology.graph import NetworkTopologyGraph
from safety.policies.rbac_vault import RBACPolicyManager, TamperEvidentAuditLog
from safety.rollback.manager import RollbackOrchestrator
from safety.change_control.blast_lock import BlastRadiusLock
from safety.change_control.fsm import ChangeLifecycle
from safety.rollback.validation import compare_operational_invariants


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
        blast_radius_lock: BlastRadiusLock | None = None,
    ) -> None:
        self.adapter = adapter
        self.topology = topology
        self.rbac = rbac
        self.audit_log = audit_log
        self.rollback_orchestrator = RollbackOrchestrator(adapter, audit_log)
        self.change_records: Dict[str, Dict[str, Any]] = {}
        self.blast_radius_lock = (
            blast_radius_lock if blast_radius_lock is not None else BlastRadiusLock()
        )
        self._lifecycles: Dict[str, ChangeLifecycle] = {}

    @staticmethod
    def _compute_plan_hash(device_id: str, base_hash: str, commands: List[str]) -> str:
        payload = json.dumps(
            {"device_id": device_id, "base_hash": base_hash, "commands": commands},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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

        self.rbac.authorize_change(
            requester_user_id,
            risk_level=str(impact.get("risk_level", "CRITICAL")),
            commands=proposed_commands,
            maintenance_window_id=maintenance_window,
            asset_criticality=str(
                self.topology.inventory.get(device_id, {}).get("criticality", "STANDARD")
            ),
        )

        lock_resources = [device_id, *impact.get("affected_neighbors", [])]
        try:
            lock_lease = self.blast_radius_lock.acquire(change_id, lock_resources)
        except RuntimeError as exc:
            self.audit_log.record(
                actor=requester_user_id, action="CHANGE_BLOCKED_BLAST_RADIUS_COLLISION",
                target=device_id, status="BLOCKED",
                details={"change_id": change_id, "resources": lock_resources, "reason": str(exc)},
            )
            raise

        # Etapa 1 — Descoberta
        try:
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
                self.blast_radius_lock.release(change_id)
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
                "blast_radius_lock": {
                    "resources": sorted(lock_lease.resources),
                    "expires_at": lock_lease.expires_at.isoformat(),
                },
                "fsm_state": "6_HUMAN_APPROVAL",
            }

            lifecycle = ChangeLifecycle()
            lifecycle.advance_to(6)
            self._lifecycles[change_id] = lifecycle

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
        except Exception:
            self.blast_radius_lock.release(change_id)
            raise

    def approve_change_stage_6(
        self,
        change_id: str,
        approver_user_id: str,
        confirmed_plan_hash: str,
        confirmed_maintenance_window: str,
        approval_channel: str = "EXPLICIT_WEB_APPROVAL",
        signature_ed25519: str | None = None,
        approval_timestamp: str | None = None,
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

        if rec["status"] != "AWAITING_HUMAN_APPROVAL":
            raise RuntimeError("CHANGE_STAGE_INVALID: approval is only allowed at stage 6")
        if approver_user_id == rec["requester_user_id"]:
            raise PermissionError("SEPARATION_OF_DUTIES_REQUIRED: requester cannot approve own change")

        if confirmed_plan_hash != rec["plan_hash"]:
            self._reject_plan_approval(rec, approver_user_id, "plan hash mismatch")
            raise ValueError(
                f"PLAN_HASH_MISMATCH: Hash informado ({confirmed_plan_hash}) difere do hash do plano ({rec['plan_hash']})."
            )
        if confirmed_maintenance_window != rec["maintenance_window"]:
            self._reject_plan_approval(rec, approver_user_id, "maintenance window mismatch")
            raise ValueError("MAINTENANCE_WINDOW_MISMATCH: Janela de manutenção divergente.")

        # Public keys are provisioned out of band; the private key never reaches
        # Veronica. Canonical JSON binds the signer, plan, window and timestamp.
        public_key_pem = self.rbac.approval_public_keys.get(approver_user_id)
        if not signature_ed25519 or not approval_timestamp or not public_key_pem:
            self._reject_plan_approval(rec, approver_user_id, "Ed25519 signature or trusted key missing")
            raise PermissionError("ED25519_APPROVAL_REQUIRED: signature and trusted key required.")
        signed_payload = json.dumps(
            {
                "operator_id": approver_user_id,
                "artifact_sha256_hash": rec["plan_hash"],
                "maintenance_window_id": confirmed_maintenance_window,
                "timestamp": approval_timestamp,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            from cryptography.hazmat.primitives import serialization

            public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
            public_key.verify(base64.b64decode(signature_ed25519, validate=True), signed_payload)
        except Exception as exc:
            self.audit_log.record(
                actor=approver_user_id,
                action="INVALID_ED25519_APPROVAL",
                target=change_id,
                status="DENIED",
                details={"reason": "signature verification failed"},
            )
            self._reject_plan_approval(rec, approver_user_id, "Ed25519 signature verification failed")
            raise PermissionError("ED25519_SIGNATURE_INVALID: invalid approval signature.") from exc

        try:
            lease = self.blast_radius_lock.acquire(
                change_id, rec["blast_radius_lock"]["resources"]
            )
            rec["blast_radius_lock"]["expires_at"] = lease.expires_at.isoformat()
        except RuntimeError as exc:
            self._reject_plan_approval(rec, approver_user_id, str(exc))
            raise

        approval_token = f"APPR-{change_id}-{uuid.uuid4().hex[:10].upper()}"
        approval_data = {
            "approved": True,
            "approver_user_id": approver_user_id,
            "approval_token": approval_token,
            "plan_hash": rec["plan_hash"],
            "baseline_config_hash": rec["stages"]["stage_2_baseline"]["config_hash"],
            "maintenance_window": confirmed_maintenance_window,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "approval_timestamp": approval_timestamp,
            "signature_ed25519": signature_ed25519,
        }
        rec["stages"]["stage_6_approval"] = approval_data
        rec["status"] = "APPROVED_READY_FOR_EXECUTION"
        rec["current_stage"] = 7
        self._lifecycles[change_id].advance_to(7)
        rec["fsm_state"] = self._lifecycles[change_id].state

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

    def _reject_plan_approval(
        self, record: Dict[str, Any], actor: str, reason: str
    ) -> None:
        """Reject a failed approval attempt and release its active reservation."""
        change_id = record["change_id"]
        record["status"] = "REJECTED_APPROVAL"
        record["approval_rejection_reason"] = reason
        self.blast_radius_lock.release(change_id)
        self.audit_log.record(
            actor=actor,
            action="CHANGE_APPROVAL_REJECTED",
            target=record["device_id"],
            status="REJECTED",
            details={"change_id": change_id, "reason": reason},
        )

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

        if rec["status"] != "APPROVED_READY_FOR_EXECUTION" or rec["current_stage"] != 7:
            raise RuntimeError("CHANGE_STAGE_INVALID: execution is only allowed at stage 7")
        if approval.get("plan_hash") != rec.get("plan_hash") or approval.get("maintenance_window") != rec.get("maintenance_window"):
            raise PermissionError("APPROVAL_BINDING_MISMATCH: approval no longer matches the plan")

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
            self._lifecycles[change_id].advance_to(8)
        except Exception as exc:
            rec["status"] = "ABORTED_PRE_EXECUTION"
            self.audit_log.record(
                actor=executor_user_id,
                action="CHANGE_EXECUTION_ABORTED",
                target=device_id,
                status="ABORTED",
                details={"change_id": change_id, "error": str(exc)},
            )
            self.blast_radius_lock.release(change_id)
            raise

        rec["stages"]["stage_7_execution"] = exec_res
        rec["current_stage"] = 8
        rec["fsm_state"] = self._lifecycles[change_id].state

        # Etapa 8 — Pós-validação
        post_state = self.adapter.get_operational_state(device_id)
        verification = self.adapter.verify_change(
            device_id=device_id,
            baseline_state=baseline_state,
            post_state=post_state,
        )
        invariant_check = compare_operational_invariants(baseline_state, post_state)
        verification["invariant_check"] = invariant_check
        verification["passed"] = bool(verification.get("passed")) and invariant_check["passed"]
        finalize = getattr(self.adapter, "finalize_change", None)
        simulated_adapter = type(self.adapter).__name__ == "DigitalTwinNetworkAdapter"
        if verification["passed"] and not callable(finalize) and not simulated_adapter:
            verification["passed"] = False
            verification["summary"] = "COMMIT_FINALIZATION_UNAVAILABLE: refusing to leave an unconfirmed trial active"
        if not invariant_check["passed"]:
            verification["summary"] = (
                "POST_VALIDATION_INVARIANT_FAILED: "
                + ", ".join(invariant_check["failed_invariants"])
            )
        rec["stages"]["stage_8_post_validation"] = verification

        # Etapa 9 — Rollback automático se a pós-validação falhar
        if not verification["passed"]:
            rec["current_stage"] = 9
            self._lifecycles[change_id].advance_to(9)
            try:
                rb_res = self.rollback_orchestrator.trigger_rollback(
                    change_id=change_id,
                    device_id=device_id,
                    rollback_plan=rollback_plan,
                    reason=verification["summary"],
                    actor=executor_user_id,
                )
            except Exception:
                self.blast_radius_lock.release(change_id)
                raise
            rec["stages"]["stage_9_rollback"] = rb_res
            rec["status"] = "ROLLED_BACK_AFTER_POSTCHECK_FAILURE"
        else:
            self._lifecycles[change_id].advance_to(9)
            if callable(finalize):
                finalize(device_id=device_id, execution=exec_res)
            rec["stages"]["stage_9_rollback"] = {"triggered": False, "reason": "Pós-validação aprovada."}
            rec["status"] = "COMPLETED_VERIFIED"

        rec["current_stage"] = 10
        self._lifecycles[change_id].advance_to(10)
        rec["fsm_state"] = self._lifecycles[change_id].state
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
        self.blast_radius_lock.release(change_id)
        return rec
