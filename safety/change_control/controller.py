"""
Controlador Determinístico de Mudanças Seguras da Veronica (Seção 8 e Fase K).
Independente do modelo neural, obriga que toda alteração percorra rigorosamente
as 10 etapas:
1. Descoberta
2. Baseline
3. Análise de impacto
4. Planejamento (validação de sintaxe, OOB e commit confirmed)
5. Simulação (Digital Twin e diff unificado para co-edição no NetCanvas)
6. Aprovação humana explícita (proibida aprovação implícita ou por comando de voz — ADR-03 / R-09)
7. Execução controlada (com verificação estrita de concorrência e aborto se o baseline divergir)
8. Pós-validação
9. Rollback automático em caso de falha
10. Auditoria imutável SHA-256 (com suporte a multi-tenant)
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional
import uuid
import base64

from networking.vendors.adapters import BaseVendorAdapter
from networking.topology.graph import NetworkTopologyGraph
from safety.policies.rbac_vault import RBACPolicyManager, TamperEvidentAuditLog
from safety.rollback.manager import RollbackOrchestrator
from safety.change_control.blast_lock import BlastRadiusLock
from safety.change_control.fsm import ChangeLifecycle
from safety.rollback.validation import compare_operational_invariants

try:
    from core.contracts import NetCanvasArtifact
except ImportError:
    NetCanvasArtifact = None


_CONTROLLER_REGISTRY: Dict[str, Any] = {}


class ChangePlan(dict):
    """
    Plano determinístico de mudança segura da Veronica (Seção 8 e Fase K).
    Subclasse de dict compatível com dicionários legados e acessível via atributos,
    contendo diff unificado, relatório de pré-condições (OOB e commit confirmed),
    hash de baseline e suporte à inspeção e co-edição como artefato NetCanvas.
    """

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'ChangePlan' object has no attribute '{name}'")

    @property
    def change_id(self) -> str:
        return str(self.get("change_id", ""))

    @property
    def plan_id(self) -> str:
        return self.change_id

    @property
    def plan_hash(self) -> str:
        return str(self.get("plan_hash", ""))

    @property
    def target(self) -> str:
        return str(self.get("device_id", ""))

    @property
    def device_id(self) -> str:
        return str(self.get("device_id", ""))

    @property
    def baseline_config_hash(self) -> str:
        return str(
            self.get("baseline_config_hash")
            or self.get("stages", {}).get("stage_2_baseline", {}).get("config_hash", "")
        )

    @property
    def baseline_hash(self) -> str:
        return self.baseline_config_hash

    @property
    def unified_diff(self) -> str:
        sim = self.get("stages", {}).get("stage_5_simulation", {})
        diff = sim.get("diff", {})
        if isinstance(diff, dict):
            return str(diff.get("unified_diff", ""))
        return str(self.get("unified_diff", ""))

    @property
    def diff(self) -> str:
        return self.unified_diff

    @property
    def preconditions(self) -> Dict[str, Any]:
        return self.get("stages", {}).get("stage_4_plan", {}).get("preconditions", {})

    @property
    def commands(self) -> List[str]:
        return self.get("stages", {}).get("stage_4_plan", {}).get("commands", [])

    @property
    def candidate_config(self) -> str:
        return str(self.get("stages", {}).get("stage_5_simulation", {}).get("candidate_config", ""))

    @property
    def tenant_id(self) -> str:
        return str(self.get("tenant_id", "default"))

    def to_netcanvas_artifact(self) -> Any:
        """
        Converte o plano em artefato interativo NetCanvas para inspeção e co-edição no Workspace.
        """
        baseline = self.get("stages", {}).get("stage_2_baseline", {})
        sim = self.get("stages", {}).get("stage_5_simulation", {})
        diff_data = sim.get("diff", {})
        unified_diff = diff_data.get("unified_diff", "") if isinstance(diff_data, dict) else str(diff_data)

        proposed_text = unified_diff or json.dumps(sim, default=str)
        baseline_text = baseline.get("running_config") or json.dumps(baseline, default=str)

        if NetCanvasArtifact is not None:
            return NetCanvasArtifact(
                id=self.change_id,
                version=int(self.get("version", 1)),
                type="change_plan_10_steps",
                baseline_content=baseline_text,
                proposed_content=proposed_text,
                blast_radius_graph=self.get("stages", {}).get("stage_3_impact_analysis"),
                change_stage=str(self.get("fsm_state") or self.get("current_stage", 6)),
                sha256_hash=self.plan_hash,
            )

        return {
            "id": self.change_id,
            "version": int(self.get("version", 1)),
            "type": "change_plan_10_steps",
            "baseline_content": baseline_text,
            "proposed_content": proposed_text,
            "blast_radius_graph": self.get("stages", {}).get("stage_3_impact_analysis"),
            "change_stage": str(self.get("fsm_state") or self.get("current_stage", 6)),
            "sha256_hash": self.plan_hash,
        }

    def co_edit(self, new_commands: List[str], controller: Any | None = None) -> ChangePlan:
        """Co-edita os comandos propostos no Workspace NetCanvas antes da Etapa 6."""
        ctrl = controller or _CONTROLLER_REGISTRY.get(self.change_id)
        if ctrl is None:
            raise RuntimeError("Controller não referenciado no ChangePlan para co-edição.")
        return ctrl.co_edit_plan(self.change_id, new_commands)


class DeterministicChangeController:
    """
    Motor determinístico de controle de mudanças em 10 etapas (Seção 8 e Fase K).
    Garante sincronização com artefatos do Workspace e isolamento Multi-Tenant.
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

    def stage_change(
        self,
        target: Any,
        intent: Any,
        *,
        requester_user_id: str = "usr-eng-01",
        title: str | None = None,
        maintenance_window: str = "2026-10-18 02:00-04:00 BRT",
        origin_channel: str = "TEXT_UI",
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> ChangePlan:
        """
        Gera um ChangePlan completo percorrendo as Etapas 1 a 5 da Seção 8:
        1. Descoberta
        2. Baseline
        3. Análise de impacto
        4. Planejamento (validação sintática, OOB e commit confirmed)
        5. Simulação (Digital Twin e diff unificado)

        O ChangePlan resultante pode ser inspecionado e co-editado como um
        artefato NetCanvas no Workspace antes da Etapa 6 (Aprovação Humana).
        """
        # Extração de device_id de target
        if isinstance(target, str):
            device_id = target
        elif isinstance(target, dict):
            device_id = target.get("device_id") or target.get("id") or target.get("hostname") or str(target)
            tenant_id = target.get("tenant_id") or tenant_id
        elif hasattr(target, "device_id"):
            device_id = str(getattr(target, "device_id"))
        elif hasattr(target, "hostname"):
            device_id = str(getattr(target, "hostname"))
        else:
            device_id = str(target)

        # Extração de proposed_commands de intent
        proposed_commands: List[str] = []
        if isinstance(intent, list):
            proposed_commands = intent
        elif isinstance(intent, str):
            proposed_commands = [line.strip() for line in intent.splitlines() if line.strip()]
        elif isinstance(intent, dict):
            proposed_commands = intent.get("commands") or intent.get("proposed_commands") or []
            title = title or intent.get("title")
            requester_user_id = intent.get("requester_user_id") or requester_user_id
            maintenance_window = intent.get("maintenance_window") or maintenance_window
            origin_channel = intent.get("origin_channel") or origin_channel
            tenant_id = intent.get("tenant_id") or tenant_id
        elif hasattr(intent, "commands"):
            proposed_commands = list(getattr(intent, "commands"))
        elif hasattr(intent, "proposed_commands"):
            proposed_commands = list(getattr(intent, "proposed_commands"))

        if "commands" in kwargs and not proposed_commands:
            proposed_commands = kwargs["commands"]
        if "proposed_commands" in kwargs and not proposed_commands:
            proposed_commands = kwargs["proposed_commands"]

        if not title:
            title = f"Plano de Mudança Segura para {device_id}"

        return self.create_change_plan_stages_1_to_5(
            requester_user_id=requester_user_id,
            device_id=device_id,
            title=title,
            proposed_commands=proposed_commands,
            maintenance_window=maintenance_window,
            origin_channel=origin_channel,
            tenant_id=tenant_id,
        )

    def create_change_plan_stages_1_to_5(
        self,
        requester_user_id: str,
        device_id: str,
        title: str,
        proposed_commands: List[str],
        maintenance_window: str = "2026-10-18 02:00-04:00 BRT",
        origin_channel: str = "TEXT_UI",
        tenant_id: str = "default",
    ) -> ChangePlan:
        """
        Executa as Etapas 1 a 5 (Descoberta, Baseline, Impacto, Planejamento e Simulação)
        e estaciona o plano no estado `AWAITING_HUMAN_APPROVAL` (Etapa 6).
        Gera um ChangePlan com diff unificado, relatório de pré-condições (OOB e commit confirmed),
        e hash SHA-256 de baseline, pronto para inspeção e co-edição no NetCanvas.
        """
        self.rbac.require_permission(requester_user_id, "change:propose", tenant_id=tenant_id)

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
                tenant_id=tenant_id,
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
            tenant_id=tenant_id,
        )

        lock_resources = [device_id, *impact.get("affected_neighbors", [])]
        try:
            lock_lease = self.blast_radius_lock.acquire(change_id, lock_resources)
        except RuntimeError as exc:
            self.audit_log.record(
                actor=requester_user_id,
                action="CHANGE_BLOCKED_BLAST_RADIUS_COLLISION",
                target=device_id,
                status="BLOCKED",
                details={"change_id": change_id, "resources": lock_resources, "reason": str(exc)},
                tenant_id=tenant_id,
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

            # Verificação estrita de OOB e commit confirmed nas pré-condições
            has_commit_confirmed = any(
                any(token in cmd.lower() for token in ("commit confirmed", "commit trial"))
                for cmd in proposed_commands
            )
            is_write_op = any(
                not cmd.strip().lower().startswith(("show", "display", "get", "ping", "traceroute"))
                for cmd in proposed_commands
            )
            oob_avail = bool(preconditions.get("oob_available", True))
            preconditions["oob_verified"] = oob_avail
            preconditions["commit_confirmed_verified"] = has_commit_confirmed or not is_write_op
            preconditions["commit_confirmed_present"] = has_commit_confirmed

            if is_write_op and not has_commit_confirmed:
                if any(k in " ".join(proposed_commands).lower() for k in ("bgp", "ospf", "isis", "interface", "ip address")):
                    preconditions["blocking_reasons"].append(
                        "SAFE_CHANGE_GUARD_BLOCKED: Exigência estrita de 'commit confirmed' ou 'commit trial' com timer para prevenção de lockout."
                    )
                    preconditions["passed"] = False

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
                    tenant_id=tenant_id,
                )
                raise ValueError(
                    f"SAFE_CHANGE_GUARD_BLOCKED: Comandos inválidos ou pré-condições violadas. "
                    f"Rejeitados={validation['rejected_commands']}; Motivos={preconditions['blocking_reasons']}"
                )

            rollback_plan = self.adapter.prepare_rollback(device_id, backup, proposed_commands)
            plan_hash = self._compute_plan_hash(
                device_id, running_cfg["config_hash"], validation["valid_commands"]
            )

            # Etapa 5 — Simulação em Digital Twin (Candidate + Diff unificado)
            candidate = self.adapter.generate_candidate(device_id, validation["valid_commands"])
            diff = self.adapter.compare_configurations(
                running_cfg["running_config"], candidate["candidate_config"]
            )
            unified_diff_str = diff.get("unified_diff", "") if isinstance(diff, dict) else str(diff)

            record = ChangePlan({
                "change_id": change_id,
                "title": title,
                "device_id": device_id,
                "requester_user_id": requester_user_id,
                "origin_channel": origin_channel,
                "maintenance_window": maintenance_window,
                "tenant_id": tenant_id,
                "version": 1,
                "status": "AWAITING_HUMAN_APPROVAL",
                "current_stage": 6,
                "plan_hash": plan_hash,
                "baseline_config_hash": baseline["config_hash"],
                "unified_diff": unified_diff_str,
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
                        "candidate_config": candidate.get("candidate_config", ""),
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
            })
            _CONTROLLER_REGISTRY[change_id] = self

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
                tenant_id=tenant_id,
            )
            record["stages"]["stage_10_audit"].append(audit_entry["entry_id"])
            self.change_records[change_id] = record
            return record
        except Exception:
            self.blast_radius_lock.release(change_id)
            raise

    def co_edit_plan(
        self,
        change_id: str,
        new_commands: List[str],
        editor_user_id: str = "usr-eng-01",
        tenant_id: str | None = None,
    ) -> ChangePlan:
        """
        Co-edição em tempo real de comandos do plano antes da Etapa 6 (Aprovação Humana).
        Recalcula validação sintática, pré-condições (OOB e commit confirmed),
        Digital Twin candidate, diff unificado e o hash SHA-256 do plano.
        """
        if change_id not in self.change_records:
            raise KeyError(f"Plano de mudança '{change_id}' não encontrado.")
        rec = self.change_records[change_id]

        if rec["status"] != "AWAITING_HUMAN_APPROVAL" or rec["current_stage"] != 6:
            raise RuntimeError(
                f"COEDIT_NOT_ALLOWED: Co-edição de plano somente é permitida na Etapa 6 antes da aprovação. "
                f"Estado atual: {rec['status']}"
            )

        eff_tenant = tenant_id or rec.get("tenant_id", "default")
        self.rbac.require_permission(editor_user_id, "change:propose", tenant_id=eff_tenant)

        device_id = rec["device_id"]
        validation = self.adapter.validate_commands(device_id, new_commands)
        preconditions = self.adapter.check_preconditions(device_id, new_commands)

        has_commit_confirmed = any(
            any(token in cmd.lower() for token in ("commit confirmed", "commit trial"))
            for cmd in new_commands
        )
        is_write_op = any(
            not cmd.strip().lower().startswith(("show", "display", "get", "ping", "traceroute"))
            for cmd in new_commands
        )
        preconditions["oob_verified"] = bool(preconditions.get("oob_available", True))
        preconditions["commit_confirmed_verified"] = has_commit_confirmed or not is_write_op
        preconditions["commit_confirmed_present"] = has_commit_confirmed

        if is_write_op and not has_commit_confirmed:
            if any(k in " ".join(new_commands).lower() for k in ("bgp", "ospf", "isis", "interface", "ip address")):
                preconditions["blocking_reasons"].append(
                    "SAFE_CHANGE_GUARD_BLOCKED: Exigência estrita de 'commit confirmed' ou 'commit trial' com timer para prevenção de lockout."
                )
                preconditions["passed"] = False

        if not validation["all_valid"] or not preconditions["passed"]:
            raise ValueError(
                f"SAFE_CHANGE_GUARD_BLOCKED: Comandos inválidos ou pré-condições violadas na co-edição. "
                f"Rejeitados={validation['rejected_commands']}; Motivos={preconditions['blocking_reasons']}"
            )

        running_cfg = self.adapter.get_running_config(device_id)
        backup = rec["stages"]["stage_2_baseline"]
        rollback_plan = self.adapter.prepare_rollback(device_id, backup, validation["valid_commands"])
        new_plan_hash = self._compute_plan_hash(
            device_id, running_cfg["config_hash"], validation["valid_commands"]
        )

        candidate = self.adapter.generate_candidate(device_id, validation["valid_commands"])
        diff = self.adapter.compare_configurations(
            running_cfg["running_config"], candidate["candidate_config"]
        )
        unified_diff_str = diff.get("unified_diff", "") if isinstance(diff, dict) else str(diff)

        rec["plan_hash"] = new_plan_hash
        rec["stages"]["stage_4_plan"]["commands"] = validation["valid_commands"]
        rec["stages"]["stage_4_plan"]["preconditions"] = preconditions
        rec["stages"]["stage_4_plan"]["rollback_plan"] = rollback_plan
        rec["stages"]["stage_5_simulation"]["candidate_hash"] = candidate["candidate_hash"]
        rec["stages"]["stage_5_simulation"]["candidate_config"] = candidate.get("candidate_config", "")
        rec["stages"]["stage_5_simulation"]["diff"] = diff
        rec["unified_diff"] = unified_diff_str
        rec["version"] = int(rec.get("version", 1)) + 1
        rec["updated_at"] = datetime.now(timezone.utc).isoformat()

        audit_entry = self.audit_log.record(
            actor=editor_user_id,
            action="CHANGE_PLAN_COEDITED",
            target=device_id,
            status="AWAITING_HUMAN_APPROVAL",
            details={
                "change_id": change_id,
                "new_plan_hash": new_plan_hash,
                "version": rec["version"],
                "commands": validation["valid_commands"],
            },
            tenant_id=eff_tenant,
        )
        rec["stages"]["stage_10_audit"].append(audit_entry["entry_id"])

        plan = ChangePlan(rec)
        _CONTROLLER_REGISTRY[change_id] = self
        self.change_records[change_id] = plan
        return plan

    def authorize_change(
        self,
        plan_id: str,
        approver_id: str,
        token: str | None = None,
        window_id: str | None = None,
        *,
        approval_channel: str = "EXPLICIT_WEB_APPROVAL",
        signature_ed25519: str | None = None,
        approval_timestamp: str | None = None,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Autoriza explicitamente o plano de mudança na Etapa 6 (Seção 8 e ADR-03).
        Bloqueia 100% tentativas originadas por voz (`VOICE`).
        """
        if approval_channel.upper() == "VOICE":
            self.audit_log.record(
                actor=approver_id,
                action="VOICE_APPROVAL_BLOCKED",
                target=plan_id,
                status="DENIED",
                details={"reason": "O modo de voz não pode autorizar alterações em equipamentos."},
                tenant_id=tenant_id or "default",
            )
            raise PermissionError(
                "VOICE_APPROVAL_PROHIBITED: O modo de voz não pode autorizar alterações em equipamentos. "
                "Exige aprovação humana formal vinculada ao plano."
            )

        if plan_id not in self.change_records:
            raise KeyError(f"Plano de mudança '{plan_id}' não encontrado.")
        rec = self.change_records[plan_id]

        confirmed_plan_hash = kwargs.get("confirmed_plan_hash") or token or rec["plan_hash"]
        confirmed_maintenance_window = kwargs.get("confirmed_maintenance_window") or window_id or rec["maintenance_window"]
        sig = signature_ed25519 or kwargs.get("signature_ed25519")
        ts = approval_timestamp or kwargs.get("approval_timestamp")

        return self.approve_change_stage_6(
            change_id=plan_id,
            approver_user_id=approver_id,
            confirmed_plan_hash=confirmed_plan_hash,
            confirmed_maintenance_window=confirmed_maintenance_window,
            approval_channel=approval_channel,
            signature_ed25519=sig,
            approval_timestamp=ts,
            tenant_id=tenant_id,
        )

    def approve_change_stage_6(
        self,
        change_id: str,
        approver_user_id: str,
        confirmed_plan_hash: str,
        confirmed_maintenance_window: str,
        approval_channel: str = "EXPLICIT_WEB_APPROVAL",
        signature_ed25519: str | None = None,
        approval_timestamp: str | None = None,
        tenant_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Etapa 6 — Aprovação Humana Explícita.
        Bloqueia aprovações originadas por canal de voz (`VOICE`) conforme ADR-03 / R-09 e Seção 16.8!
        """
        if approval_channel.upper() == "VOICE":
            self.audit_log.record(
                actor=approver_user_id,
                action="VOICE_APPROVAL_BLOCKED",
                target=change_id,
                status="DENIED",
                details={"reason": "O modo de voz não pode autorizar alterações em equipamentos."},
                tenant_id=tenant_id or "default",
            )
            raise PermissionError(
                "VOICE_APPROVAL_PROHIBITED: O modo de voz não pode autorizar alterações em equipamentos. "
                "Exige aprovação humana formal vinculada ao plano."
            )

        if change_id not in self.change_records:
            raise KeyError(f"Plano de mudança '{change_id}' não encontrado.")
        rec = self.change_records[change_id]

        eff_tenant = tenant_id or rec.get("tenant_id", "default")
        self.rbac.require_permission(approver_user_id, "change:approve", tenant_id=eff_tenant)

        if rec["status"] != "AWAITING_HUMAN_APPROVAL":
            raise RuntimeError("CHANGE_STAGE_INVALID: approval is only allowed at stage 6")
        if approver_user_id == rec["requester_user_id"]:
            raise PermissionError("SEPARATION_OF_DUTIES_REQUIRED: requester cannot approve own change")

        if confirmed_plan_hash != rec["plan_hash"]:
            self._reject_plan_approval(rec, approver_user_id, "plan hash mismatch", tenant_id=eff_tenant)
            raise ValueError(
                f"PLAN_HASH_MISMATCH: Hash informado ({confirmed_plan_hash}) difere do hash do plano ({rec['plan_hash']})."
            )
        if confirmed_maintenance_window != rec["maintenance_window"]:
            self._reject_plan_approval(rec, approver_user_id, "maintenance window mismatch", tenant_id=eff_tenant)
            raise ValueError("MAINTENANCE_WINDOW_MISMATCH: Janela de manutenção divergente.")

        public_key_pem = self.rbac.approval_public_keys.get(approver_user_id)
        if not signature_ed25519 or not approval_timestamp or not public_key_pem:
            self._reject_plan_approval(rec, approver_user_id, "Ed25519 signature or trusted key missing", tenant_id=eff_tenant)
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
                tenant_id=eff_tenant,
            )
            self._reject_plan_approval(rec, approver_user_id, "Ed25519 signature verification failed", tenant_id=eff_tenant)
            raise PermissionError("ED25519_SIGNATURE_INVALID: invalid approval signature.") from exc

        try:
            lease = self.blast_radius_lock.acquire(
                change_id, rec["blast_radius_lock"]["resources"]
            )
            rec["blast_radius_lock"]["expires_at"] = lease.expires_at.isoformat()
        except RuntimeError as exc:
            self._reject_plan_approval(rec, approver_user_id, str(exc), tenant_id=eff_tenant)
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
            tenant_id=eff_tenant,
        )
        rec["stages"]["stage_10_audit"].append(audit_entry["entry_id"])
        return rec

    def _reject_plan_approval(
        self, record: Dict[str, Any], actor: str, reason: str, tenant_id: str = "default"
    ) -> None:
        """Rejeita a tentativa de aprovação e libera o bloqueio ativo."""
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
            tenant_id=tenant_id,
        )

    def execute_change(
        self,
        plan_id: str,
        authorization_receipt: Any = None,
        executor_user_id: str = "usr-eng-01",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Executa uma mudança autorizada em conformidade com docs/INTERFACES.md.
        Extrai o token de autorização e despacha para as Etapas 7 a 10.
        """
        approval_token: str | None = None
        if isinstance(authorization_receipt, dict):
            approval_token = (
                authorization_receipt.get("approval_token")
                or authorization_receipt.get("stages", {}).get("stage_6_approval", {}).get("approval_token")
            )
        elif isinstance(authorization_receipt, str):
            approval_token = authorization_receipt

        if not approval_token:
            approval_token = kwargs.get("approval_token")

        if not approval_token:
            raise PermissionError("UNAUTHORIZED_EXECUTION_BLOCKED: approval_token ausente.")

        return self.execute_change_stages_7_to_10(
            change_id=plan_id,
            executor_user_id=executor_user_id,
            approval_token=approval_token,
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
        Realiza aborto imediato com CONCURRENT_MODIFICATION_DETECTED caso o hash
        de baseline tenha divergido antes da aplicação dos comandos.
        """
        if change_id not in self.change_records:
            raise KeyError(f"Plano de mudança '{change_id}' não encontrado.")
        rec = self.change_records[change_id]
        tenant_id = rec.get("tenant_id", "default")

        self.rbac.require_permission(executor_user_id, "change:execute", tenant_id=tenant_id)

        approval = rec["stages"].get("stage_6_approval")
        if not approval or not approval.get("approved") or approval.get("approval_token") != approval_token:
            self.audit_log.record(
                actor=executor_user_id,
                action="UNAUTHORIZED_EXECUTION_ATTEMPT",
                target=rec["device_id"],
                status="BLOCKED",
                details={"change_id": change_id},
                tenant_id=tenant_id,
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

        # Etapa 7 — Verificação estrita de concorrência antes da aplicação
        current_running = self.adapter.get_running_config(device_id)
        current_baseline_hash = current_running.get("config_hash")
        if current_baseline_hash and current_baseline_hash != expected_hash:
            rec["status"] = "ABORTED_PRE_EXECUTION"
            self.blast_radius_lock.release(change_id)
            self.audit_log.record(
                actor=executor_user_id,
                action="CONCURRENT_MODIFICATION_ABORT",
                target=device_id,
                status="ABORTED",
                details={
                    "change_id": change_id,
                    "expected_hash": expected_hash,
                    "current_hash": current_baseline_hash,
                    "reason": "CONCURRENT_MODIFICATION_DETECTED",
                },
                tenant_id=tenant_id,
            )
            raise RuntimeError(
                f"CONCURRENT_MODIFICATION_DETECTED: O hash da configuração atual ({current_baseline_hash}) "
                f"diverge do baseline ({expected_hash}). Outro operador modificou o equipamento."
            )

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
                tenant_id=tenant_id,
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
                try:
                    finalize(device_id=device_id, execution=exec_res)
                except TypeError:
                    finalize(device_id=device_id)
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
            tenant_id=tenant_id,
        )
        rec["stages"]["stage_10_audit"].append(audit_entry["entry_id"])
        self.blast_radius_lock.release(change_id)
        return rec

    def trigger_abort_and_rollback(
        self,
        plan_id: str,
        reason: str,
        actor: str = "usr-eng-01",
    ) -> Dict[str, Any]:
        """Aborta imediatamente a mudança e aciona rollback orquestrado."""
        if plan_id not in self.change_records:
            raise KeyError(f"Plano de mudança '{plan_id}' não encontrado.")
        rec = self.change_records[plan_id]
        tenant_id = rec.get("tenant_id", "default")
        device_id = rec["device_id"]
        rollback_plan = rec.get("stages", {}).get("stage_4_plan", {}).get("rollback_plan", {})
        rb_res = self.rollback_orchestrator.trigger_rollback(
            change_id=plan_id,
            device_id=device_id,
            rollback_plan=rollback_plan,
            reason=reason,
            actor=actor,
        )
        rec["stages"]["stage_9_rollback"] = rb_res
        rec["status"] = "ROLLED_BACK_ABORTED"
        self.blast_radius_lock.release(plan_id)
        self.audit_log.record(
            actor=actor,
            action="CHANGE_ABORTED_AND_ROLLED_BACK",
            target=device_id,
            status="ROLLED_BACK",
            details={"change_id": plan_id, "reason": reason, "rollback_result": rb_res},
            tenant_id=tenant_id,
        )
        return rb_res
