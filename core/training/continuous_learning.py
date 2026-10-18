"""
Sistema de Aprendizado Contínuo e Evolução Controlada da Veronica (Seção 10).
Implementa:
- Armazenamento estruturado de incidentes operacionais resolvidos e validados (`record_validated_incident`)
- Registro de correções humanas (`submit_human_correction`) em estado `PENDING_VERIFICATION`
- Aprovação e curadoria por engenheiro autorizado (`verify_and_approve_correction`) antes de
  ingressar no dataset de treinamento versionado
- Garantias determinísticas de segurança:
  1. Conteúdo externo ou recém-ingerido jamais modifica políticas de segurança (RBAC/Guardrails).
  2. Nenhum incidente ou correção atualiza pesos em produção automaticamente.
  3. Segredos são mascarados (`scrub_secrets`) e tentativas de Prompt Injection são bloqueadas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from knowledge.ingestion.ingestor import detect_prompt_injection, scrub_secrets
from safety.policies.rbac_vault import RBACPolicyManager, TamperEvidentAuditLog


AUTHORIZED_REVIEWER_ROLES = {"NETWORK_ENGINEER", "CHANGE_APPROVER", "AI_TRAINER", "ADMIN"}


@dataclass
class ValidatedIncidentRecord:
    incident_id: str
    timestamp: str
    devices: List[str]
    vendor: str
    os_name: str
    os_version: str
    domain: str
    symptom: str
    root_cause: str
    diagnostic_steps_executed: List[str]
    resolution_commands: List[str]
    resolution_plan_id: str
    rollback_tested: bool
    recorded_by: str
    human_approved_for_training: bool = False
    approved_by: Optional[str] = None
    dataset_version: Optional[str] = None
    secrets_redacted: int = 0
    security_policies_modified: bool = False
    production_weights_updated: bool = False


@dataclass
class HumanCorrectionRecord:
    correction_id: str
    submitted_at: str
    submitted_by: str
    vendor: str
    os_name: str
    os_version: str
    domain: str
    prompt: str
    incorrect_output: str
    corrected_output: str
    evidence_reference: str
    status: str = "PENDING_VERIFICATION"  # PENDING_VERIFICATION, APPROVED_FOR_TRAINING, REJECTED
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    reviewer_notes: str = ""
    dataset_version: Optional[str] = None
    security_quarantined: bool = False
    quarantine_reasons: List[str] = field(default_factory=list)
    secrets_redacted: int = 0
    security_policies_modified: bool = False
    production_weights_updated: bool = False


class ContinuousLearningPipeline:
    """
    Pipeline de Aprendizado Contínuo da Veronica (Seção 10).
    Separa rigorosamente:
    1. Indexação imediata na base documental (via KnowledgeIngestor/RAG)
    2. Registro de incidentes e correções humanas em quarentena de verificação (`PENDING_VERIFICATION`)
    3. Dataset de treinamento versionado (somente itens verificados e aprovados por humano autorizado)
    4. Treinamento em ambiente isolado (`TRAINING`), sem jamais alterar pesos em `PRODUCTION`
       ou políticas de segurança automaticamente.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        rbac: Optional[RBACPolicyManager] = None,
        audit_log: Optional[TamperEvidentAuditLog] = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir is not None:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.rbac = rbac or RBACPolicyManager()
        self.audit_log = audit_log if audit_log is not None else TamperEvidentAuditLog()

        self.validated_incidents: Dict[str, ValidatedIncidentRecord] = {}
        self.corrections: Dict[str, HumanCorrectionRecord] = {}
        self.versioned_training_dataset: List[Dict[str, Any]] = []

        # Invariantes imutáveis de segurança da Seção 10
        self.auto_update_production_weights: bool = False
        self.allow_external_policy_modification: bool = False

    def _verify_authorized_engineer(self, user_id: str) -> Dict[str, Any]:
        user = self.rbac.users.get(user_id)
        if not user or user.get("role") not in AUTHORIZED_REVIEWER_ROLES:
            role = user.get("role", "UNKNOWN") if user else "UNKNOWN"
            raise PermissionError(
                f"UNAUTHORIZED_REVIEWER: Usuário '{user_id}' (papel={role}) não possui "
                f"autorização para aprovar dados de treinamento neural."
            )
        return user

    def record_validated_incident(
        self,
        recorded_by: str,
        devices: List[str],
        symptom: str,
        root_cause: str,
        diagnostic_steps_executed: List[str],
        resolution_commands: List[str],
        resolution_plan_id: str,
        vendor: str = "Generic",
        os_name: str = "Unspecified",
        os_version: str = "Unspecified",
        domain: str = "networking.operations",
        rollback_tested: bool = True,
        approve_for_training: bool = False,
        approver_id: Optional[str] = None,
        dataset_version: str = "v1.1.0-incidents",
    ) -> Dict[str, Any]:
        """
        Registra um incidente operacional resolvido e validado em ambiente real ou laboratório.
        Garante higienização de segredos, detecção de prompt injection e proibição de
        atualização automática de pesos em produção.
        """
        self._verify_authorized_engineer(recorded_by)
        if not rollback_tested:
            raise ValueError(
                "ROLLBACK_NOT_TESTED: Apenas incidentes com plano de rollback validado podem "
                "ser registrados como aprendizado operacional."
            )

        combined_text = "\n".join(
            [symptom, root_cause] + diagnostic_steps_executed + resolution_commands
        )
        _, red_sym = scrub_secrets(symptom)
        clean_symptom, _ = scrub_secrets(symptom)
        clean_root_cause, red_rc = scrub_secrets(root_cause)
        clean_diag_steps: List[str] = []
        clean_res_cmds: List[str] = []
        total_redacted = red_sym + red_rc

        for step in diagnostic_steps_executed:
            c_step, r_cnt = scrub_secrets(step)
            clean_diag_steps.append(c_step)
            total_redacted += r_cnt

        for cmd in resolution_commands:
            c_cmd, r_cnt = scrub_secrets(cmd)
            clean_res_cmds.append(c_cmd)
            total_redacted += r_cnt

        injected, reasons = detect_prompt_injection(combined_text)
        if injected:
            self.audit_log.record(
                actor=recorded_by,
                action="INCIDENT_RECORD_BLOCKED_INJECTION",
                target=resolution_plan_id,
                status="BLOCKED",
                details={"reasons": reasons, "security_policies_modified": False},
            )
            raise ValueError(
                f"EXTERNAL_CONTENT_POLICY_GUARD: Tentativa de injeção ou comando destrutivo "
                f"detectada no relato de incidente: {'; '.join(reasons)}"
            )

        incident_id = f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()

        approved_flag = False
        effective_approver: Optional[str] = None
        effective_ds_version: Optional[str] = None

        if approve_for_training:
            reviewer = approver_id or recorded_by
            self._verify_authorized_engineer(reviewer)
            approved_flag = True
            effective_approver = reviewer
            effective_ds_version = dataset_version

        record = ValidatedIncidentRecord(
            incident_id=incident_id,
            timestamp=now_iso,
            devices=list(devices),
            vendor=vendor,
            os_name=os_name,
            os_version=os_version,
            domain=domain,
            symptom=clean_symptom,
            root_cause=clean_root_cause,
            diagnostic_steps_executed=clean_diag_steps,
            resolution_commands=clean_res_cmds,
            resolution_plan_id=resolution_plan_id,
            rollback_tested=rollback_tested,
            recorded_by=recorded_by,
            human_approved_for_training=approved_flag,
            approved_by=effective_approver,
            dataset_version=effective_ds_version,
            secrets_redacted=total_redacted,
            security_policies_modified=False,
            production_weights_updated=False,
        )
        self.validated_incidents[incident_id] = record

        if approved_flag:
            self._append_to_versioned_dataset(
                sample_id=f"sample-{incident_id.lower()}",
                dataset_version=dataset_version,
                domain=domain,
                vendor=vendor,
                os_name=os_name,
                os_version=os_version,
                verified_by=effective_approver or recorded_by,
                source_ref=incident_id,
                prompt=f"Pergunta: Incidente em {', '.join(devices)} ({vendor} {os_name} {os_version}): {clean_symptom}",
                completion=(
                    f"Resposta: Causa raiz confirmada: {clean_root_cause}. "
                    f"Diagnostico: {'; '.join(clean_diag_steps)}. "
                    f"Resolucao validada ({resolution_plan_id}): {'; '.join(clean_res_cmds)}."
                ),
            )

        self.audit_log.record(
            actor=recorded_by,
            action="VALIDATED_INCIDENT_RECORDED",
            target=incident_id,
            status="APPROVED_FOR_DATASET" if approved_flag else "STORED_PENDING_CURATION",
            details={
                "devices": devices,
                "resolution_plan_id": resolution_plan_id,
                "human_approved_for_training": approved_flag,
                "secrets_redacted": total_redacted,
                "security_policies_modified": False,
                "production_weights_updated": False,
            },
        )
        return asdict(record)

    def submit_human_correction(
        self,
        submitted_by: str,
        prompt: str,
        incorrect_output: str,
        corrected_output: str,
        vendor: str = "Generic",
        os_name: str = "Unspecified",
        os_version: str = "Unspecified",
        domain: str = "networking.general",
        evidence_reference: str = "Revisão técnica de engenharia",
    ) -> Dict[str, Any]:
        """
        Registra uma correção humana sobre uma resposta da Veronica.
        Toda correção nasce obrigatoriamente em estado `PENDING_VERIFICATION` e
        JAMAIS entra no dataset de treinamento versionado nem altera pesos antes de
        `verify_and_approve_correction` por um engenheiro autorizado.
        """
        clean_prompt, r1 = scrub_secrets(prompt)
        clean_incorrect, r2 = scrub_secrets(incorrect_output)
        clean_corrected, r3 = scrub_secrets(corrected_output)
        total_redacted = r1 + r2 + r3

        injected, reasons = detect_prompt_injection(f"{clean_prompt}\n{clean_corrected}")

        correction_id = f"CORR-{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()

        record = HumanCorrectionRecord(
            correction_id=correction_id,
            submitted_at=now_iso,
            submitted_by=submitted_by,
            vendor=vendor,
            os_name=os_name,
            os_version=os_version,
            domain=domain,
            prompt=clean_prompt,
            incorrect_output=clean_incorrect,
            corrected_output=clean_corrected,
            evidence_reference=evidence_reference,
            status="PENDING_VERIFICATION",
            verified_by=None,
            verified_at=None,
            dataset_version=None,
            security_quarantined=injected,
            quarantine_reasons=reasons,
            secrets_redacted=total_redacted,
            security_policies_modified=False,
            production_weights_updated=False,
        )
        self.corrections[correction_id] = record

        self.audit_log.record(
            actor=submitted_by,
            action="HUMAN_CORRECTION_SUBMITTED",
            target=correction_id,
            status="QUARANTINED" if injected else "PENDING_VERIFICATION",
            details={
                "vendor": vendor,
                "os_name": os_name,
                "security_quarantined": injected,
                "quarantine_reasons": reasons,
                "secrets_redacted": total_redacted,
                "security_policies_modified": False,
                "production_weights_updated": False,
            },
        )
        return asdict(record)

    def verify_and_approve_correction(
        self,
        correction_id: str,
        reviewer_id: str,
        dataset_version: str = "v1.1.0-curated",
        reviewer_notes: str = "Correção técnica validada contra documentação oficial.",
    ) -> Dict[str, Any]:
        """
        Verifica e aprova uma correção humana pendente (`PENDING_VERIFICATION`),
        promovendo-a para o dataset de treinamento versionado sem atualizar automaticamente
        os pesos do modelo em produção.
        """
        self._verify_authorized_engineer(reviewer_id)

        if correction_id not in self.corrections:
            raise KeyError(f"Correção '{correction_id}' não encontrada.")

        record = self.corrections[correction_id]
        if record.security_quarantined:
            self.audit_log.record(
                actor=reviewer_id,
                action="CORRECTION_APPROVAL_BLOCKED_QUARANTINE",
                target=correction_id,
                status="BLOCKED",
                details={"reasons": record.quarantine_reasons},
            )
            raise ValueError(
                f"QUARANTINED_CORRECTION_BLOCKED: A correção '{correction_id}' contém padrões de "
                f"violação de segurança ou comando destrutivo e não pode entrar no dataset: "
                f"{'; '.join(record.quarantine_reasons)}"
            )

        if record.status != "PENDING_VERIFICATION":
            raise ValueError(
                f"INVALID_CORRECTION_STATE: Correção '{correction_id}' está em estado '{record.status}', "
                f"esperado 'PENDING_VERIFICATION'."
            )

        record.status = "APPROVED_FOR_TRAINING"
        record.verified_by = reviewer_id
        record.verified_at = datetime.now(timezone.utc).isoformat()
        record.reviewer_notes = reviewer_notes
        record.dataset_version = dataset_version
        record.security_policies_modified = False
        record.production_weights_updated = False

        sample = self._append_to_versioned_dataset(
            sample_id=f"sample-{correction_id.lower()}",
            dataset_version=dataset_version,
            domain=record.domain,
            vendor=record.vendor,
            os_name=record.os_name,
            os_version=record.os_version,
            verified_by=reviewer_id,
            source_ref=correction_id,
            prompt=f"Pergunta: {record.prompt}",
            completion=f"Resposta: {record.corrected_output}",
        )

        self.audit_log.record(
            actor=reviewer_id,
            action="HUMAN_CORRECTION_APPROVED",
            target=correction_id,
            status="APPROVED_FOR_TRAINING",
            details={
                "sample_id": sample["sample_id"],
                "dataset_version": dataset_version,
                "security_policies_modified": False,
                "production_weights_updated": False,
            },
        )
        return asdict(record)

    def reject_correction(
        self,
        correction_id: str,
        reviewer_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Rejeita uma correção humana, impedindo sua entrada no dataset de treinamento."""
        self._verify_authorized_engineer(reviewer_id)
        if correction_id not in self.corrections:
            raise KeyError(f"Correção '{correction_id}' não encontrada.")

        record = self.corrections[correction_id]
        record.status = "REJECTED"
        record.verified_by = reviewer_id
        record.verified_at = datetime.now(timezone.utc).isoformat()
        record.reviewer_notes = reason

        self.audit_log.record(
            actor=reviewer_id,
            action="HUMAN_CORRECTION_REJECTED",
            target=correction_id,
            status="REJECTED",
            details={"reason": reason},
        )
        return asdict(record)

    def _append_to_versioned_dataset(
        self,
        sample_id: str,
        dataset_version: str,
        domain: str,
        vendor: str,
        os_name: str,
        os_version: str,
        verified_by: str,
        source_ref: str,
        prompt: str,
        completion: str,
    ) -> Dict[str, Any]:
        entry = {
            "sample_id": sample_id,
            "dataset_version": dataset_version,
            "domain": domain,
            "vendor": vendor,
            "os_name": os_name,
            "os_version": os_version,
            "verified_by_human": True,
            "verified_by": verified_by,
            "source_ref": source_ref,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "completion": completion,
            "text": f"{prompt} {completion}",
        }
        self.versioned_training_dataset.append(entry)
        if self.storage_dir is not None:
            jsonl_path = self.storage_dir / f"dataset_{dataset_version}.jsonl"
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def get_training_corpus(self, dataset_version: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Retorna apenas as amostras verificadas e aprovadas por engenheiro autorizado,
        no formato consumido pelo `VeronicaTrainer`.
        Itens em `PENDING_VERIFICATION` ou `REJECTED` jamais são incluídos.
        """
        samples: List[Dict[str, str]] = []
        for item in self.versioned_training_dataset:
            if not item.get("verified_by_human", False):
                continue
            if dataset_version is not None and item.get("dataset_version") != dataset_version:
                continue
            samples.append(
                {
                    "id": item["sample_id"],
                    "domain": item["domain"],
                    "vendor": item["vendor"],
                    "os": f"{item['os_name']} {item['os_version']}".strip(),
                    "text": item["text"],
                }
            )
        return samples
