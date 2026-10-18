"""
Gerenciador de Rollback e Recuperação Operacional da Veronica (Seção 8 Etapa 9).
"""

from __future__ import annotations

from typing import Any, Dict
from networking.vendors.adapters import BaseVendorAdapter
from safety.policies.rbac_vault import TamperEvidentAuditLog


class RollbackOrchestrator:
    """Executa e verifica reversões determinísticas quando a pós-validação falha."""

    def __init__(self, adapter: BaseVendorAdapter, audit_log: TamperEvidentAuditLog) -> None:
        self.adapter = adapter
        self.audit_log = audit_log

    def trigger_rollback(
        self,
        change_id: str,
        device_id: str,
        rollback_plan: Dict[str, Any],
        reason: str,
        actor: str = "SYSTEM_WATCHDOG",
    ) -> Dict[str, Any]:
        result = self.adapter.rollback_change(device_id, rollback_plan)
        status = "ROLLED_BACK_SUCCESS" if result.get("rolled_back") else "ROLLBACK_FAILED_ESCALATE_OOB"
        self.audit_log.record(
            actor=actor,
            action="CHANGE_ROLLBACK",
            target=device_id,
            status=status,
            details={
                "change_id": change_id,
                "reason": reason,
                "restored_config_hash": result.get("restored_config_hash"),
                "commands": result.get("rollback_commands_executed"),
            },
        )
        return result
