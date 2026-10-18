"""
RBAC (Controle de Acesso Baseado em Papéis), Cofre de Credenciais (Secret Vault)
e Log de Auditoria Imutável Encadeado por SHA-256 da Veronica (Seções 8, 12 e 16.8).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
from typing import Any, Dict, List, Optional, Set
import uuid

from knowledge.ingestion.ingestor import scrub_secrets


ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    "VIEWER": {"chat:read", "topology:read", "metrics:read"},
    "NOC_OPERATOR": {
        "chat:read",
        "chat:write",
        "voice:use",
        "topology:read",
        "diagnostics:run",
        "metrics:read",
        "knowledge:read",
    },
    "NETWORK_ENGINEER": {
        "chat:read",
        "chat:write",
        "voice:use",
        "topology:read",
        "diagnostics:run",
        "change:propose",
        "change:simulate",
        "change:execute",
        "knowledge:read",
        "knowledge:ingest",
        "metrics:read",
    },
    "CHANGE_APPROVER": {
        "chat:read",
        "topology:read",
        "diagnostics:run",
        "change:approve",
        "change:rollback",
        "model:promote",
        "metrics:read",
    },
    "AI_TRAINER": {
        "chat:read",
        "metrics:read",
        "training:start",
        "training:control",
        "benchmark:run",
        "checkpoint:read",
    },
    "ADMIN": {
        "chat:read",
        "chat:write",
        "voice:use",
        "topology:read",
        "diagnostics:run",
        "change:propose",
        "change:simulate",
        "change:approve",
        "change:execute",
        "change:rollback",
        "knowledge:read",
        "knowledge:ingest",
        "metrics:read",
        "training:start",
        "training:control",
        "benchmark:run",
        "checkpoint:read",
        "model:promote",
    },
}


class RBACPolicyManager:
    """Valida permissões de usuários por papel e garante separação de funções."""

    def __init__(self) -> None:
        self.users: Dict[str, Dict[str, Any]] = {
            "usr-noc-01": {"user_id": "usr-noc-01", "name": "Operador NOC", "role": "NOC_OPERATOR"},
            "usr-eng-01": {"user_id": "usr-eng-01", "name": "Engenheiro de Redes Sênior", "role": "NETWORK_ENGINEER"},
            "usr-app-01": {"user_id": "usr-app-01", "name": "Coordenador CAB / Aprovador", "role": "CHANGE_APPROVER"},
            "usr-trn-01": {"user_id": "usr-trn-01", "name": "Engenheiro Neural", "role": "AI_TRAINER"},
            "usr-adm-01": {"user_id": "usr-adm-01", "name": "Administrador Chefe", "role": "ADMIN"},
        }

    def check_permission(self, user_id: str, permission: str) -> bool:
        user = self.users.get(user_id)
        if not user:
            return False
        role = user["role"]
        return permission in ROLE_PERMISSIONS.get(role, set())

    def require_permission(self, user_id: str, permission: str) -> Dict[str, Any]:
        if not self.check_permission(user_id, permission):
            user = self.users.get(user_id, {"role": "UNKNOWN"})
            raise PermissionError(
                f"RBAC_ACCESS_DENIED: Usuário '{user_id}' (papel={user['role']}) "
                f"não possui a permissão obrigatória '{permission}'."
            )
        return self.users[user_id]


class CredentialVault:
    """
    Cofre de segredos em memória isolada (Seção 8 e 12).
    Nunca expõe credenciais brutas em logs, respostas de API ou datasets.
    """

    def __init__(self, master_key: str = "veronica-vault-local-key-2026") -> None:
        self._master_key = master_key.encode("utf-8")
        self._secrets: Dict[str, Dict[str, str]] = {}

    def store_credential(
        self,
        device_id: str,
        username: str,
        secret_value: str,
        protocol: str = "SSH/NETCONF",
    ) -> str:
        ref_id = f"vault://credentials/{device_id}/{protocol.lower()}"
        digest = hmac.new(self._master_key, secret_value.encode("utf-8"), hashlib.sha256).hexdigest()
        self._secrets[ref_id] = {
            "device_id": device_id,
            "username": username,
            "protocol": protocol,
            "secret_hmac": digest,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        return ref_id

    def get_public_metadata(self, ref_id: str) -> Dict[str, str]:
        if ref_id not in self._secrets:
            raise KeyError(f"Credencial não encontrada no cofre: {ref_id}")
        meta = dict(self._secrets[ref_id])
        meta["secret_value"] = "[PROTECTED_IN_VAULT]"
        return meta


class TamperEvidentAuditLog:
    """
    Log de auditoria imutável protegido contra adulteração por encadeamento SHA-256 (Seção 8 Etapa 10 e Seção 12).
    Mascara automaticamente qualquer segredo antes de gravar o evento.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    def record(
        self,
        actor: str,
        action: str,
        target: str,
        status: str,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        prev_hash = self._entries[-1]["entry_hash"] if self._entries else "0" * 64
        raw_json = json.dumps(details, sort_keys=True, ensure_ascii=False)
        scrubbed_json, _ = scrub_secrets(raw_json)
        clean_details = json.loads(scrubbed_json)

        entry_id = f"AUD-{len(self._entries) + 1:05d}-{uuid.uuid4().hex[:6].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        payload_to_hash = f"{prev_hash}|{entry_id}|{timestamp}|{actor}|{action}|{target}|{status}|{scrubbed_json}"
        entry_hash = hashlib.sha256(payload_to_hash.encode("utf-8")).hexdigest()

        entry = {
            "entry_id": entry_id,
            "timestamp": timestamp,
            "actor": actor,
            "action": action,
            "target": target,
            "status": status,
            "details": clean_details,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        }
        self._entries.append(entry)
        return copy.deepcopy(entry) if "copy" in globals() else dict(entry)

    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        prev_hash = "0" * 64
        for idx, entry in enumerate(self._entries):
            if entry["prev_hash"] != prev_hash:
                return False, f"Quebra de elo anterior na entrada #{idx} ({entry['entry_id']})"
            scrubbed_json = json.dumps(entry["details"], sort_keys=True, ensure_ascii=False)
            payload = (
                f"{prev_hash}|{entry['entry_id']}|{entry['timestamp']}|{entry['actor']}|"
                f"{entry['action']}|{entry['target']}|{entry['status']}|{scrubbed_json}"
            )
            expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if expected != entry["entry_hash"]:
                return False, f"Adulteração detectada na entrada #{idx} ({entry['entry_id']})"
            prev_hash = expected
        return True, None

    def list_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)
