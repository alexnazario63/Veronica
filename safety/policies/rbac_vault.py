"""
RBAC (Controle de Acesso Baseado em Papéis), Cofre de Credenciais (Secret Vault)
e Log de Auditoria Imutável Encadeado por SHA-256 da Veronica (Seções 8, 12 e 16.8).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import base64
import copy
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

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
        self.approval_public_keys: Dict[str, str] = {}
        self.users: Dict[str, Dict[str, Any]] = {
            "usr-noc-01": {"user_id": "usr-noc-01", "name": "Operador NOC", "role": "NOC_OPERATOR"},
            "usr-eng-01": {"user_id": "usr-eng-01", "name": "Engenheiro de Redes Sênior", "role": "NETWORK_ENGINEER"},
            "usr-app-01": {"user_id": "usr-app-01", "name": "Coordenador CAB / Aprovador", "role": "CHANGE_APPROVER"},
            "usr-trn-01": {"user_id": "usr-trn-01", "name": "Engenheiro Neural", "role": "AI_TRAINER"},
            "usr-adm-01": {"user_id": "usr-adm-01", "name": "Administrador Chefe", "role": "ADMIN"},
        }
        self._casbin_enforcer = self._build_casbin_enforcer()

    @staticmethod
    def _build_casbin_enforcer() -> Any | None:
        """Build an in-memory Casbin RBAC/ABAC policy; absence uses local rules."""
        try:
            import casbin
        except ImportError:
            return None
        try:
            return RBACPolicyManager._configure_casbin(casbin)
        except Exception:
            # A version/API mismatch falls back to the deterministic local policy.
            return None

    @staticmethod
    def _configure_casbin(casbin: Any) -> Any:
        model = casbin.Model()
        model.load_model_from_text(
            """
            [request_definition]
            r = sub, risk, criticality, command_type, in_window, act
            [policy_definition]
            p = role, risk, criticality, command_type, in_window, act
            [policy_effect]
            e = some(where (p.eft == allow))
            [matchers]
            m = r.sub == p.role && (p.risk == "*" || p.risk == r.risk) && (p.criticality == "*" || p.criticality == r.criticality) && (p.command_type == "*" || p.command_type == r.command_type) && r.in_window == p.in_window && r.act == p.act
            """
        )
        enforcer = casbin.Enforcer(model)
        for role, permissions in ROLE_PERMISSIONS.items():
            for permission in permissions - {"change:propose"}:
                enforcer.add_policy(role, "*", "*", "*", "true", permission)

        # A proposal policy explicitly crosses risk, asset criticality, command
        # class, role and a confirmed maintenance-window identifier.
        for role in ("NETWORK_ENGINEER", "ADMIN"):
            risks = ("LOW", "MEDIUM", "HIGH") if role == "NETWORK_ENGINEER" else (
                "LOW", "MEDIUM", "HIGH", "CRITICAL"
            )
            criticalities = ("STANDARD", "HIGH") if role == "NETWORK_ENGINEER" else (
                "STANDARD", "HIGH", "CRITICAL"
            )
            for risk in risks:
                for criticality in criticalities:
                    for command_type in ("read", "write", "routing"):
                        enforcer.add_policy(
                            role, risk, criticality, command_type, "true", "change:propose"
                        )
        return enforcer

    def register_approval_public_key(self, user_id: str, public_key_pem: str) -> None:
        """Provision a trusted Ed25519 public key for a human approver."""
        if not self.check_permission(user_id, "change:approve"):
            raise PermissionError("Only an authorized change approver may be provisioned.")
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
        except ValueError as exc:
            raise ValueError("Invalid approval public key") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("Approval key must use Ed25519")
        self.approval_public_keys[user_id] = public_key_pem

    def check_permission(self, user_id: str, permission: str) -> bool:
        user = self.users.get(user_id)
        if not user:
            return False
        role = user["role"]
        if self._casbin_enforcer is not None:
            try:
                return bool(
                    self._casbin_enforcer.enforce(
                        role, "LOW", "STANDARD", "read", "true", permission
                    )
                )
            except Exception:
                return False
        return permission in ROLE_PERMISSIONS.get(role, set())

    def require_permission(self, user_id: str, permission: str) -> Dict[str, Any]:
        if not self.check_permission(user_id, permission):
            user = self.users.get(user_id, {"role": "UNKNOWN"})
            raise PermissionError(
                f"RBAC_ACCESS_DENIED: Usuário '{user_id}' (papel={user['role']}) "
                f"não possui a permissão obrigatória '{permission}'."
            )
        return self.users[user_id]

    def authorize_change(
        self,
        user_id: str,
        *,
        risk_level: str,
        commands: List[str],
        maintenance_window_id: str,
        asset_criticality: str = "STANDARD",
    ) -> Dict[str, Any]:
        """Apply deterministic ABAC constraints in addition to the role check."""
        user = self.require_permission(user_id, "change:propose")
        if not maintenance_window_id.strip():
            raise PermissionError("ABAC_DENIED: a maintenance window is required")
        normalized = [command.strip().lower() for command in commands]
        if not normalized or any(not command for command in normalized):
            raise PermissionError("ABAC_DENIED: empty change command set")
        if any(any(word in command for word in ("shutdown", "erase", "zeroize", "reset", "delete ")) for command in normalized):
            raise PermissionError("ABAC_DENIED: destructive command class is prohibited")
        command_type = (
            "routing" if any(any(token in command for token in ("bgp", "ospf", "isis", "route", "mtu")) for command in normalized)
            else "write"
        )
        criticality = asset_criticality.strip().upper()
        risk = risk_level.strip().upper()
        if criticality not in {"STANDARD", "HIGH", "CRITICAL"}:
            raise PermissionError("ABAC_DENIED: unknown asset criticality")
        if risk not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            raise PermissionError("ABAC_DENIED: unknown change risk")
        if self._casbin_enforcer is not None:
            try:
                allowed = self._casbin_enforcer.enforce(
                    user["role"], risk, criticality, command_type, "true", "change:propose"
                )
            except Exception:
                allowed = False
            if not allowed:
                raise PermissionError("ABAC_DENIED: Casbin policy rejected this change context")
            return user

        role = user["role"]
        if (risk == "CRITICAL" or criticality == "CRITICAL") and role != "ADMIN":
            raise PermissionError("ABAC_DENIED: critical changes/assets require ADMIN role")
        if risk == "HIGH" and role not in {"NETWORK_ENGINEER", "ADMIN"}:
            raise PermissionError("ABAC_DENIED: HIGH changes require engineering role")
        return user


class CredentialVault:
    """
    Cofre de segredos em memória isolada (Seção 8 e 12).
    Nunca expõe credenciais brutas em logs, respostas de API ou datasets.
    """

    def __init__(self, master_key: str = "veronica-vault-local-key-2026") -> None:
        # Hash arbitrary configured key material to the AES-256 key size.
        self._master_key = hashlib.sha256(master_key.encode("utf-8")).digest()
        self._secrets: Dict[str, Dict[str, str]] = {}

    def store_credential(
        self,
        device_id: str,
        username: str,
        secret_value: str,
        protocol: str = "SSH/NETCONF",
    ) -> str:
        ref_id = f"vault://credentials/{device_id}/{protocol.lower()}"
        nonce = os.urandom(12)
        aad = ref_id.encode("utf-8")
        ciphertext = AESGCM(self._master_key).encrypt(nonce, secret_value.encode("utf-8"), aad)
        self._secrets[ref_id] = {
            "device_id": device_id,
            "username": username,
            "protocol": protocol,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        return ref_id

    def retrieve_credential(self, ref_id: str) -> str:
        """Decrypt a secret for an authenticated device adapter."""
        item = self._secrets.get(ref_id)
        if item is None:
            raise KeyError(f"Credencial não encontrada no cofre: {ref_id}")
        try:
            plaintext = AESGCM(self._master_key).decrypt(
                base64.b64decode(item["nonce"]),
                base64.b64decode(item["ciphertext"]),
                ref_id.encode("utf-8"),
            )
        except Exception as exc:
            raise ValueError("CREDENTIAL_INTEGRITY_ERROR: credencial não pôde ser autenticada") from exc
        return plaintext.decode("utf-8")

    def get_public_metadata(self, ref_id: str) -> Dict[str, str]:
        if ref_id not in self._secrets:
            raise KeyError(f"Credencial não encontrada no cofre: {ref_id}")
        item = self._secrets[ref_id]
        return {key: item[key] for key in ("device_id", "username", "protocol", "stored_at")}


class TamperEvidentAuditLog:
    """
    Log de auditoria imutável protegido contra adulteração por encadeamento SHA-256 (Seção 8 Etapa 10 e Seção 12).
    Mascara automaticamente qualquer segredo antes de gravar o evento.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []

    @property
    def head_hash(self) -> str:
        return self._entries[-1]["entry_hash"] if self._entries else "0" * 64

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
        return copy.deepcopy(entry)

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
        return copy.deepcopy(self._entries)
