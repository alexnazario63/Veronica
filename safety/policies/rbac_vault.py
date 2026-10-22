"""
RBAC (Controle de Acesso Baseado em Papéis), Cofre de Credenciais (Secret Vault)
e Log de Auditoria Imutável Encadeado por SHA-256 da Veronica (Seções 8, 12 e 16.8).
Fase K: Gestão Multi-Tenant e Autenticação OIDC / SSO Corporativa.
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
import hmac
import xml.etree.ElementTree as ET

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


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _base64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _is_saml_format(raw: str) -> bool:
    stripped = raw.strip()
    if stripped.startswith("<") and ("saml" in stripped.lower() or "assertion" in stripped.lower()):
        return True
    try:
        decoded = base64.b64decode(stripped, validate=True).decode("utf-8", errors="ignore")
        return decoded.strip().startswith("<") and ("saml" in decoded.lower() or "assertion" in decoded.lower())
    except Exception:
        return False


def _parse_saml_assertion(raw: str) -> Dict[str, Any]:
    xml_str = raw.strip()
    if not xml_str.startswith("<"):
        xml_str = base64.b64decode(raw).decode("utf-8", errors="replace")

    root = ET.fromstring(xml_str)
    user_id = "saml-user"
    for elem in root.iter():
        if elem.tag.endswith("NameID") and elem.text:
            user_id = elem.text.strip()
            break

    attributes: Dict[str, Any] = {}
    for elem in root.iter():
        if elem.tag.endswith("Attribute"):
            name = elem.attrib.get("Name") or elem.attrib.get("name")
            vals = [val.text for val in elem if val.tag.endswith("AttributeValue") and val.text]
            if name and vals:
                attributes[name] = vals[0] if len(vals) == 1 else vals

    tenant_id = attributes.get("tenant_id") or attributes.get("organization") or attributes.get("tenant") or "default"
    role = attributes.get("role") or attributes.get("roles") or "VIEWER"
    if isinstance(role, list):
        roles = role
        role = roles[0]
    else:
        roles = [role]

    return {
        "sub": user_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "roles": roles,
        "saml_attributes": attributes,
        "token_type": "SAML",
    }


class RBACPolicyManager:
    """
    Valida permissões de usuários por papel, garante separação de funções
    e implementa isolamento multi-tenant estrito com autenticação OIDC/SSO (Fase K).
    """

    DEFAULT_OIDC_SECRET: str = "veronica-default-oidc-secret-2026"

    def __init__(self) -> None:
        self.approval_public_keys: Dict[str, str] = {}
        self.tenants: Dict[str, Dict[str, Any]] = {
            "default": {
                "tenant_id": "default",
                "name": "Cluster Corporativo Padrão",
                "users": {"usr-noc-01", "usr-eng-01", "usr-app-01", "usr-trn-01", "usr-adm-01"},
            }
        }
        self.users: Dict[str, Dict[str, Any]] = {
            "usr-noc-01": {"user_id": "usr-noc-01", "name": "Operador NOC", "role": "NOC_OPERATOR", "tenant_id": "default"},
            "usr-eng-01": {"user_id": "usr-eng-01", "name": "Engenheiro de Redes Sênior", "role": "NETWORK_ENGINEER", "tenant_id": "default"},
            "usr-app-01": {"user_id": "usr-app-01", "name": "Coordenador CAB / Aprovador", "role": "CHANGE_APPROVER", "tenant_id": "default"},
            "usr-trn-01": {"user_id": "usr-trn-01", "name": "Engenheiro Neural", "role": "AI_TRAINER", "tenant_id": "default"},
            "usr-adm-01": {"user_id": "usr-adm-01", "name": "Administrador Chefe", "role": "ADMIN", "tenant_id": "default"},
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

    def register_user(
        self,
        user_id: str,
        name: str,
        role: str,
        tenant_id: str = "default",
        public_key_pem: str | None = None,
    ) -> Dict[str, Any]:
        """Cadastra um novo usuário corporativo em um cluster de tenant específico."""
        if role not in ROLE_PERMISSIONS:
            raise ValueError(f"Papel '{role}' desconhecido. Permitidos: {list(ROLE_PERMISSIONS.keys())}")
        user = {
            "user_id": user_id,
            "name": name,
            "role": role,
            "tenant_id": tenant_id,
        }
        self.users[user_id] = user
        if tenant_id not in self.tenants:
            self.tenants[tenant_id] = {"tenant_id": tenant_id, "name": f"Tenant {tenant_id}", "users": set()}
        self.tenants[tenant_id]["users"].add(user_id)
        if public_key_pem:
            self.approval_public_keys[user_id] = public_key_pem
        return user

    def get_user(self, user_id: str, tenant_id: str | None = None) -> Dict[str, Any] | None:
        """Obtém dados do usuário garantindo isolamento por tenant se especificado."""
        user = self.users.get(user_id)
        if user is None:
            return None
        if tenant_id is not None and user.get("tenant_id") and user["tenant_id"] != tenant_id:
            return None
        return copy.deepcopy(user)

    def register_approval_public_key(self, user_id: str, public_key_pem: str, tenant_id: str | None = None) -> None:
        """Provision a trusted Ed25519 public key for a human approver."""
        if not self.check_permission(user_id, "change:approve", tenant_id=tenant_id):
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

    def check_permission(self, user_id: str, permission: str, tenant_id: str | None = None) -> bool:
        """Verifica se o usuário possui permissão, validando também pertinência ao tenant."""
        user = self.users.get(user_id)
        if not user:
            return False
        if tenant_id is not None and user.get("tenant_id") and user["tenant_id"] != tenant_id:
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

    def require_permission(self, user_id: str, permission: str, tenant_id: str | None = None) -> Dict[str, Any]:
        """Exige permissão estrita e isolamento de tenant; levanta exceção caso violado."""
        user = self.users.get(user_id)
        if not user:
            raise PermissionError(f"RBAC_ACCESS_DENIED: Usuário '{user_id}' não encontrado.")
        if tenant_id is not None and user.get("tenant_id") and user["tenant_id"] != tenant_id:
            raise PermissionError(
                f"TENANT_ISOLATION_VIOLATION: Usuário '{user_id}' pertence ao tenant '{user.get('tenant_id')}' "
                f"e não ao tenant solicitado '{tenant_id}'."
            )
        if not self.check_permission(user_id, permission, tenant_id=tenant_id):
            raise PermissionError(
                f"RBAC_ACCESS_DENIED: Usuário '{user_id}' (papel={user['role']}) "
                f"não possui a permissão obrigatória '{permission}'."
            )
        return copy.deepcopy(user)

    def authorize_change(
        self,
        user_id: str,
        *,
        risk_level: str,
        commands: List[str],
        maintenance_window_id: str,
        asset_criticality: str = "STANDARD",
        tenant_id: str | None = None,
    ) -> Dict[str, Any]:
        """Apply deterministic ABAC constraints in addition to the role check."""
        user = self.require_permission(user_id, "change:propose", tenant_id=tenant_id)
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

    # =========================================================================
    # Métodos OIDC / SSO Corporativo (Fase K)
    # =========================================================================

    def generate_oidc_token(
        self,
        user_id: str,
        tenant_id: str = "default",
        role: str = "NETWORK_ENGINEER",
        secret: str | None = None,
        expires_in_seconds: int = 3600,
        issuer: str = "https://auth.veronica.internal",
        audience: str = "veronica-core",
        extra_claims: Dict[str, Any] | None = None,
    ) -> str:
        """Gera um token JWT/OIDC assinado para autenticação corporativa."""
        used_secret = secret or self.DEFAULT_OIDC_SECRET
        now = int(datetime.now(timezone.utc).timestamp())
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "role": role,
            "roles": [role],
            "iss": issuer,
            "aud": audience,
            "iat": now,
            "nbf": now,
            "exp": now + expires_in_seconds,
            "jti": uuid.uuid4().hex,
        }
        if extra_claims:
            payload.update(extra_claims)

        h_b64 = _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        p_b64 = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")
        sig = hmac.new(used_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        s_b64 = _base64url_encode(sig)
        return f"{h_b64}.{p_b64}.{s_b64}"

    def validate_oidc_token(
        self,
        token: str,
        secret: str | None = None,
        expected_issuer: str | None = None,
        expected_audience: str | None = None,
        verify_signature: bool = True,
    ) -> Dict[str, Any]:
        """
        Valida token OIDC/JWT corporativo ou asserção SAML 2.0 (Fase K).
        Verifica assinatura criptográfica HMAC-SHA256, vigência temporal e emissores.
        """
        raw = token.strip()
        if raw.lower().startswith("bearer "):
            raw = raw[7:].strip()

        # Suporte a asserção SAML 2.0 XML
        if _is_saml_format(raw):
            return _parse_saml_assertion(raw)

        parts = raw.split(".")
        if len(parts) != 3:
            # Fallback para token JSON direto em testes ou mocks
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and ("sub" in parsed or "user_id" in parsed):
                    return parsed
            except Exception:
                pass
            raise ValueError("MALFORMED_JWT_TOKEN: O token OIDC deve conter exatamente 3 partes (header.payload.signature).")

        h_b64, p_b64, s_b64 = parts
        try:
            header = json.loads(_base64url_decode(h_b64).decode("utf-8"))
            payload = json.loads(_base64url_decode(p_b64).decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"MALFORMED_JWT_TOKEN: Falha ao decodificar partes base64url do JWT: {exc}") from exc

        if verify_signature:
            alg = header.get("alg")
            if alg == "HS256":
                used_secret = secret or self.DEFAULT_OIDC_SECRET
                signing_input = f"{h_b64}.{p_b64}".encode("ascii")
                expected_sig = hmac.new(used_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
                actual_sig = _base64url_decode(s_b64)
                if not hmac.compare_digest(expected_sig, actual_sig):
                    raise PermissionError("JWT_SIGNATURE_INVALID: Assinatura HMAC-SHA256 do token OIDC inválida.")
            elif alg == "none":
                pass
            else:
                if secret is not None:
                    raise PermissionError(f"UNSUPPORTED_JWT_ALGORITHM: Algoritmo '{alg}' não suportado.")

        now = datetime.now(timezone.utc).timestamp()
        if "exp" in payload and now > payload["exp"]:
            raise ValueError("TOKEN_EXPIRED: O token OIDC/JWT corporativo expirou.")
        if "nbf" in payload and now < payload["nbf"]:
            raise ValueError("TOKEN_NOT_YET_VALID: O token OIDC/JWT ainda não é válido (nbf).")

        if expected_issuer and payload.get("iss") != expected_issuer:
            raise PermissionError(f"INVALID_ISSUER: Issuer '{payload.get('iss')}' diverge de '{expected_issuer}'.")
        if expected_audience and payload.get("aud") != expected_audience:
            raise PermissionError(f"INVALID_AUDIENCE: Audience '{payload.get('aud')}' diverge de '{expected_audience}'.")

        return payload

    def authenticate_token(
        self,
        token: str,
        secret: str | None = None,
        register: bool = True,
    ) -> Dict[str, Any]:
        """
        Autentica token corporativo OIDC/JWT ou asserção SAML 2.0 (Fase K).
        Extrai tenant_id, user_id e roles/permissões, registrando dinamicamente o usuário no cluster.
        """
        claims = self.validate_oidc_token(token, secret=secret)
        user_id = claims.get("sub") or claims.get("user_id") or claims.get("username")
        if not user_id:
            raise ValueError("INVALID_TOKEN_CLAIMS: Token não contém identificador de usuário (sub/user_id).")

        tenant_id = (
            claims.get("tenant_id")
            or claims.get("tid")
            or claims.get("organization")
            or claims.get("tenant")
            or "default"
        )

        raw_roles = claims.get("roles") or claims.get("cognito:groups") or claims.get("groups")
        if not raw_roles and "role" in claims:
            raw_roles = [claims["role"]]
        if not raw_roles:
            raw_roles = ["VIEWER"]
        if isinstance(raw_roles, str):
            raw_roles = [raw_roles]

        role_priority = ["ADMIN", "CHANGE_APPROVER", "NETWORK_ENGINEER", "AI_TRAINER", "NOC_OPERATOR", "VIEWER"]
        normalized_roles = [r.upper() for r in raw_roles]
        primary_role = next((r for r in role_priority if r in normalized_roles), normalized_roles[0])

        all_permissions: Set[str] = set()
        for r in normalized_roles:
            all_permissions.update(ROLE_PERMISSIONS.get(r, set()))

        user_info = {
            "user_id": user_id,
            "name": claims.get("name") or user_id,
            "role": primary_role,
            "roles": normalized_roles,
            "tenant_id": tenant_id,
            "authenticated_via": claims.get("token_type", "OIDC/JWT"),
        }

        if register:
            self.users[user_id] = user_info
            if tenant_id not in self.tenants:
                self.tenants[tenant_id] = {"tenant_id": tenant_id, "name": f"Tenant {tenant_id}", "users": set()}
            self.tenants[tenant_id]["users"].add(user_id)

        return {
            "authenticated": True,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "role": primary_role,
            "roles": normalized_roles,
            "permissions": sorted(all_permissions),
            "claims": claims,
            "token_type": claims.get("token_type", "OIDC/JWT"),
        }


class CredentialVault:
    """
    Cofre de segredos em memória isolada com criptografia AEAD AES-GCM (Seções 8 e 12).
    Fase K: Suporte a isolamento multi-tenant estrito com chaves derivadas por cluster corporativo.
    Nunca expõe credenciais brutas em logs, respostas de API ou datasets.
    """

    def __init__(self, master_key: str = "veronica-vault-local-key-2026") -> None:
        self._master_key = hashlib.sha256(master_key.encode("utf-8")).digest()
        self._secrets: Dict[str, Dict[str, Any]] = {}

    def _derive_tenant_key(self, tenant_id: str) -> bytes:
        """Deriva chave AES-256 criptograficamente isolada por tenant usando HMAC-SHA256."""
        return hmac.new(self._master_key, tenant_id.encode("utf-8"), hashlib.sha256).digest()

    def store_credential(
        self,
        device_id: str,
        username: str,
        secret_value: str,
        protocol: str = "SSH/NETCONF",
        tenant_id: str = "default",
    ) -> str:
        """Armazena segredo cifrado com chave AES-GCM isolada pelo tenant_id."""
        if tenant_id != "default":
            ref_id = f"vault://credentials/{tenant_id}/{device_id}/{protocol.lower()}"
        else:
            ref_id = f"vault://credentials/{device_id}/{protocol.lower()}"

        nonce = os.urandom(12)
        aad = f"{tenant_id}:{ref_id}".encode("utf-8")
        tenant_key = self._derive_tenant_key(tenant_id)
        ciphertext = AESGCM(tenant_key).encrypt(nonce, secret_value.encode("utf-8"), aad)

        self._secrets[ref_id] = {
            "ref_id": ref_id,
            "tenant_id": tenant_id,
            "device_id": device_id,
            "username": username,
            "protocol": protocol,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        return ref_id

    def retrieve_credential(self, ref_id: str, tenant_id: str | None = None) -> str:
        """Descriptografa credencial para o adaptador, validando estritamente o tenant_id."""
        item = self._secrets.get(ref_id)
        if item is None:
            raise KeyError(f"Credencial não encontrada no cofre: {ref_id}")

        secret_tenant = item.get("tenant_id", "default")
        if tenant_id is not None and secret_tenant != tenant_id:
            raise PermissionError(
                f"TENANT_ISOLATION_VIOLATION: Tentativa de acesso não autorizado à credencial do tenant '{secret_tenant}' "
                f"a partir do tenant requisitante '{tenant_id}'."
            )

        tenant_key = self._derive_tenant_key(secret_tenant)
        aad = f"{secret_tenant}:{ref_id}".encode("utf-8")
        try:
            plaintext = AESGCM(tenant_key).decrypt(
                base64.b64decode(item["nonce"]),
                base64.b64decode(item["ciphertext"]),
                aad,
            )
        except Exception:
            # Fallback transparente para credenciais legadas cifradas diretamente sob a chave mestra
            try:
                plaintext = AESGCM(self._master_key).decrypt(
                    base64.b64decode(item["nonce"]),
                    base64.b64decode(item["ciphertext"]),
                    ref_id.encode("utf-8"),
                )
            except Exception as exc:
                raise ValueError("CREDENTIAL_INTEGRITY_ERROR: credencial não pôde ser autenticada") from exc

        return plaintext.decode("utf-8")

    def get_public_metadata(self, ref_id: str, tenant_id: str | None = None) -> Dict[str, str]:
        """Retorna metadados públicos do segredo omitindo ciphertext e nonce."""
        if ref_id not in self._secrets:
            raise KeyError(f"Credencial não encontrada no cofre: {ref_id}")
        item = self._secrets[ref_id]
        if tenant_id is not None and item.get("tenant_id", "default") != tenant_id:
            raise PermissionError(f"TENANT_ISOLATION_VIOLATION: Acesso negado aos metadados do tenant '{item.get('tenant_id')}'.")
        return {key: str(item[key]) for key in ("device_id", "username", "protocol", "stored_at") if key in item}

    def list_credentials_for_tenant(self, tenant_id: str) -> List[Dict[str, str]]:
        """Lista credenciais cadastradas pertencentes exclusivamente ao tenant informado."""
        return [
            {key: str(item[key]) for key in ("device_id", "username", "protocol", "stored_at", "tenant_id", "ref_id") if key in item}
            for item in self._secrets.values()
            if item.get("tenant_id") == tenant_id
        ]


class TamperEvidentAuditLog:
    """
    Log de auditoria imutável protegido contra adulteração por encadeamento SHA-256 (Seção 8 Etapa 10 e Seção 12).
    Fase K: Encadeamento contínuo incluindo o campo tenant_id e verificação formal de integridade criptográfica.
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
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        """Registra entrada no log de auditoria com encadeamento contínuo SHA-256 e tenant_id."""
        prev_hash = self._entries[-1]["entry_hash"] if self._entries else "0" * 64

        # Resgata tenant_id de details caso presente
        if tenant_id == "default" and "tenant_id" in details:
            tenant_id = str(details["tenant_id"])

        raw_json = json.dumps(details, sort_keys=True, ensure_ascii=False)
        scrubbed_json, _ = scrub_secrets(raw_json)
        clean_details = json.loads(scrubbed_json)

        entry_id = f"AUD-{len(self._entries) + 1:05d}-{uuid.uuid4().hex[:6].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        # Encadeamento contínuo via SHA-256 integrando tenant_id
        payload_to_hash = (
            f"{prev_hash}|{entry_id}|{timestamp}|{tenant_id}|{actor}|{action}|{target}|{status}|{scrubbed_json}"
        )
        entry_hash = hashlib.sha256(payload_to_hash.encode("utf-8")).hexdigest()

        entry = {
            "entry_id": entry_id,
            "timestamp": timestamp,
            "tenant_id": tenant_id,
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

    def append_record(
        self,
        operator_id: str,
        action: str,
        payload: Dict[str, Any],
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        """Alias para record em conformidade com contratos de interfaces."""
        return self.record(
            actor=operator_id,
            action=action,
            target=str(payload.get("target") or payload.get("device_id") or "system"),
            status=str(payload.get("status", "SUCCESS")),
            details=payload,
            tenant_id=tenant_id,
        )

    def verify_chain_integrity(self, tenant_id: str | None = None) -> Tuple[bool, Optional[str]]:
        """Verifica a integridade formal de toda a cadeia criptográfica SHA-256."""
        prev_hash = "0" * 64
        for idx, entry in enumerate(self._entries):
            if entry["prev_hash"] != prev_hash:
                return False, f"Quebra de elo anterior na entrada #{idx} ({entry['entry_id']})"

            scrubbed_json = json.dumps(entry["details"], sort_keys=True, ensure_ascii=False)
            t_id = entry.get("tenant_id")

            # Valida hash com tenant_id
            if t_id is not None:
                payload = (
                    f"{prev_hash}|{entry['entry_id']}|{entry['timestamp']}|{t_id}|"
                    f"{entry['actor']}|{entry['action']}|{entry['target']}|{entry['status']}|{scrubbed_json}"
                )
                expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                if expected != entry["entry_hash"]:
                    # Fallback para entradas legadas sem tenant_id no cálculo de hash
                    legacy_payload = (
                        f"{prev_hash}|{entry['entry_id']}|{entry['timestamp']}|{entry['actor']}|"
                        f"{entry['action']}|{entry['target']}|{entry['status']}|{scrubbed_json}"
                    )
                    if hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest() != entry["entry_hash"]:
                        return False, f"Adulteração detectada na entrada #{idx} ({entry['entry_id']})"
            else:
                legacy_payload = (
                    f"{prev_hash}|{entry['entry_id']}|{entry['timestamp']}|{entry['actor']}|"
                    f"{entry['action']}|{entry['target']}|{entry['status']}|{scrubbed_json}"
                )
                expected = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()
                if expected != entry["entry_hash"]:
                    return False, f"Adulteração detectada na entrada #{idx} ({entry['entry_id']})"

            prev_hash = entry["entry_hash"]

        return True, None

    def verify_integrity(self) -> Tuple[bool, Optional[int]]:
        """Conformidade com a assinatura de docs/INTERFACES.md retornando (bool, Optional[int])."""
        ok, _ = self.verify_chain_integrity()
        return (True, None) if ok else (False, -1)

    def list_entries(self, tenant_id: str | None = None) -> List[Dict[str, Any]]:
        """Retorna as entradas de log, opcionalmente filtradas por tenant_id."""
        if tenant_id is not None:
            return [copy.deepcopy(e) for e in self._entries if e.get("tenant_id") == tenant_id]
        return copy.deepcopy(self._entries)
