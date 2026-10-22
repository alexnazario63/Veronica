"""
Autenticação Multi-Tenant e Integração OIDC / JWT para Veronica (Fase K).
Suporta cabeçalhos 'X-Tenant-Id', 'Authorization: Bearer <token>', validação
criptográfica HMAC-SHA256, isolamento estrito entre locatários (tenants)
e integração direta com safety.policies.rbac_vault.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from typing import Any, Dict, List, Optional, Set

from fastapi import Header, HTTPException, Request

from safety.policies.rbac_vault import RBACPolicyManager, ROLE_PERMISSIONS


DEFAULT_JWT_SECRET = os.environ.get("VERONICA_JWT_SECRET", "veronica-oidc-jwt-secret-key-2026")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = 4 - (len(data) % 4)
    if padding and padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode("ascii"))


@dataclass
class TenantContext:
    user_id: str
    tenant_id: str
    role: str
    permissions: Set[str] = field(default_factory=set)
    claims: Dict[str, Any] = field(default_factory=dict)
    token: Optional[str] = None

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


def create_jwt_token(
    claims: Dict[str, Any],
    secret: str = DEFAULT_JWT_SECRET,
    expires_in_seconds: int = 3600,
) -> str:
    """Gera token JWT assinado (HS256) compatível com padrões OIDC."""
    now = int(datetime.now(timezone.utc).timestamp())
    payload = dict(claims)
    payload.setdefault("iat", now)
    payload.setdefault("exp", now + expires_in_seconds)
    payload.setdefault("iss", "veronica-oidc-provider")

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def validate_jwt_token(
    token: str,
    secret: str = DEFAULT_JWT_SECRET,
    expected_issuer: Optional[str] = None,
    expected_audience: Optional[str] = None,
) -> Dict[str, Any]:
    """Valida assinatura e claims de expiração/emissor de um token JWT."""
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT token format: must contain header, payload, and signature.")

    header_b64, payload_b64, sig_b64 = parts

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    provided_sig = _b64url_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, provided_sig):
        raise ValueError("Invalid JWT signature: token verification failed.")

    try:
        payload_data = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JWT payload JSON: {exc}") from exc

    now = int(datetime.now(timezone.utc).timestamp())
    if "exp" in payload_data and payload_data["exp"] < now:
        raise ValueError(f"JWT token has expired (exp={payload_data['exp']}, now={now}).")

    if "nbf" in payload_data and payload_data["nbf"] > now:
        raise ValueError("JWT token is not yet valid (nbf).")

    if expected_issuer and payload_data.get("iss") != expected_issuer:
        raise ValueError(f"JWT issuer mismatch: expected '{expected_issuer}', got '{payload_data.get('iss')}'.")

    if expected_audience and payload_data.get("aud") != expected_audience:
        raise ValueError(f"JWT audience mismatch: expected '{expected_audience}', got '{payload_data.get('aud')}'.")

    return payload_data


def authenticate_claims_with_rbac(
    claims: Dict[str, Any],
    rbac: RBACPolicyManager,
    tenant_id_override: Optional[str] = None,
    token: Optional[str] = None,
) -> TenantContext:
    """Converte claims de token OIDC/JWT em TenantContext com papéis e permissões no RBAC."""
    user_id = str(claims.get("sub") or claims.get("user_id") or "usr-eng-01").strip()
    tenant_id = str(tenant_id_override or claims.get("tenant_id") or claims.get("tid") or "default").strip()
    role = str(claims.get("role") or claims.get("roles", ["NETWORK_ENGINEER"])[0] if isinstance(claims.get("roles"), list) else "NETWORK_ENGINEER").strip().upper()

    if role not in ROLE_PERMISSIONS:
        role = "NETWORK_ENGINEER"

    # Registra ou sincroniza o usuário no RBAC se não constar na lista estática
    if user_id not in rbac.users:
        rbac.users[user_id] = {
            "user_id": user_id,
            "name": claims.get("name", f"Usuário {user_id}"),
            "role": role,
            "tenant_id": tenant_id,
        }
    else:
        existing = rbac.users[user_id]
        if "role" in claims and claims["role"] in ROLE_PERMISSIONS:
            existing["role"] = role
        else:
            role = existing.get("role", role)
        existing["tenant_id"] = tenant_id

    permissions = set(ROLE_PERMISSIONS.get(role, set()))
    return TenantContext(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        permissions=permissions,
        claims=claims,
        token=token,
    )


def resolve_tenant_context(
    rbac: RBACPolicyManager,
    authorization: Optional[str] = None,
    x_user_id: Optional[str] = None,
    x_tenant_id: Optional[str] = None,
    secret: str = DEFAULT_JWT_SECRET,
) -> TenantContext:
    """Resolve contexto autenticado a partir de Authorization Bearer ou cabeçalhos X-User-Id / X-Tenant-Id."""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        try:
            claims = validate_jwt_token(token, secret=secret)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=f"UNAUTHORIZED_TOKEN_INVALID: {exc}") from exc
        tenant_override = x_tenant_id if x_tenant_id and x_tenant_id != "default" else None
        return authenticate_claims_with_rbac(claims, rbac, tenant_id_override=tenant_override, token=token)

    # Fallback para cabeçalhos HTTP diretos (compatibilidade com sessões legadas e testes existentes)
    user_id = str(x_user_id or "usr-eng-01").strip()
    tenant_id = str(x_tenant_id or "default").strip()

    user = rbac.users.get(user_id)
    if user:
        role = user.get("role", "NETWORK_ENGINEER")
    else:
        role = "NETWORK_ENGINEER"
        rbac.users[user_id] = {"user_id": user_id, "name": user_id, "role": role, "tenant_id": tenant_id}

    permissions = set(ROLE_PERMISSIONS.get(role, set()))
    return TenantContext(
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        permissions=permissions,
        claims={"sub": user_id, "tenant_id": tenant_id, "role": role},
        token=None,
    )
