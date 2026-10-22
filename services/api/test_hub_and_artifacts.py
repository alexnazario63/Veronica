"""
Testes Unitários e de Integração dos novos contratos do Hub, API e Workspace:
1. Multi-Tenant Auth (Fase K): X-Tenant-Id, OIDC/JWT Token Validation e RBAC
2. Contextual Mentions (@device, @artifact, @vendor) e Seleção Parcial de Linhas com ArtifactStore.propose_patch
3. ArtifactStore: Contratos do INTERFACES.md, Tipos Ricos, diff_versions e vínculo com DeterministicChangeController
4. Streaming WebSocket de Chat (/api/conversations/{id}/ws)
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from services.api.app import create_app
from services.api.auth import create_jwt_token, validate_jwt_token
from services.api.workspace.store import ArtifactStore, SUPPORTED_ARTIFACT_KINDS


@pytest.fixture
def api_client(tmp_path: Path):
    import os
    prior_lock = os.environ.get("VERONICA_SAFETY_LOCK_DB")
    os.environ["VERONICA_SAFETY_LOCK_DB"] = str(tmp_path / "safety-locks.sqlite3")
    try:
        app = create_app(tmp_path)
        with TestClient(app) as client:
            yield client
    finally:
        if prior_lock is None:
            os.environ.pop("VERONICA_SAFETY_LOCK_DB", None)
        else:
            os.environ["VERONICA_SAFETY_LOCK_DB"] = prior_lock


def test_artifact_store_interfaces_and_rich_types(tmp_path: Path) -> None:
    """Valida todos os contratos de INTERFACES.md Seção 6 e tipos ricos de artefatos."""
    store = ArtifactStore(tmp_path / "artifacts.sqlite3")

    # 1. Suporte a tipos ricos de artefatos
    rich_kinds = [
        "config_diff",
        "change_plan_10_steps",
        "diagnostic_report_11_steps",
        "topology_blast_radius",
        "rca_document",
    ]
    for kind in rich_kinds:
        assert kind in SUPPORTED_ARTIFACT_KINDS

    # 2. save_artifact incremental
    v1 = store.save_artifact(
        artifact_id="art-bgp-01",
        title="Política BGP Route-Map Core",
        content="route-policy BGP_EDGE permit node 10\n apply cost 100\n",
        kind="config_diff",
        author="usr-eng-01",
    )
    assert v1 == 1

    art_v1 = store.get_artifact("art-bgp-01")
    assert art_v1 is not None
    assert art_v1["version"] == 1
    assert art_v1["title"] == "Política BGP Route-Map Core"
    assert art_v1["kind"] == "config_diff"

    # 3. propose_patch e versionamento
    patch_diff = (
        "--- art-bgp-01_v1\n"
        "+++ art-bgp-01_v2\n"
        "@@ -1,2 +1,2 @@\n"
        " route-policy BGP_EDGE permit node 10\n"
        "- apply cost 100\n"
        "+ apply cost 20\n"
    )
    patch_res = store.propose_patch("art-bgp-01", version=1, patch_diff=patch_diff, author="usr-eng-01")
    assert patch_res["status"] == "PROPOSED"
    assert patch_res["new_version"] == 2

    art_v2 = store.get_artifact("art-bgp-01", version=2)
    assert art_v2 is not None
    assert "apply cost 20" in art_v2["content"]

    # 4. diff_versions
    diff_res = store.diff_versions("art-bgp-01", 1, 2)
    assert diff_res["artifact_id"] == "art-bgp-01"
    assert diff_res["v1"] == 1
    assert diff_res["v2"] == 2
    assert "- apply cost 100" in diff_res["diff"]
    assert "+ apply cost 20" in diff_res["diff"]
    assert diff_res["lines_added"] >= 1
    assert diff_res["lines_removed"] >= 1

    # 5. get_version_history
    history = store.get_version_history("art-bgp-01")
    assert len(history) == 2
    assert history[0]["version"] == 1
    assert history[1]["version"] == 2

    # 6. list_artifacts
    arts = store.list_artifacts()
    assert len(arts) >= 1
    assert any(a["id"] == "art-bgp-01" for a in arts)

    # 7. Vínculo direto com ID do plano do DeterministicChangeController
    store.link_change_plan("art-bgp-01", "chg-20261022-core-01")
    updated = store.get_artifact("art-bgp-01")
    assert updated["plan_id"] == "chg-20261022-core-01"


def test_multi_tenant_oidc_jwt_auth_and_isolation(api_client: TestClient) -> None:
    """Valida Fase K: Emissão e validação de JWT, cabeçalho X-Tenant-Id e isolamento estrito."""
    # 1. Emissão de Token para Tenant Alpha
    tok_alpha_resp = api_client.post(
        "/api/auth/token",
        json={
            "user_id": "usr-corp-01",
            "tenant_id": "tenant-alpha",
            "role": "NETWORK_ENGINEER",
            "expires_in_seconds": 1800,
        },
    )
    assert tok_alpha_resp.status_code == 200
    token_alpha = tok_alpha_resp.json()["access_token"]
    assert len(token_alpha.split(".")) == 3

    # 2. Verificação de endpoint /api/auth/me usando Bearer Token
    me_resp = api_client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token_alpha}"},
    )
    assert me_resp.status_code == 200
    me = me_resp.json()
    assert me["user_id"] == "usr-corp-01"
    assert me["tenant_id"] == "tenant-alpha"
    assert me["role"] == "NETWORK_ENGINEER"
    assert "change:propose" in me["permissions"]

    # 3. Tentativa com token forjado / inválido deve retornar 401
    bad_resp = api_client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer token.invalido.forjado"},
    )
    assert bad_resp.status_code == 401
    assert "UNAUTHORIZED_TOKEN_INVALID" in bad_resp.json()["detail"]

    # 4. Isolamento estrito entre locatários (Tenant Alpha vs Tenant Beta)
    conv_alpha = api_client.post(
        "/api/conversations",
        json={"title": "Investigação Privada Tenant Alpha"},
        headers={"Authorization": f"Bearer {token_alpha}"},
    ).json()
    conv_id = conv_alpha["conversation_id"]

    # Cria token para Tenant Beta
    tok_beta = api_client.post(
        "/api/auth/token",
        json={
            "user_id": "usr-corp-02",
            "tenant_id": "tenant-beta",
            "role": "NETWORK_ENGINEER",
        },
    ).json()["access_token"]

    # Usuário do Tenant Beta tentando acessar conversa do Tenant Alpha recebe 403
    forbidden_resp = api_client.get(
        f"/api/conversations/{conv_id}",
        headers={"Authorization": f"Bearer {tok_beta}"},
    )
    assert forbidden_resp.status_code == 403
    assert "SESSION_ISOLATION_GUARD" in forbidden_resp.json()["detail"]


def test_contextual_mentions_and_selected_lines_patch_flow(api_client: TestClient) -> None:
    """
    Valida menções contextuais (@device, @artifact, @vendor) e comando
    de ajuste parcial em linhas selecionadas aplicando patch via ArtifactStore.
    """
    # 1. Cria artefato base no Workspace
    create_art_resp = api_client.post(
        "/api/workspace/artifacts",
        json={
            "artifact_id": "art-route-map-01",
            "title": "Route-Map de Filtro Core",
            "content": (
                "ip prefix-list PL_CORE permit 10.0.0.0/8\n"
                "route-policy RP_CORE permit node 10\n"
                " apply cost 100\n"
                " apply local-preference 100\n"
            ),
            "kind": "config_diff",
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert create_art_resp.status_code == 200
    assert create_art_resp.json()["version"] == 1

    # 2. Cria conversa e envia comando com menções e linhas selecionadas
    conv = api_client.post(
        "/api/conversations",
        json={"title": "Ajuste Fino de Políticas de Roteamento"},
        headers={"X-User-Id": "usr-eng-01"},
    ).json()
    conv_id = conv["conversation_id"]

    msg_resp = api_client.post(
        f"/api/conversations/{conv_id}/messages",
        json={
            "content": "Ajusta apenas a política de route-map destas linhas selecionadas no @art-route-map-01 para o @NE8000-CORE-01 da @huawei",
            "modality": "TEXT",
            "selected_artifact_id": "art-route-map-01",
            "selected_lines_range": {"start": 3, "end": 4},
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert msg_resp.status_code == 200
    payload = msg_resp.json()["structured_payload"]

    # Verifica captura de menções
    mentions = payload["context_mentions"]
    assert "NE8000-CORE-01" in mentions["devices"]
    assert "art-route-map-01" in mentions["artifacts"]
    assert "huawei" in mentions["vendors"]

    # Verifica que o patch foi proposto automaticamente no ArtifactStore
    assert payload["proposed_patch"] is not None
    assert payload["proposed_patch"]["artifact_id"] == "art-route-map-01"
    assert payload["proposed_patch"]["new_version"] == 2
    assert "```diff" in msg_resp.json()["content"]

    # 3. Confirma que a nova versão v2 no ArtifactStore contém o patch
    v2_art_resp = api_client.get(
        "/api/workspace/artifacts/art-route-map-01",
        params={"version": 2},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert v2_art_resp.status_code == 200
    art_v2 = v2_art_resp.json()["artifact"]
    assert art_v2["version"] == 2

    # 4. Compara versões v1 e v2 via endpoint de diff
    diff_endpoint_resp = api_client.get(
        "/api/workspace/artifacts/art-route-map-01/diff",
        params={"v1": 1, "v2": 2},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert diff_endpoint_resp.status_code == 200
    diff_data = diff_endpoint_resp.json()
    assert diff_data["lines_added"] >= 1


def test_change_plan_links_directly_to_artifact_store(api_client: TestClient) -> None:
    """Garante que a criação de um plano em 10 etapas gere e vincule um artefato no ArtifactStore."""
    plan_resp = api_client.post(
        "/api/network/changes",
        json={
            "device_id": "NE8000-CORE-01",
            "title": "Ajuste BGP Neighbor Timer",
            "proposed_commands": [
                "system-view",
                "bgp 65001",
                "peer 10.0.0.2 timer keepalive 30 hold 90",
                "commit trial 120",
            ],
            "origin_channel": "TEXT_UI",
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert plan_resp.status_code == 200
    change_id = plan_resp.json()["change_id"]

    # Verifica se o artefato foi criado e vinculado com kind change_plan_10_steps
    art_resp = api_client.get(f"/api/workspace/artifacts/{change_id}")
    assert art_resp.status_code == 200
    art = art_resp.json()["artifact"]
    assert art["id"] == change_id
    assert art["kind"] == "change_plan_10_steps"
    assert art["plan_id"] == change_id


def test_websocket_chat_streaming(api_client: TestClient) -> None:
    """Valida o endpoint de WebSocket streaming SSE/WS para conversas contínuas."""
    conv_resp = api_client.post(
        "/api/conversations",
        json={"title": "Sessão WebSocket"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    conv_id = conv_resp.json()["conversation_id"]

    with api_client.websocket_connect(f"/api/conversations/{conv_id}/ws") as ws:
        # Envia mensagem
        ws.send_json({
            "content": "Verificar status BGP no NE8000-CORE-01",
            "modality": "TEXT",
            "mode": "chat",
        })

        events = []
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt.get("type") == "complete":
                break

        assert len(events) >= 2
        assert any(e.get("type") == "token" for e in events)
        complete_evt = events[-1]
        assert complete_evt["type"] == "complete"
        assert complete_evt["structured_payload"] is not None
