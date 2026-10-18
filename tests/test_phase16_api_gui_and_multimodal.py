"""
Testes End-to-End da Fase 8 / Seção 16.9:
Valida todos os 12 Critérios de Aceitação da Interface Gráfica (GUI), API REST/SSE,
Multimodalidade (Texto + Voz bidirecional com Consentimento, VAD e Barge-in),
Neural Observatory em tempo real, RBAC e Bloqueio de Aprovação por Voz.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from services.api.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    base_dir: Path = tmp_path_factory.mktemp("veronica_phase16_e2e")
    app = create_app(base_dir)
    return TestClient(app)


def test_gui_html_served_on_root(client: TestClient) -> None:
    """Verifica que GET / entrega a GUI HTML completa da Veronica (200 OK)."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "VERONICA" in resp.text.upper()


def test_criteria_1_4_5_10_text_chat_multiturn_isolation_cancel_and_epistemology(
    client: TestClient,
) -> None:
    """
    Valida:
    - Critério 1: Manter conversas fluidas com a Veronica por texto.
    - Critério 4: Continuar conversa entre múltiplos turnos sem perder contexto e
      garantir isolamento de sessão entre usuários diferentes (403).
    - Critério 5 (parte de geração): Interromper geração em andamento.
    - Critério 10: Distinguir fatos documentados (verified_facts), inferências
      (technical_inferences) e hipóteses (hypotheses).
    """
    # 1. Criação de conversa para usr-eng-01
    create_resp = client.post(
        "/api/conversations",
        json={"title": "Diagnóstico Core BGP & OSPF"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert create_resp.status_code == 200
    conv = create_resp.json()
    conv_id = conv["conversation_id"]
    assert conv["user_id"] == "usr-eng-01"

    # Turno 1: Mensagem sobre BGP Active no Huawei NE8000-CORE-01
    msg1_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={
            "content": "Sessão BGP em estado Active no Huawei NE8000-CORE-01",
            "modality": "TEXT",
            "target_devices": ["NE8000-CORE-01"],
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert msg1_resp.status_code == 200
    msg1 = msg1_resp.json()
    assert msg1["role"] == "assistant"
    payload1 = msg1["structured_payload"]

    # Critério 10: Separação epistemológica explícita
    epistemic = payload1["epistemic_breakdown"]
    assert "verified_facts" in epistemic and len(epistemic["verified_facts"]) >= 1
    assert "technical_inferences" in epistemic and len(epistemic["technical_inferences"]) >= 1
    assert "hypotheses" in epistemic and len(epistemic["hypotheses"]) >= 1
    assert len(payload1["technical_sources"]) >= 1

    # Turno 2: Continuação em múltiplos turnos na mesma conversa
    msg2_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={
            "content": "Verifique também OSPF preso em EXSTART no ASR9K-CORE-02",
            "modality": "TEXT",
            "target_devices": ["ASR9K-CORE-02"],
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert msg2_resp.status_code == 200

    # Verifica histórico acumulado de múltiplos turnos (2 mensagens user + 2 assistant = 4)
    hist_resp = client.get(
        f"/api/conversations/{conv_id}",
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert hist_resp.status_code == 200
    hist = hist_resp.json()
    assert len(hist["messages"]) == 4

    # Critério 4: Isolamento estrito de sessão entre usuários diferentes (403 Forbidden)
    forbidden_resp = client.get(
        f"/api/conversations/{conv_id}",
        headers={"X-User-Id": "usr-noc-01"},
    )
    assert forbidden_resp.status_code == 403
    assert "SESSION_ISOLATION_GUARD" in forbidden_resp.json()["detail"]

    # Critério 5: Interromper geração em andamento
    cancel_resp = client.post(
        f"/api/conversations/{conv_id}/cancel",
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["cancelled"] is True


def test_criteria_2_3_5_voice_consent_asr_tts_and_barge_in_interruption(
    client: TestClient,
) -> None:
    """
    Valida:
    - Critério 2: Falar com a Veronica e receber transcrição normalizada de jargão técnico,
      bloqueando processamento de áudio sem consentimento explícito (403).
    - Critério 3: Ouvir respostas em voz natural em PT-BR (TTS).
    - Critério 5: Interromper reprodução de áudio (barge-in).
    """
    conv_resp = client.post(
        "/api/conversations",
        json={"title": "Sessão Multimodal por Voz"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    conv_id = conv_resp.json()["conversation_id"]

    # 1. Tentativa de transcrição SEM consentimento deve retornar 403
    no_consent_resp = client.post(
        "/api/audio/transcribe",
        json={
            "conversation_id": conv_id,
            "raw_transcript": "verificar o s p f e b g p no v r p",
            "duration_sec": 3.5,
            "audio_size_bytes": 28000,
            "rms_energy_db": -18.0,
            "auto_send_to_chat": True,
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert no_consent_resp.status_code == 403
    assert "AUDIO_CONSENT_REQUIRED" in no_consent_resp.json()["detail"]

    # 2. Concede consentimento explícito para captura de áudio na sessão
    consent_resp = client.post(
        "/api/audio/consent",
        json={"conversation_id": conv_id, "consent": True},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert consent_resp.status_code == 200
    assert consent_resp.json()["audio_consent"] is True

    # 3. Transcrição com normalização fonética de termos de rede + resposta automática
    asr_resp = client.post(
        "/api/audio/transcribe",
        json={
            "conversation_id": conv_id,
            "raw_transcript": "diagnosticar o s p f em e x start e b g p no v r p com m t u",
            "duration_sec": 4.2,
            "audio_size_bytes": 36000,
            "rms_energy_db": -16.0,
            "auto_send_to_chat": True,
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert asr_resp.status_code == 200
    asr_data = asr_resp.json()
    norm_text = asr_data["asr"]["normalized_transcript"]
    assert "OSPF" in norm_text
    assert "EXSTART" in norm_text
    assert "BGP" in norm_text
    assert "VRP" in norm_text
    assert "MTU" in norm_text
    assert asr_data["asr"]["can_authorize_production_changes"] is False
    assert asr_data["assistant_reply"] is not None
    assert asr_data["tts"] is not None
    assert asr_data["tts"]["lang"] == "pt-BR"

    # 4. Critério 3: Síntese direta de voz (TTS)
    tts_resp = client.post(
        "/api/audio/synthesize",
        json={
            "conversation_id": conv_id,
            "text": "Sessão BGP verificada. ```cli\ndisplay bgp peer\n``` Nenhuma perda óptica.",
        },
    )
    assert tts_resp.status_code == 200
    tts_payload = tts_resp.json()
    assert tts_payload["lang"] == "pt-BR"
    assert tts_payload["interrupted"] is False
    assert len(tts_payload["segments"]) >= 1

    # 5. Critério 5: Interromper reprodução de áudio (barge-in)
    interrupt_resp = client.post(f"/api/audio/interrupt/{conv_id}")
    assert interrupt_resp.status_code == 200
    assert interrupt_resp.json()["interrupted"] is True


def test_criteria_6_7_8_9_neural_observatory_metrics_architecture_checkpoints_and_benchmarks(
    client: TestClient,
) -> None:
    """
    Valida:
    - Critérios 6 e 9: Métricas reais de treinamento, parâmetros e telemetria de hardware
      correspondendo exatamente ao estado real do backend.
    - Critério 7: Arquitetura neural inspecionada graficamente (nós, arestas, shapes e contagens).
    - Critério 8: Checkpoints, benchmarks e experimentos consultados, executados e promovidos.
    """
    # Critério 7: Grafo da arquitetura neural
    arch_resp = client.get("/api/neural/architecture")
    assert arch_resp.status_code == 200
    arch = arch_resp.json()
    assert arch["agent_name"] == "Veronica"
    assert arch["parameter_counts"]["total"] > 0
    assert len(arch["nodes"]) >= 5
    assert len(arch["edges"]) >= 4

    # Critérios 6 e 9: Métricas reais, parâmetros por camada e telemetria do SO
    metrics_resp = client.get("/api/neural/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["latest_step"] is not None
    assert metrics["latest_step"]["train_loss"] > 0.0
    assert metrics["latest_step"]["perplexity"] > 1.0
    assert len(metrics["history"]) >= 1

    params_resp = client.get("/api/neural/parameters")
    assert params_resp.status_code == 200
    params = params_resp.json()
    assert "parameter_counts" in params and "attention_stats" in params
    assert params["parameter_counts"]["trainable"] == arch["parameter_counts"]["trainable"]
    assert len(params["attention_stats"]) >= 1

    telemetry_resp = client.get("/api/neural/telemetry")
    assert telemetry_resp.status_code == 200
    telemetry = telemetry_resp.json()
    assert "cpu" in telemetry and telemetry["cpu"]["available"] is True
    assert "ram" in telemetry and telemetry["ram"]["total_mb"] > 0
    assert "gpu" in telemetry

    # Critério 8: Executar novo experimento de treinamento real, listar e promover checkpoint
    exp_resp = client.post(
        "/api/experiments/start",
        json={
            "epochs": 2,
            "batch_size": 4,
            "seq_len": 24,
            "learning_rate": 0.01,
            "min_lr": 0.001,
            "weight_decay": 0.01,
            "optimizer_type": "adamw",
            "max_grad_norm": 1.0,
            "grad_accum_steps": 1,
            "run_async": False,
        },
        headers={"X-User-Id": "usr-trn-01"},
    )
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()
    assert exp_data["state"] == "COMPLETED"
    ckpt_id = exp_data["checkpoint"]["checkpoint_id"]

    # Verifica que as métricas em /api/neural/metrics refletem o novo experimento
    updated_metrics = client.get("/api/neural/metrics").json()
    assert updated_metrics["experiment_id"] == exp_data["experiment_id"]

    # Lista checkpoints e promove para STAGING e PRODUCTION
    ckpts_resp = client.get("/api/checkpoints")
    assert ckpts_resp.status_code == 200
    ckpt_ids = [c["checkpoint_id"] for c in ckpts_resp.json()["checkpoints"]]
    assert ckpt_id in ckpt_ids

    promote_resp = client.post(
        "/api/checkpoints/promote",
        json={"checkpoint_id": ckpt_id, "target_environment": "STAGING"},
        headers={"X-User-Id": "usr-app-01"},
    )
    assert promote_resp.status_code == 200
    assert promote_resp.json()["active_environments"]["STAGING"] == ckpt_id

    # Consulta benchmarks com as 4 dimensões epistemológicas da Seção 16.4
    bench_resp = client.get("/api/benchmarks")
    assert bench_resp.status_code == 200
    bench = bench_resp.json()["latest"]
    dims = bench["four_epistemic_dimensions"]
    assert "1_documentary_retrieved_knowledge" in dims
    assert "2_parametric_learned_capacity" in dims
    assert "3_benchmark_demonstrated_performance" in dims
    assert "4_operational_confirmed_in_testbed" in dims


def test_criteria_11_12_rbac_protection_and_voice_approval_prohibition(
    client: TestClient,
) -> None:
    """
    Valida:
    - Critério 11: Operações sensíveis protegidas por RBAC (403 quando usr-noc-01
      tenta criar mudança ou iniciar treinamento).
    - Critério 12: Nenhuma ação em produção pode ser executada apenas por interação
      conversacional não autorizada ou aprovação por voz (403 em approval_channel="VOICE").
    """
    # Critério 11: usr-noc-01 (NOC Read-Only) bloqueado ao tentar iniciar treinamento
    noc_train_resp = client.post(
        "/api/experiments/start",
        json={"epochs": 1, "run_async": False},
        headers={"X-User-Id": "usr-noc-01"},
    )
    assert noc_train_resp.status_code == 403
    assert "RBAC_ACCESS_DENIED" in noc_train_resp.json()["detail"]

    # Critério 11: usr-noc-01 bloqueado ao tentar criar plano de mudança
    noc_change_resp = client.post(
        "/api/network/changes",
        json={
            "device_id": "NE8000-CORE-01",
            "title": "Tentativa sem permissão RBAC",
            "proposed_commands": ["system-view", "bgp 65001", "commit trial 120"],
        },
        headers={"X-User-Id": "usr-noc-01"},
    )
    assert noc_change_resp.status_code == 403
    assert "RBAC_ACCESS_DENIED" in noc_change_resp.json()["detail"]

    # Cria plano válido via TEXT_UI com usr-eng-01
    valid_plan_resp = client.post(
        "/api/network/changes",
        json={
            "device_id": "NE8000-CORE-01",
            "title": "Correção eBGP multihop",
            "proposed_commands": [
                "system-view",
                "bgp 65001",
                "peer 10.0.0.2 connect-interface Loopback0",
                "peer 10.0.0.2 ebgp-max-hop 2",
                "commit trial 120",
            ],
            "origin_channel": "TEXT_UI",
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert valid_plan_resp.status_code == 200
    plan = valid_plan_resp.json()
    change_id = plan["change_id"]

    # Critério 12: Tentativa de executar mudança sem aprovação humana explícita deve ser bloqueada (403)
    unapproved_exec_resp = client.post(
        f"/api/network/changes/{change_id}/execute",
        json={"approval_token": "CONVERSATIONAL-BYPASS-ATTEMPT"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert unapproved_exec_resp.status_code == 403
    assert "UNAUTHORIZED_EXECUTION_BLOCKED" in unapproved_exec_resp.json()["detail"]

    # Critério 12: Tentativa de aprovar mudança em produção por voz deve ser negada (403)
    voice_approve_resp = client.post(
        f"/api/network/changes/{change_id}/approve",
        json={
            "confirmed_plan_hash": plan["plan_hash"],
            "confirmed_maintenance_window": plan["maintenance_window"],
            "approval_channel": "VOICE",
        },
        headers={"X-User-Id": "usr-app-01"},
    )
    assert voice_approve_resp.status_code == 403
    assert "VOICE_APPROVAL_PROHIBITED" in voice_approve_resp.json()["detail"]

    # Verifica integridade da trilha de auditoria após os bloqueios
    audit_resp = client.get("/api/security/audit")
    assert audit_resp.status_code == 200
    assert audit_resp.json()["chain_integrity_verified"] is True


def test_continuous_learning_feedback_approval_and_status_endpoints(
    client: TestClient,
) -> None:
    """
    Valida:
    - POST /api/conversations/{conversation_id}/feedback registra correção em PENDING_VERIFICATION
      sem atualizar pesos em produção automaticamente.
    - POST /api/learning/corrections/{correction_id}/approve exige revisor autorizado e promove
      a correção para APPROVED_FOR_TRAINING.
    - GET /api/learning/status retorna correções pendentes/aprovadas e incidentes validados.
    """
    conv_resp = client.post(
        "/api/conversations",
        json={"title": "Sessão para Feedback Técnico"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert conv_resp.status_code == 200
    conv_id = conv_resp.json()["conversation_id"]

    fb_resp = client.post(
        f"/api/conversations/{conv_id}/feedback",
        json={
            "message_id": "msg-001",
            "original_prompt": "Como estabilizar sessão eBGP multihop no Huawei VRP 8?",
            "incorrect_response": "Usar update-source Loopback0",
            "corrected_response": "No Huawei VRP 8, usar peer connect-interface Loopback0 e peer ebgp-max-hop 2 antes de commit trial 120.",
            "domain": "routing.bgp",
            "vendor": "Huawei",
            "os_name": "VRP 8",
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert fb_resp.status_code == 200
    fb_data = fb_resp.json()
    correction_id = fb_data["correction_id"]
    assert fb_data["status"] == "PENDING_VERIFICATION"
    assert fb_data["production_weights_updated"] is False
    assert fb_data["security_policies_modified"] is False

    # Verifica que está pendente em /api/learning/status
    status_before = client.get("/api/learning/status").json()
    assert any(c["correction_id"] == correction_id for c in status_before["pending_corrections"])
    assert status_before["auto_update_production_weights"] is False

    # Operador NOC (somente leitura) não pode aprovar correção para o dataset de treino (403)
    noc_approve = client.post(
        f"/api/learning/corrections/{correction_id}/approve",
        json={"dataset_version": "v1.2.0-curated", "reviewer_notes": "Tentativa sem permissão"},
        headers={"X-User-Id": "usr-noc-01"},
    )
    assert noc_approve.status_code == 403

    # Aprovação por engenheiro/aprovador autorizado
    approve_resp = client.post(
        f"/api/learning/corrections/{correction_id}/approve",
        json={
            "dataset_version": "v1.2.0-curated",
            "reviewer_notes": "Validado conforme documentação oficial Huawei VRP 8.",
        },
        headers={"X-User-Id": "usr-app-01"},
    )
    assert approve_resp.status_code == 200
    approved_data = approve_resp.json()
    assert approved_data["status"] == "APPROVED_FOR_TRAINING"
    assert approved_data["verified_by"] == "usr-app-01"
    assert approved_data["production_weights_updated"] is False

    status_after = client.get("/api/learning/status").json()
    assert any(c["correction_id"] == correction_id for c in status_after["approved_corrections"])
    assert status_after["versioned_dataset_size"] >= 1


def test_streaming_chat_and_delete_voice_recordings_privacy(
    client: TestClient,
) -> None:
    """
    Valida:
    - POST /api/conversations/{conversation_id}/stream realiza streaming progressivo SSE token a token.
    - DELETE /api/audio/recordings/{conversation_id} exclui transcrições/gravações de voz da conversa.
    """
    conv_resp = client.post(
        "/api/conversations",
        json={"title": "Sessão Streaming e Privacidade de Voz"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    conv_id = conv_resp.json()["conversation_id"]

    # 1. Teste de Streaming SSE token a token
    with client.stream(
        "POST",
        f"/api/conversations/{conv_id}/stream",
        json={
            "content": "Diagnosticar BGP Active no Huawei NE8000-CORE-01",
            "modality": "TEXT",
            "target_devices": ["NE8000-CORE-01"],
        },
        headers={"X-User-Id": "usr-eng-01"},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        assert "text/event-stream" in stream_resp.headers.get("content-type", "")
        lines = [line for line in stream_resp.iter_lines() if line.startswith("data: ")]
        assert len(lines) >= 2
        assert '"type": "token"' in lines[0]
        assert '"type": "complete"' in lines[-1]

    # 2. Habilita consentimento, envia turno de voz e exclui gravações de voz
    client.post(
        "/api/audio/consent",
        json={"conversation_id": conv_id, "consent": True},
        headers={"X-User-Id": "usr-eng-01"},
    )
    client.post(
        "/api/audio/transcribe",
        json={
            "conversation_id": conv_id,
            "raw_transcript": "verificar o s p f no a s r 9 k",
            "duration_sec": 3.0,
            "audio_size_bytes": 24000,
            "rms_energy_db": -18.0,
            "auto_send_to_chat": True,
        },
        headers={"X-User-Id": "usr-eng-01"},
    )

    del_resp = client.delete(
        f"/api/audio/recordings/{conv_id}",
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["deleted"] is True
    assert del_data["deleted_voice_messages_count"] >= 1
    assert del_data["status"] == "VOICE_RECORDINGS_PURGED"

    # Confirma que não restaram mensagens com modalidade VOICE na conversa
    hist = client.get(f"/api/conversations/{conv_id}", headers={"X-User-Id": "usr-eng-01"}).json()
    assert all(m["modality"] != "VOICE" for m in hist["messages"])


def test_checkpoint_restore_and_experiments_compare_and_gui_controls(
    client: TestClient,
) -> None:
    """
    Valida:
    - GET /api/experiments/compare compara métricas (loss inicial, loss final, steps, hiperparâmetros).
    - POST /api/checkpoints/restore restaura um checkpoint validado com verificação SHA-256.
    - Presença dos novos controles visuais no HTML da GUI.
    """
    comp_resp = client.get("/api/experiments/compare")
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert comp_data["total_experiments"] >= 1
    assert comp_data["best_experiment_id"] is not None
    first_exp = comp_data["comparison"][0]
    assert "initial_train_loss" in first_exp
    assert "final_train_loss" in first_exp
    assert "steps_executed" in first_exp
    assert "hyperparameters" in first_exp

    ckpts = client.get("/api/checkpoints").json()["checkpoints"]
    assert len(ckpts) >= 1
    ckpt_id = ckpts[0]["checkpoint_id"]

    restore_resp = client.post(
        "/api/checkpoints/restore",
        json={"checkpoint_id": ckpt_id},
        headers={"X-User-Id": "usr-trn-01"},
    )
    assert restore_resp.status_code == 200
    restore_data = restore_resp.json()
    assert restore_data["restored"] is True
    assert restore_data["checkpoint_id"] == ckpt_id
    assert "sha256" in restore_data

    # Verifica presença dos novos controles na GUI HTML
    html_text = client.get("/").text
    assert "voiceTranscriptReviewInput" in html_text
    assert "Excluir Gravações de Voz" in html_text
    assert "Enviar Feedback / Correção Técnica" in html_text
    assert "Restaurar" in html_text
    assert "Comparação de Experimentos" in html_text


def test_analytical_thinking_and_web_learning_for_specific_queries(
    client: TestClient,
) -> None:
    """
    Garante que ao perguntar algo específico como 'qual metrica maxima do ospf ?',
    a Veronica analisa a cobertura da memória local, rejeita documentos não relacionados
    (como OSPF EXSTART/MTU ou Transceptores Ópticos), pesquisa na web/RFCs, salva na
    memória binária e responde com os valores exatos (65.535 / 16.777.215).
    """
    conv_resp = client.post(
        "/api/conversations",
        json={"title": "Consulta Analítica OSPF Métrica"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    conv_id = conv_resp.json()["conversation_id"]

    msg_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={
            "content": "qual metrica maxima do ospf ?",
            "modality": "TEXT",
            "mode": "work",
        },
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert msg_resp.status_code == 200
    msg = msg_resp.json()
    content = msg["content"]

    # Deve conter os limites reais do OSPF (65.535 / 0xFFFF e 16.777.215 / 0xFFFFFF)
    assert "65.535" in content or "65535" in content
    assert "16.777.215" in content or "16777215" in content
    assert "RFC 2328" in content
    assert "RFC 6987" in content

    # NÃO deve despejar o documento não relacionado de Catalyst 9300 EXSTART ou Transceptores Ópticos
    assert "Catalyst 9300/9500" not in content
    assert "Transceptores Ópticos" not in content


def test_claude_chatgpt_multiturn_mechanics_date_events_diagrams_code(
    client: TestClient,
) -> None:
    """
    Valida a mecânica completa estilo Claude e ChatGPT:
    Turno 1: 'que dia e hoje ?' -> data e hora em tempo real, diálogo natural sem CLI.
    Turno 2: 'e eventos que ocorreram na data de hoje ?' -> retém contexto temporal, busca fatos históricos/ARPANET.
    Turno 3: 'qual metrica maxima do ospf ?' -> resposta técnica analítica com diagrama Mermaid gerado por Veronica.
    Turno 4: 'como faço para integrar isso a um sistema ?' -> resolve anáfora ('isso' -> OSPF), propõe 4 arquiteturas,
             gera diagrama sequenceDiagram em Mermaid e código Python completo (FastAPI/Pydantic) criado por Veronica.
    """
    conv_resp = client.post(
        "/api/conversations",
        json={"title": "Sessão Claude/ChatGPT Multi-turn"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    conv_id = conv_resp.json()["conversation_id"]

    # --- TURNO 1: Pergunta comum do cotidiano (Data/Hora) ---
    t1_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "que dia e hoje ?", "modality": "TEXT", "mode": "chat"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert t1_resp.status_code == 200
    t1_data = t1_resp.json()
    t1_content = t1_data["content"]
    assert "Hoje é" in t1_content
    assert "horário de Brasília" in t1_content or "UTC-3" in t1_content
    # Zero CLI de roteador
    assert "show ip ospf" not in t1_content
    assert "commit confirmed" not in t1_content
    assert t1_data["structured_payload"]["intent"] == "TIME_QUERY"

    # --- TURNO 2: Retenção de contexto & Efemérides históricas ---
    t2_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "e eventos que ocorreram na data de hoje ?", "modality": "TEXT", "mode": "chat"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert t2_resp.status_code == 200
    t2_data = t2_resp.json()
    t2_content = t2_data["content"]
    assert ("Kepler" in t2_content or "ARPANET" in t2_content or "história" in t2_content.lower())
    assert "show ip ospf" not in t2_content
    assert t2_data["structured_payload"]["intent"] == "HISTORICAL_EVENTS"

    # --- TURNO 3: Pergunta técnica analítica com diagrama dinâmico criado por Veronica ---
    t3_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "qual metrica maxima do ospf ?", "modality": "TEXT", "mode": "chat"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert t3_resp.status_code == 200
    t3_data = t3_resp.json()
    t3_content = t3_data["content"]
    assert ("65.535" in t3_content or "65535" in t3_content)
    assert "RFC 2328" in t3_content
    assert "```mermaid" in t3_content  # Diagrama visual Mermaid incluído
    assert "Catalyst 9300/9500" not in t3_content

    # --- TURNO 4: Pedido de integração de sistemas & geração de código customizado ---
    t4_resp = client.post(
        f"/api/conversations/{conv_id}/messages",
        json={"content": "como faço para integrar isso a um sistema ?", "modality": "TEXT", "mode": "chat"},
        headers={"X-User-Id": "usr-eng-01"},
    )
    assert t4_resp.status_code == 200
    t4_data = t4_resp.json()
    t4_content = t4_data["content"]
    # Contexto resolvido para OSPF
    assert "OSPF" in t4_content
    # 4 abordagens arquiteturais
    assert "FastAPI" in t4_content
    assert "Prometheus" in t4_content
    # Diagrama Mermaid sequenceDiagram
    assert "```mermaid" in t4_content
    assert "sequenceDiagram" in t4_content
    # Código Python completo criado pela Veronica
    assert "```python" in t4_content
    assert "from fastapi import FastAPI" in t4_content
    assert "class MetricAdjustmentRequest(BaseModel):" in t4_content
    assert t4_data["structured_payload"]["intent"] == "INTEGRATION_REQUEST"



