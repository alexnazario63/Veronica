"""
Suíte de Testes Automatizados para Ontogênese, Neurogênese Artificial e Memórias Humanas da Veronica.
Valida:
1. DynamicNeuron e Neurogênese com conservação matemática de ativações e LTP Hebbiano.
2. NeurogenesisCircuit integrado ao VeronicaTransformer.
3. MemorySystem (Trabalho, Episódica Autobiográfica, Grafo Semântico e Ciclo de Sono/Consolidação).
4. Córtex Perceptual Visual ("Vê") e Córtex de Compreensão Leitora ("Lê").
5. Orquestrador de Desenvolvimento Cognitivo (Estágios, Idade Cognitiva, Marcos Históricos).
6. Endpoints REST da API FastAPI e Diálogo Conversacional com Autoconsciência Ontogenética.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest
import numpy as np

from fastapi.testclient import TestClient

from core.neural.tensor import Tensor
from core.transformer import VeronicaConfig, VeronicaTransformer
from core.development import (
    BinaryMachineMemoryStore,
    DynamicNeuron,
    NeurogenesisCircuit,
    MemorySystem,
    VisualPerceptionCortex,
    ReadingComprehensionCortex,
    DevelopmentEngine,
    DevelopmentalStage,
)
from services.api.app import create_app


def test_dynamic_neuron_mathematics_and_hebbian_ltp():
    """Testa o neurônio artificial parametrizado vivo, forward, projeção e plasticidade sináptica."""
    in_dim = 48
    out_dim = 48
    seed_vec = np.ones((in_dim,), dtype=np.float64)
    neuron = DynamicNeuron(
        in_features=in_dim,
        out_features=out_dim,
        trigger_concept="BGP_SESSION_ACTIVE",
        trigger_type="READING",
        seed_vector=seed_vec,
        seed=123,
    )

    assert neuron.developmental_stage == "NEONATAL"
    assert neuron.firing_count == 0
    assert neuron.synaptic_strength == 1.0
    assert 0.1 <= neuron.myelination_score <= 0.2

    # Forward simples com Tensor
    x_data = np.ones((1, 4, in_dim), dtype=np.float64) * 0.5
    x = Tensor(x_data, requires_grad=True)
    act = neuron(x)
    assert act.shape == (1, 4, 1)

    # Projeção sináptica de saída
    proj = neuron.project_outgoing(act)
    assert proj.shape == (1, 4, out_dim)

    # Simulação de disparos repetidos com regra Hebbiana
    initial_ltp = neuron.synaptic_strength
    initial_myelin = neuron.myelination_score
    for _ in range(15):
        neuron.fire_and_adapt(activation_val=0.8, context_concept="ROUTING_OSPF")

    assert neuron.firing_count == 15
    assert neuron.synaptic_strength > initial_ltp
    assert neuron.myelination_score > initial_myelin
    assert neuron.developmental_stage in ("SYNAPTICALLY_INTEGRATED", "MATURE")
    assert "ROUTING_OSPF" in neuron.co_activated_concepts

    meta = neuron.export_meta()
    assert meta["neuron_id"].startswith("NEURON-")
    assert meta["trigger_concept"] == "BGP_SESSION_ACTIVE"
    assert len(meta["coords_3d"]) == 3


def test_neurogenesis_circuit_and_transformer_integration():
    """Valida a expansão dinâmica do circuito de neurogênese e integração ao VeronicaTransformer."""
    d_model = 48
    circuit = NeurogenesisCircuit(d_model=d_model, seed=42)

    # Cria 3 neurônios dinâmicos
    n1 = circuit.birth_neuron(trigger_concept="TOPOLOGY_RING", trigger_type="VISION")
    n2 = circuit.birth_neuron(trigger_concept="DWDM_OPTICAL", trigger_type="VISION")
    n3 = circuit.birth_neuron(trigger_concept="BGP_EVPN", trigger_type="READING")

    assert len(circuit.neurons) == 3

    # Forward integrado com modulação residual
    x = Tensor(np.ones((2, 6, d_model), dtype=np.float64), requires_grad=False)
    mod_x, stats = circuit.forward_integrated(x, context_concept="TOPOLOGY_RING")

    assert mod_x.shape == x.shape
    assert stats["total_neurons"] == 3
    assert "firing_neurons" in stats

    # Integração ao VeronicaTransformer
    cfg = VeronicaConfig(
        vocab_size=128,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=64,
        norm_type="rmsnorm",
        positional_type="rope",
        activation="gelu",
        seed=10,
    )
    model = VeronicaTransformer(cfg)
    model.attach_neurogenesis_circuit(circuit)

    input_ids = np.array([[5, 12, 19, 23]], dtype=np.int64)
    logits, _ = model.forward(input_ids)
    assert logits.shape == (1, 4, 128)

    # Exportação gráfica deve conter o circuito de neurogênese e neurônios nascidos
    graph = model.export_architecture_graph()
    node_ids = [node["id"] for node in graph["nodes"]]
    assert "neurogenesis_associative_circuit" in node_ids

    # O slice micro deve conter amostras de neurônios nascidos
    born_micro = [m for m in graph["micro_neuron_slice"] if m.get("is_born_neuron")]
    assert len(born_micro) >= 3


def test_human_like_multi_tiered_memory_system():
    """Valida Memória de Trabalho, Episódica, Semântica e Ciclo de Sono."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        mem = MemorySystem(storage_dir=Path(tmp_dir), max_working_memory=3)


        # 1. Memória de Trabalho (limite de 3 itens)
        mem.push_working_memory("ITEM_1", "TEXT", "Primeiro item")
        mem.push_working_memory("ITEM_2", "TEXT", "Segundo item")
        mem.push_working_memory("ITEM_3", "TEXT", "Terceiro item")
        mem.push_working_memory("ITEM_4", "TEXT", "Quarto item deve deslocar o primeiro")

        active_wm = mem.get_working_memory_snapshot()
        assert len(active_wm) == 3
        concepts_wm = [item["concept"] for item in active_wm]
        assert "ITEM_1" not in concepts_wm
        assert "ITEM_4" in concepts_wm

        # 2. Memória Episódica
        ep1 = mem.record_episodic_memory(
            modality="VISION",
            experience_summary="Vi uma topologia em anel com 4 roteadores Huawei NE8000.",
            perceptual_details={"topology": "ANEL", "nodes": 4},
            emotive_valence="FASCINIO",
            salience=0.9,
            associated_concepts=["TOPOLOGY_RING", "HUAWEI_NE8000"],
            created_neuron_ids=["N-01", "N-02"],
        )
        assert ep1.memory_id.startswith("EPM-")
        assert ep1.consolidation_level == 0.1

        # Busca episódica
        search_res = mem.search_episodic_memories("topologia em anel")
        assert len(search_res) >= 1
        assert "EPM-" in search_res[0]["memory_id"]

        # 3. Memória Semântica
        mem.add_semantic_relation(
            source_concept="TOPOLOGY_RING",
            target_concept="PROTECTION_SWITCHING",
            relation="ENABLES",
            strength=0.8,
        )
        assert "topology_ring" in mem.semantic_nodes
        assert "protection_switching" in mem.semantic_nodes

        # 4. Ciclo de Sono e Consolidação
        cycle_res = mem.run_sleep_and_consolidation_cycle()
        assert cycle_res["memories_consolidated"] >= 1
        assert cycle_res["synaptic_edges_strengthened"] >= 1
        assert "reflexão" in cycle_res["reflection_dream_report"].lower()
        # O nível de consolidação do episódio aumentou
        assert mem.episodic_memories[ep1.memory_id].consolidation_level > 0.5


def test_visual_cortex_and_reading_comprehension():
    """Valida a percepção multimodal: Ver imagens/topologias e Ler documentos."""
    vis_cortex = VisualPerceptionCortex(d_model=48)
    read_cortex = ReadingComprehensionCortex(d_model=48)

    # 1. Córtex Visual ("Vê")
    # Imagem simulada com topologia em anel e alerta
    vis_res = vis_cortex.see(
        image_input="diagrama de rede com topologia em anel contendo NE8000-Core-01 e enlace óptico vermelho rompido",
        context_hint="Alerta no Core",
    )
    assert vis_res.detected_topology == "ANEL"
    assert len(vis_res.detected_components) >= 2
    assert len(vis_res.visual_alarms) >= 1
    assert vis_res.perceptual_embedding.shape == (48,)
    assert "TOPOLOGY_ANEL" in vis_res.suggested_concepts_for_neurogenesis

    # 2. Córtex de Leitura ("Lê")
    technical_doc = """
    RFC 4271 — Border Gateway Protocol 4 (BGP-4).
    O protocolo BGP deve manter conexões TCP na porta 179 entre peers configurados.
    Quando ocorre divergência de MTU ou bloqueio de firewall, a sessão permanece em estado Active.
    Recomenda-se no Huawei VRP e Cisco IOS-XR a utilização de commit confirmed para evitar perda de gerência.
    Comandos essenciais: display bgp peer e show bgp summary.
    """
    read_res = read_cortex.read(technical_doc, title="RFC 4271 BGP-4")
    assert read_res.word_count > 30
    assert "BGP" in read_res.detected_protocols
    assert "Huawei" in read_res.detected_vendors or "Cisco" in read_res.detected_vendors
    assert len(read_res.detected_commands) >= 1
    assert len(read_res.extracted_rules_and_insights) >= 1
    assert read_res.reading_embedding.shape == (48,)


def test_development_engine_lifecycle_and_growth():
    """Valida a ontogênese completa da Veronica: Ver, Ler, Dormir, Novos Neurônios e Auto-Relato."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        engine = DevelopmentEngine(d_model=48, storage_dir=Path(tmp_dir))

        assert engine.current_stage == DevelopmentalStage.INFANCY
        initial_neurons = len(engine.neurogenesis.circuit.neurons)
        assert initial_neurons == 0

        # Veronica "Vê"
        see_out = engine.see_and_learn(
            image_input="diagrama com anel de fibra óptica 100GE e 4 roteadores Huawei NE8000",
            context_hint="Visão de Core",
            user_id="usr-eng-01",
        )
        assert see_out["new_neurons_count"] >= 1
        assert len(engine.neurogenesis.circuit.neurons) >= 1
        assert engine.total_images_seen == 1
        assert "Primeiro Olhar (Visão de Imagem)" in [m.name for m in engine.milestones]

        # Veronica "Lê"
        read_out = engine.read_and_learn(
            text_content="Protocolo OSPFv3 com áreas NSSA em roteadores Juniper Junos e Cisco IOS-XR. Comandos show ospf neighbor.",
            title="Manual de Engenharia OSPFv3",
            user_id="usr-eng-01",
        )
        assert read_out["new_neurons_count"] >= 1
        assert engine.total_texts_read == 1
        assert "Primeira Leitura Ativa" in [m.name for m in engine.milestones]

        # Veronica "Dorme" (Consolidação de Memórias)
        sleep_out = engine.sleep_and_consolidate(user_id="usr-eng-01")
        assert sleep_out["consolidation_cycle"]["memories_consolidated"] >= 2
        assert "dream_report" in sleep_out

        # Auto-Relato Introspectivo
        reflection = engine.introspective_reflection()
        assert reflection["born_neurons_count"] >= 2
        assert reflection["episodic_memories_count"] >= 2
        assert "Verônica" in reflection["introspective_narrative"]
        assert "novos neurônios" in reflection["introspective_narrative"]

        # Persistência e restauração
        engine.save_state()
        engine_restored = DevelopmentEngine(d_model=48, storage_dir=Path(tmp_dir))
        assert engine_restored.total_images_seen == 1
        assert engine_restored.total_texts_read == 1
        assert len(engine_restored.milestones) >= 2


def test_api_development_endpoints_and_chat_awareness():
    """Valida os endpoints REST FastAPI de desenvolvimento e a interação via Chat."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        app = create_app(data_dir=Path(tmp_dir))

        client = TestClient(app)

        # 1. GET /api/development/status
        res_status = client.get("/api/development/status")
        assert res_status.status_code == 200
        st_data = res_status.json()
        assert "developmental_stage" in st_data
        assert "total_born_neurons" in st_data
        assert "introspective_narrative" in st_data

        # 2. POST /api/development/see ("Ver")
        res_see = client.post(
            "/api/development/see",
            json={
                "image_data": "diagrama com topologia em anel e 3 switches CloudEngine",
                "context_hint": "Topologia de Acesso",
            },
            headers={"x-user-id": "usr-eng-01"},
        )
        assert res_see.status_code == 200
        see_data = res_see.json()
        assert see_data["new_neurons_count"] >= 1
        assert "perceptual_analysis" in see_data

        # 3. POST /api/development/read ("Ler")
        res_read = client.post(
            "/api/development/read",
            json={
                "title": "Guia de Boas Práticas BGP",
                "content": "Configuração de BGP peer no Huawei VRP: peer 10.0.0.1 as-number 65000. Requer commit.",
            },
            headers={"x-user-id": "usr-eng-01"},
        )
        assert res_read.status_code == 200
        read_data = res_read.json()
        assert read_data["new_neurons_count"] >= 1
        assert "reading_analysis" in read_data

        # 4. POST /api/development/consolidate ("Sono")
        res_sleep = client.post(
            "/api/development/consolidate",
            headers={"x-user-id": "usr-eng-01"},
        )
        assert res_sleep.status_code == 200
        sleep_data = res_sleep.json()
        assert "dream_report" in sleep_data

        # 5. GET /api/development/neurons
        res_neurons = client.get("/api/development/neurons")
        assert res_neurons.status_code == 200
        n_data = res_neurons.json()
        assert n_data["total_born_neurons"] >= 2
        assert len(n_data["nodes"]) >= 2

        # 6. GET /api/development/memories
        res_mem = client.get("/api/development/memories")
        assert res_mem.status_code == 200
        mem_data = res_mem.json()
        assert mem_data["total_episodic_memories"] >= 2
        assert len(mem_data["recent_episodic_timeline"]) >= 2

        # 7. Diálogo Conversacional no Chat sobre seu próprio desenvolvimento
        conv_res = client.post("/api/conversations", json={"title": "Perguntando sobre crescimento"})
        conv_id = conv_res.json()["conversation_id"]

        msg_res = client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": "Veronica, como você está se desenvolvendo e quantos neurônios novos você já criou?"},
            headers={"x-user-id": "usr-eng-01"},
        )
        assert msg_res.status_code == 200
        payload = msg_res.json()["structured_payload"]
        assert payload["intent"] == "DEVELOPMENT"
        assert "neurônios" in msg_res.json()["content"].lower()
        assert "memórias" in msg_res.json()["content"].lower()


def test_binary_machine_memory_store_and_auto_persistence():
    """Valida BinaryMachineMemoryStore, magic header, serialização de registros e auto-persistência."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        bin_file = tmp_path / "custom_memory.bin"

        mem = MemorySystem(storage_dir=tmp_path)
        mem.push_working_memory("WM_BGP", "TEXT", "Investigando peering BGP com Loopback")
        mem.record_episodic_memory(
            modality="CONVERSATION",
            experience_summary="Operador relatou falha de adjacência OSPF.",
            associated_concepts=["OSPF", "MTU"],
            salience=0.85,
        )

        # 1. Salvar manualmente e verificar arquivo binário
        store = BinaryMachineMemoryStore(memory_system=mem)
        bytes_written = store.save_binary_file(bin_file)
        assert bytes_written > len(BinaryMachineMemoryStore.MAGIC_HEADER)
        assert bin_file.exists()

        # 2. Inspecionar cabeçalho e estatísticas
        stats = BinaryMachineMemoryStore.inspect_binary_file(bin_file)
        assert stats["file_exists"] is True
        assert stats["is_valid_binary"] is True
        assert stats["magic_header"] == "VRNC_MEM\x02\x00"
        assert stats["episodic_count"] == 1
        assert stats["semantic_nodes_count"] >= 6  # base concepts + OSPF/MTU

        # 3. Carregar em nova instância de MemorySystem a partir do arquivo
        mem2 = MemorySystem()
        load_res = store.load_binary_file(bin_file, memory_system=mem2)
        assert load_res["episodic_count"] == 1
        assert "ospf" in mem2.semantic_nodes

        # 4. Auto-salvamento em veronica_machine_memory.bin
        auto_bin = tmp_path / "veronica_machine_memory.bin"
        assert auto_bin.exists()
        auto_stats = mem.get_binary_stats()
        assert auto_stats["is_valid_binary"] is True
        assert auto_stats["file_exists"] is True
        assert auto_stats["episodic_count"] >= 1

        # 5. Rejeição de arquivo com header corrompido
        corrupt_file = tmp_path / "corrupt.bin"
        corrupt_file.write_bytes(b"INVALID_HEADER_GARBAGE_12345678901234567890")
        with pytest.raises(ValueError, match="Magic header inválido"):
            store.load_binary_file(corrupt_file, mem2)


def test_chat_modes_chat_agent_work_and_binary_stats_endpoint():
    """Valida os modos CHAT, AGENT, WORK e o endpoint GET /api/memory/binary/stats."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        app = create_app(data_dir=Path(tmp_dir))
        client = TestClient(app)

        # 1. GET /api/memory/binary/stats
        stats_resp = client.get("/api/memory/binary/stats")
        assert stats_resp.status_code == 200
        stats_data = stats_resp.json()
        assert stats_data["file_exists"] is True
        assert stats_data["is_valid_binary"] is True
        assert "VRNC_MEM" in stats_data["magic_header"]
        assert stats_data["semantic_nodes_count"] >= 6

        # 2. Cria conversa
        conv = client.post("/api/conversations", json={"title": "Testes Multimodais de Modos"}).json()
        cid = conv["conversation_id"]

        # 3. Modo CHAT (padrão)
        chat_resp = client.post(
            f"/api/conversations/{cid}/messages",
            json={
                "content": "Sessão BGP em estado Active no Huawei NE8000-CORE-01",
                "mode": "chat",
            },
            headers={"x-user-id": "usr-eng-01"},
        )
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        assert chat_data["structured_payload"]["mode"] == "chat"
        assert "epistemic_breakdown" in chat_data["structured_payload"]
        assert len(chat_data["structured_payload"]["epistemic_breakdown"]["verified_facts"]) >= 1

        # 4. Modo AGENT (ReAct Thought -> Action -> Observation)
        agent_resp = client.post(
            f"/api/conversations/{cid}/messages",
            json={
                "content": "Verifique também OSPF preso em EXSTART no ASR9K-CORE-02",
                "mode": "agent",
            },
            headers={"x-user-id": "usr-eng-01"},
        )
        assert agent_resp.status_code == 200
        agent_data = agent_resp.json()
        assert agent_data["structured_payload"]["mode"] == "agent"
        assert "Thought" in agent_data["content"]
        assert "Action" in agent_data["content"]
        assert "Observation" in agent_data["content"]

        # 5. Modo WORK (Geração de artefatos CLI e relatórios)
        work_resp = client.post(
            f"/api/conversations/{cid}/messages",
            json={
                "content": "Gere o roteiro de comandos e rollback para correção de MTU no OSPF",
                "mode": "work",
            },
            headers={"x-user-id": "usr-eng-01"},
        )
        assert work_resp.status_code == 200
        work_data = work_resp.json()
        assert work_data["structured_payload"]["mode"] == "work"
        assert "```cli" in work_data["content"]
        assert "Rollback" in work_data["content"] or "rollback" in work_data["content"].lower()

        # 6. Memória binária atualizada em tempo real com as interações
        updated_stats = client.get("/api/memory/binary/stats").json()
        assert updated_stats["episodic_count"] >= 3


def test_autonomous_learning_daemon_background_neurogenesis_and_memory_without_dialogue():
    """
    Valida a capacidade da Veronica de aprender, criar memórias e gerar novos neurônios
    sozinha em segundo plano, independente de haver diálogo humano ou não.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        app = create_app(data_dir=tmp_path)
        client = TestClient(app)

        # 1. Verifica status do daemon de auto-aprendizado autônomo na inicialização
        status_resp = client.get("/api/development/autonomous-learning/status")
        assert status_resp.status_code == 200
        status = status_resp.json()
        assert status["is_running"] is True
        initial_neurons = status["total_neurons_in_circuit"]
        initial_memories = status["total_episodic_memories"]

        # 2. Executa ciclos de pensamento autônomo (sem qualquer interação conversacional)
        cycle1 = client.post("/api/development/autonomous-learning/cycle").json()
        assert cycle1["cycle"] >= 1
        assert cycle1["concept"] is not None
        assert len(cycle1["neurons_born"]) >= 1
        assert cycle1["memory_id"] is not None

        cycle2 = client.post("/api/development/autonomous-learning/cycle").json()
        assert cycle2["cycle"] >= 2
        assert len(cycle2["neurons_born"]) >= 1

        # 3. Verifica que os neurônios e memórias cresceram espontaneamente
        status2 = client.get("/api/development/autonomous-learning/status").json()
        assert status2["total_neurons_in_circuit"] > initial_neurons
        assert status2["total_episodic_memories"] > initial_memories
        assert status2["autonomous_neurons_born"] >= 2
        assert status2["autonomous_memories_created"] >= 2
        assert "Pensamento autônomo" in status2["current_thought"]

        # 4. Verifica que o arquivo binário veronica_machine_memory.bin foi gravado
        bin_stats = client.get("/api/memory/binary/stats").json()
        assert bin_stats["file_exists"] is True
        assert bin_stats["is_valid_binary"] is True
        assert bin_stats["file_size_bytes"] > 0
        assert bin_stats["episodic_count"] >= 2

        # 5. Testa controle de pausa, retomada e ajuste de intervalo
        stop_resp = client.post("/api/development/autonomous-learning/stop").json()
        assert stop_resp["is_running"] is False

        interval_resp = client.post(
            "/api/development/autonomous-learning/interval",
            json={"interval_seconds": 2.5},
        ).json()
        assert interval_resp["interval_seconds"] == 2.5

        start_resp = client.post("/api/development/autonomous-learning/start").json()
        assert start_resp["is_running"] is True

        # 6. Verifica inclusão no endpoint geral /api/development/status
        dev_status = client.get("/api/development/status").json()
        assert "autonomous_learning" in dev_status
        assert dev_status["autonomous_learning"]["is_running"] is True
        assert dev_status["total_born_neurons"] > initial_neurons


def test_multi_worker_autonomous_learning_2ms_interval_and_neuron_persistence():
    """
    Valida:
    1. Multi-workers no daemon de aprendizado autônomo (ex: 4 workers paralelos).
    2. Suporte a cadência ultra-rápida de até 2ms (0.002s / interval_ms=2.0).
    3. Persistência permanente dos neurônios criados: ao parar o motor ou recarregar
       o backend, nenhum neurônio é perdido (salvos em veronica_neurogenesis_circuit.json).
    4. Validação de limites (rejeição de intervalo < 2ms ou workers < 1 ou > 64).
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        app = create_app(data_dir=tmp_path)
        client = TestClient(app)

        # 1. Configura multi-workers (4 workers) e intervalo ultra-rápido de 2ms
        config_resp = client.post(
            "/api/development/autonomous-learning/config",
            json={"workers": 4, "interval_ms": 2.0},
        )
        assert config_resp.status_code == 200
        cfg_data = config_resp.json()
        assert cfg_data["workers"] == 4
        assert cfg_data["interval_ms"] == 2.0
        assert cfg_data["interval_seconds"] == 0.002

        # 2. Executa ciclos de geração de neurônios
        c1 = client.post("/api/development/autonomous-learning/cycle").json()
        c2 = client.post("/api/development/autonomous-learning/cycle").json()
        c3 = client.post("/api/development/autonomous-learning/cycle").json()

        st_active = client.get("/api/development/autonomous-learning/status").json()
        total_neurons = st_active["total_neurons_in_circuit"]
        assert total_neurons >= 3
        assert st_active["workers"] == 4
        assert st_active["interval_ms"] == 2.0

        # 3. Para o motor cognitivo: Os neurônios NÃO PODEM sumir!
        stop_resp = client.post("/api/development/autonomous-learning/stop").json()
        assert stop_resp["is_running"] is False

        # Status após parada: neurônios continuam existindo e não são zerados
        st_stopped = client.get("/api/development/autonomous-learning/status").json()
        assert st_stopped["is_running"] is False
        neurons_when_stopped = st_stopped["total_neurons_in_circuit"]
        assert neurons_when_stopped >= total_neurons
        assert st_stopped["autonomous_neurons_born"] >= 3

        # Verificação de estabilidade: com o motor parado, a contagem não se altera
        st_stopped_2 = client.get("/api/development/autonomous-learning/status").json()
        assert st_stopped_2["total_neurons_in_circuit"] == neurons_when_stopped

        # 4. Verifica persistência no arquivo JSON no disco
        circuit_file = tmp_path / "veronica_neurogenesis_circuit.json"
        assert circuit_file.exists()
        assert circuit_file.stat().st_size > 50

        # 5. Simula reinício completo do backend/DevelopmentEngine com o mesmo diretório
        reloaded_engine = DevelopmentEngine(d_model=48, storage_dir=tmp_path)
        assert len(reloaded_engine.neurogenesis.circuit.neurons) == neurons_when_stopped

        # Valida propriedades dos neurônios restaurados
        first_neuron = reloaded_engine.neurogenesis.circuit.neurons[0]
        assert hasattr(first_neuron, "neuron_id")
        assert first_neuron.neuron_id.startswith("NEURON-")
        assert first_neuron.trigger_concept is not None

        # 6. Validação de limites (intervalo mínimo 2ms / 0.002s e workers 1..64)
        bad_interval = client.post(
            "/api/development/autonomous-learning/interval",
            json={"interval_ms": 1.0},  # Menor que 2ms deve falhar
        )
        assert bad_interval.status_code == 422

        bad_workers = client.post(
            "/api/development/autonomous-learning/config",
            json={"workers": 0},  # Menor que 1 deve falhar
        )
        assert bad_workers.status_code == 422

        bad_workers_max = client.post(
            "/api/development/autonomous-learning/config",
            json={"workers": 100},  # Maior que 64 deve falhar
        )
        assert bad_workers_max.status_code == 422


