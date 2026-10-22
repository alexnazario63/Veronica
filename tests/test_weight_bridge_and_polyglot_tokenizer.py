"""
Testes automatizados para o Universal Weight Loader, Conector de Inferência Neural,
Tokenizador Poliglota Dinâmico (BPE / Byte-Level) e Critérios de Neurogênese Ontogenética.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pytest

from core.transformer import (
    VeronicaConfig,
    VeronicaTransformer,
    SafetensorsWeightBridge,
    NeuralInferenceConnector,
)
from core.tokenizer import TechnicalTokenizer, SPECIAL_TOKENS, BASE_NETWORK_VOCAB
from core.development import (
    DevelopmentEngine,
    DevelopmentalStage,
    MemorySystem,
    NeurogenesisEngine,
    BinaryMachineMemoryStore,
)


def test_polyglot_tokenizer_lossless_multilingual() -> None:
    tok = TechnicalTokenizer()
    assert tok.vocab_size > 0
    assert len(tok.special_tokens) == len(SPECIAL_TOKENS)

    multilingual_texts = [
        "Olá, mundo! Verônica operando com inferência neural nativa.",
        "Hello, World! Dense probabilistic inference with top-p nucleus sampling.",
        "你好世界，深度学习与全自主网络工程。",
        "Привет, мир! Архитектура трансформера с памятью.",
        "مرحبا بالعالم! شبكة عصبية فائقة القدرة.",
        "こんにちは世界！自律型エージェントの思考。",
        "안녕하세요 세계! 완벽한 바이트 레벨 토큰화.",
        "Equação: E = mc² | Símbolos: 🚀⚡✨ © ® ™ µ",
    ]

    for text in multilingual_texts:
        ids = tok.encode(text, add_bos=True, add_eos=True)
        assert tok.unk_id not in ids, f"Token <unk> encontrado em: {text}"
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == text, f"Divergência de roundtrip: {decoded} != {text}"


def test_polyglot_tokenizer_code_and_latex() -> None:
    tok = TechnicalTokenizer()

    # 1. Código com identação estrita e sintaxe de múltiplas linguagens
    code_samples = [
        # Python
        (
            "def optimize_route(prefix: str, cost: int) -> bool:\n"
            "    if cost > 100:\n"
            "        return False\n"
            "    return True\n"
        ),
        # TypeScript
        (
            "interface RoutePolicy {\n"
            "  community: string;\n"
            "  med: number;\n"
            "}\n"
        ),
        # Rust
        (
            "pub fn forward_packet(buf: &[u8]) -> Result<(), &'static str> {\n"
            "    Ok(())\n"
            "}\n"
        ),
        # SQL
        (
            "SELECT id, hostname, ip_address FROM inventory WHERE status = 'ACTIVE' ORDER BY id ASC;"
        ),
    ]

    for code in code_samples:
        ids = tok.encode(code)
        assert tok.unk_id not in ids
        decoded = tok.decode(ids)
        assert decoded == code

    # 2. Fórmulas LaTeX
    latex_formulas = [
        r"\mathcal{L}_{total} = \frac{1}{N} \sum_{i=1}^N \| y_i - \hat{y}_i \|_2^2 + \lambda \|\theta\|_2",
        r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
        r"\alpha + \beta + \gamma \leq \tau_{novelty}",
    ]

    for formula in latex_formulas:
        ids = tok.encode(formula)
        assert tok.unk_id not in ids
        decoded = tok.decode(ids)
        assert decoded == formula


def test_polyglot_tokenizer_network_primitives_and_fit() -> None:
    tok = TechnicalTokenizer()

    # Primitivas de rede
    net_sample = (
        "interface GigabitEthernet0/0/0\n"
        " ip address 10.254.1.1/30\n"
        " display bgp peer 192.168.100.2 verbose"
    )
    ids = tok.encode(net_sample, add_bos=True, add_eos=True)
    assert tok.unk_id not in ids
    assert tok.decode(ids, skip_special_tokens=True) == net_sample

    # fit dinâmico
    new_tokens_added = tok.fit(
        ["telemetria_quântica", "telemetria_quântica", "micro_serviço_óptico", "micro_serviço_óptico"],
        min_freq=2,
    )
    assert new_tokens_added >= 1


def test_safetensors_weight_bridge_roundtrip(tmp_path: Path) -> None:
    tok = TechnicalTokenizer()
    cfg = VeronicaConfig(
        vocab_size=tok.vocab_size + 64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=64,
        seed=101,
    )
    model = VeronicaTransformer(cfg)
    bridge = SafetensorsWeightBridge()

    # Salva em .safetensors
    sf_file = tmp_path / "model_weights.safetensors"
    bridge.save_weights(sf_file, model, format="safetensors")
    assert sf_file.exists()

    # Salva em .npz
    npz_file = tmp_path / "model_weights.npz"
    bridge.save_weights(npz_file, model, format="npz")
    assert npz_file.exists()

    # Carrega em um novo modelo e valida coincidência numérica
    fresh_model = VeronicaTransformer(cfg)
    report = bridge.load_into_model(sf_file, fresh_model, strict=True)
    assert report["success"] is True
    assert len(report["missing_parameters"]) == 0

    p_orig = model.named_parameters()
    p_fresh = fresh_model.named_parameters()
    for name in p_orig:
        np.testing.assert_allclose(
            p_orig[name].data,
            p_fresh[name].data,
            rtol=1e-5,
            err_msg=f"Discrepância no parâmetro {name}",
        )

    # Carregamento de NPZ via método de conveniência do próprio modelo
    fresh_npz_model = VeronicaTransformer(cfg)
    rep_npz = fresh_npz_model.load_weights(npz_file)
    assert rep_npz["success"] is True


def test_neural_inference_connector_enrichment_and_generation(tmp_path: Path) -> None:
    tok = TechnicalTokenizer()
    cfg = VeronicaConfig(
        vocab_size=tok.vocab_size + 64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=64,
        seed=42,
    )
    model = VeronicaTransformer(cfg)

    mem = MemorySystem(storage_dir=tmp_path)
    neuro = NeurogenesisEngine(d_model=32, storage_dir=tmp_path, tau_novelty=0.75, inactivity_prune_cycles=1000)
    dev = DevelopmentEngine(d_model=32, storage_dir=tmp_path)

    # Semeia conceito e memória episódica
    mem._touch_or_create_concept("OSPF", category="PROTOCOL", description="Open Shortest Path First")
    mem.commit_episodic_memory(
        event=type(
            "MockEpisodic",
            (),
            {
                "memory_id": "EPISODE-001",
                "timestamp": "2026-10-22T10:00:00Z",
                "modality": "DIAGNOSTICS",
                "experience_summary": "Convergencia OSPF restabelecida na area 0",
                "associated_concepts": ["OSPF", "AREA0"],
                "salience": 0.95,
                "consolidation_level": 0.9,
            },
        )()
    )

    connector = NeuralInferenceConnector(
        model=model,
        tokenizer=tok,
        memory_system=mem,
        neurogenesis_engine=neuro,
        development_engine=dev,
    )

    # 1. Enriquecimento de contexto com memórias em tempo de execução
    enrichment = connector.enrich_prompt_context("Status da area 0 no OSPF")
    assert "OSPF" in enrichment["enriched_prompt"]
    assert len(enrichment["retrieved_concepts"]) > 0

    # 2. Geração neural densa probabilística
    result = connector.generate_text(
        prompt="Status da area 0 no OSPF",
        max_new_tokens=12,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
    )
    assert result["tokens_generated"] > 0
    assert isinstance(result["text"], str)
    assert result["latency_ms"] >= 0.0
    assert len(result["enriched_concepts"]) > 0


def test_developmental_stages_and_neurogenesis_tau(tmp_path: Path) -> None:
    # 1. Estágios de desenvolvimento
    assert DevelopmentalStage.INFANT == DevelopmentalStage.INFANCY
    assert DevelopmentalStage.CHILD == DevelopmentalStage.EARLY_LEARNING
    assert DevelopmentalStage.ADOLESCENT == DevelopmentalStage.SYNAPTIC_EXPANSION
    assert DevelopmentalStage.SPECIALIST == DevelopmentalStage.SPECIALIZATION
    assert DevelopmentalStage.EXPERT == DevelopmentalStage.MASTERY

    # 2. Motor de Neurogênese com tau_novelty = 0.75
    engine = NeurogenesisEngine(d_model=48, storage_dir=tmp_path, tau_novelty=0.75, inactivity_prune_cycles=1000)
    assert engine.novelty_threshold == 0.75
    assert engine.inactivity_prune_cycles == 1000

    # Avaliação de surpresa com sinal novo
    signal = np.random.randn(48).astype(np.float32)
    surprise = engine.evaluate_signal_surprise("test_circuit", signal)
    assert surprise >= 0.75

    # Nascimento do neurônio
    neuron = engine.trigger_neurogenesis("test_circuit", signal, reason="Novo padrão de tráfego")
    assert neuron is not None
    assert neuron.firing_count == 0

    # Poda com limiar de inatividade
    pruned = engine.prune_idle_synapses(inactivity_threshold_cycles=1000)
    assert isinstance(pruned, int)
