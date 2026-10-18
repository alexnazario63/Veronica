"""
Testes automatizados da Fase 2 — Tokenizador Técnico, Arquitetura VeronicaTransformer,
Atenção Causal com RoPE, KV-Cache, Treinamento Next-Token e Inferência Autoregressiva.
"""

from pathlib import Path
import numpy as np

from core.neural import Tensor, gradcheck
from core.tokenizer import TechnicalTokenizer
from core.transformer import (
    VeronicaConfig,
    RotaryPositionalEmbedding,
    MultiHeadCausalAttention,
    VeronicaTransformer,
)
from core.training import TrainingHyperparameters, VeronicaTrainer
from core.inference import GenerationConfig, AutoregressiveGenerator


def test_technical_tokenizer_roundtrip_and_zero_unk() -> None:
    tok = TechnicalTokenizer()
    sample = (
        "interface GigabitEthernet0/0/0\n"
        " ip address 10.254.1.1/30\n"
        " display bgp peer 192.168.100.2 verbose"
    )
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert tok.unk_id not in ids
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert decoded == sample


def test_rope_and_multihead_attention_gradcheck() -> None:
    rng = np.random.default_rng(123)
    rope = RotaryPositionalEmbedding(head_dim=4, max_seq_len=16)
    mha = MultiHeadCausalAttention(d_model=8, n_heads=2, rope=rope, seed=10)
    x = Tensor(rng.normal(size=(2, 4, 8)), requires_grad=True, name="x")

    def fn() -> Tensor:
        out, _ = mha(x, use_cache=False)
        return (out * 0.7).mean()

    report = gradcheck(fn, [x] + mha.parameters())
    assert report["passed"] is True
    assert report["max_relative_error"] < 1e-5


def test_causal_mask_prevents_future_token_leakage() -> None:
    cfg = VeronicaConfig(vocab_size=128, d_model=16, n_heads=2, n_layers=2, d_ff=32, max_seq_len=16)
    model = VeronicaTransformer(cfg)
    model.eval()

    seq_a = np.array([[10, 20, 30, 40]], dtype=np.int64)
    seq_b = np.array([[10, 20, 30, 99]], dtype=np.int64)

    logits_a, _ = model.forward(seq_a)
    logits_b, _ = model.forward(seq_b)

    # Os logits nas posições 0, 1 e 2 devem ser idênticos mesmo alterando o token na posição 3
    assert np.allclose(logits_a.data[:, :3, :], logits_b.data[:, :3, :], atol=1e-12)


def test_kv_cache_matches_full_sequence_forward() -> None:
    cfg = VeronicaConfig(vocab_size=128, d_model=16, n_heads=2, n_layers=2, d_ff=32, max_seq_len=16)
    model = VeronicaTransformer(cfg)
    model.eval()

    seq = np.array([[5, 12, 25, 33]], dtype=np.int64)
    full_logits, _ = model.forward(seq, use_cache=False)

    # Passo 1: prefixo [5, 12, 25] com cache
    _, past_kvs = model.forward(seq[:, :3], use_cache=True)
    # Passo 2: último token [33] usando past_kvs
    cached_logits, _ = model.forward(seq[:, 3:], past_kvs=past_kvs, use_cache=True)

    assert np.allclose(full_logits.data[:, -1, :], cached_logits.data[:, -1, :], atol=1e-11)


def test_veronica_trainer_convergence_and_generation(tmp_path: Path) -> None:
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
    trainer = VeronicaTrainer(model=model, tokenizer=tok, checkpoint_dir=tmp_path)

    hparams = TrainingHyperparameters(
        epochs=6,
        batch_size=4,
        seq_len=16,
        learning_rate=0.015,
        min_lr=0.002,
        warmup_steps=1,
    )
    result = trainer.run_experiment("exp-phase2", hparams=hparams)
    assert result["state"] == "COMPLETED"
    assert result["final_train_loss"] < result["initial_train_loss"] * 0.65

    arch = model.export_architecture_graph()
    assert arch["agent_name"] == "Veronica"
    assert arch["parameter_counts"]["trainable"] > 0
    assert len(arch["nodes"]) >= 6

    gen = AutoregressiveGenerator(model, tok)
    out = gen.generate_text(
        "Pergunta: Sessao BGP em estado Active",
        gen_cfg=GenerationConfig(max_new_tokens=12, greedy=True),
    )
    assert out["tokens_generated"] > 0
    assert isinstance(out["text"], str)
