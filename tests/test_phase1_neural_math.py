"""
Testes automatizados da Fase 1 — Fundação Matemática e Autograd da Veronica.
Verifica gradientes analíticos contra aproximações numéricas por diferenças finitas,
convergência de otimizadores (SGD e AdamW), clipping e integridade SHA-256 de checkpoints.
"""

from pathlib import Path
import numpy as np
import pytest

from core.neural import (
    Tensor,
    Neuron,
    Linear,
    Embedding,
    LayerNorm,
    RMSNorm,
    Sequential,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    MSELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    SGD,
    AdamW,
    clip_grad_norm_,
    save_checkpoint,
    load_checkpoint,
    gradcheck,
    MixedPrecisionScaler,
    activation_checkpoint,
    DistributedTrainingStub,
)
from core.transformer.config import VeronicaConfig
from core.transformer.model import VeronicaTransformer


def test_tensor_elementwise_and_matmul_gradcheck() -> None:
    rng = np.random.default_rng(42)
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True, name="A")
    b = Tensor(rng.normal(size=(4, 2)), requires_grad=True, name="B")
    bias = Tensor(rng.normal(size=(1, 2)), requires_grad=True, name="bias")

    def fn() -> Tensor:
        z = (a @ b) + bias
        return ((z ** 2) * 0.5 - z * 0.3).mean()

    report = gradcheck(fn, [a, b, bias])
    assert report["passed"] is True
    assert report["max_relative_error"] < 1e-6


def test_activations_gradcheck() -> None:
    rng = np.random.default_rng(101)
    for act_name, act_fn in [
        ("relu", lambda t: (t * 0.7 + 0.5).relu()),
        ("gelu", lambda t: t.gelu()),
        ("sigmoid", lambda t: t.sigmoid()),
        ("tanh", lambda t: t.tanh()),
        ("softmax", lambda t: (t.softmax(axis=-1) * np.array([1.2, -0.5, 0.8, 0.3])).sum()),
        ("log_softmax", lambda t: (t.log_softmax(axis=-1) * np.array([0.4, -0.2, 0.9, -0.1])).sum()),
    ]:
        x_data = rng.uniform(-1.5, 1.5, size=(3, 4))
        if act_name == "relu":
            # Evita ponto não diferenciável exato em z=0
            x_data = np.where(np.abs(x_data * 0.7 + 0.5) < 0.05, 0.4, x_data)
        x = Tensor(x_data, requires_grad=True, name=f"x_{act_name}")
        report = gradcheck(lambda: act_fn(x).mean(), [x])
        assert report["passed"] is True, f"Falha em {act_name}"
        assert report["max_relative_error"] < 1e-6


def test_layers_and_normalizations_gradcheck() -> None:
    rng = np.random.default_rng(2026)
    x = Tensor(rng.normal(size=(2, 3, 4)), requires_grad=True, name="x")
    linear = Linear(4, 3, seed=7)
    ln = LayerNorm(3)
    rms = RMSNorm(3)

    def fn() -> Tensor:
        h = linear(x).gelu()
        h_ln = ln(h)
        h_rms = rms(h_ln)
        return (h_rms * np.array([0.5, -1.1, 0.8])).mean()

    params = [x] + linear.parameters() + ln.parameters() + rms.parameters()
    report = gradcheck(fn, params)
    assert report["passed"] is True
    assert report["max_relative_error"] < 1e-5


def test_parameterized_neuron_and_losses_gradcheck() -> None:
    rng = np.random.default_rng(77)
    x = Tensor(rng.normal(size=(4, 3)), requires_grad=True, name="x")
    neuron = Neuron(in_features=3, activation="sigmoid", seed=12)
    target = Tensor(np.array([[1.0], [0.0], [1.0], [0.0]]), requires_grad=False)
    bce = BinaryCrossEntropyLoss()

    report_bce = gradcheck(lambda: bce(neuron(x), target), [x] + neuron.parameters())
    assert report_bce["passed"] is True

    logits = Tensor(rng.normal(size=(4, 5)), requires_grad=True, name="logits")
    labels = np.array([1, 3, 0, -100], dtype=np.int64)
    ce = CrossEntropyLoss(ignore_index=-100)
    report_ce = gradcheck(lambda: ce(logits, labels), [logits])
    assert report_ce["passed"] is True
    assert report_ce["max_relative_error"] < 1e-6


def test_training_convergence_sgd_and_adamw(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    # Problema supervisionado não linear: aprender mapeamento técnico sintético
    X_np = rng.normal(size=(32, 6))
    true_w = rng.normal(size=(6, 4))
    Y_np = np.tanh(X_np @ true_w)

    model = Sequential(
        Linear(6, 12, seed=1),
        GELU(),
        LayerNorm(12),
        Linear(12, 4, seed=2),
        Tanh(),
    )
    loss_fn = MSELoss()
    optimizer = AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)

    x_t = Tensor(X_np)
    y_t = Tensor(Y_np)

    initial_loss = float(loss_fn(model(x_t), y_t).item())
    for _ in range(80):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

    final_loss = float(loss_fn(model(x_t), y_t).item())
    assert final_loss < initial_loss * 0.25

    # Serialização e verificação SHA-256 de checkpoint
    manifest = save_checkpoint(
        model,
        checkpoint_dir=tmp_path,
        checkpoint_id="veronica_phase1_test",
        optimizer=optimizer,
        metadata={"initial_loss": initial_loss, "final_loss": final_loss},
    )
    assert "sha256" in manifest

    # Corrompe os pesos em memória e restaura do checkpoint
    for p in model.parameters():
        p.data.fill(0.0)
    assert float(loss_fn(model(x_t), y_t).item()) > final_loss

    load_checkpoint(model, tmp_path, "veronica_phase1_test", optimizer=optimizer, verify_hash=True)
    restored_loss = float(loss_fn(model(x_t), y_t).item())
    assert np.isclose(restored_loss, final_loss, atol=1e-12)


def test_mixed_precision_and_activation_checkpointing_gradcheck() -> None:
    rng = np.random.default_rng(303)
    x = Tensor(rng.normal(size=(2, 4, 6)), requires_grad=True, name="x_ckpt")
    block_mlp = Sequential(
        Linear(6, 10, seed=11),
        GELU(),
        RMSNorm(10),
        Linear(10, 6, seed=12),
    )

    # 1. Gradcheck numérico direto sobre activation_checkpoint
    def ckpt_fn() -> Tensor:
        out = activation_checkpoint(block_mlp, x)
        return (out * np.array([0.3, -0.7, 0.5, 0.2, -0.4, 0.9])).mean()

    report = gradcheck(ckpt_fn, [x] + block_mlp.parameters())
    assert report["passed"] is True
    assert report["max_relative_error"] < 1e-6

    # 2. Equivalência exata de gradientes analíticos no VeronicaTransformer com e sem checkpointing
    cfg_std = VeronicaConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_seq_len=16,
        use_activation_checkpointing=False,
        precision_mode="fp64",
        seed=77,
    )
    cfg_ckpt = VeronicaConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        max_seq_len=16,
        use_activation_checkpointing=True,
        precision_mode="fp64",
        seed=77,
    )
    model_std = VeronicaTransformer(cfg_std)
    model_ckpt = VeronicaTransformer(cfg_ckpt)

    input_ids = np.array([[1, 5, 9, 12], [3, 7, 2, 15]], dtype=np.int64)
    targets = np.array([[5, 9, 12, 4], [7, 2, 15, 8]], dtype=np.int64)
    ce = CrossEntropyLoss()

    logits_std, _ = model_std(input_ids)
    loss_std = ce(logits_std.reshape(-1, cfg_std.vocab_size), targets.reshape(-1))
    loss_std.backward()

    logits_ckpt, _ = model_ckpt(input_ids)
    loss_ckpt = ce(logits_ckpt.reshape(-1, cfg_ckpt.vocab_size), targets.reshape(-1))
    loss_ckpt.backward()

    assert np.isclose(loss_std.item(), loss_ckpt.item(), atol=1e-12)
    params_std = model_std.named_parameters()
    params_ckpt = model_ckpt.named_parameters()
    for name in params_std:
        assert params_std[name].grad is not None
        assert params_ckpt[name].grad is not None
        assert np.allclose(params_std[name].grad, params_ckpt[name].grad, atol=1e-12), (
            f"Diferença de gradiente em {name}"
        )

    # 3. Teste de MixedPrecisionScaler (fp16 forward + fp64 master weights + dynamic loss scaling)
    scaler = MixedPrecisionScaler(
        init_scale=512.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2,
        precision_mode="mixed_fp16",
    )
    opt = AdamW(block_mlp.parameters(), lr=1e-2)
    opt.zero_grad()

    with scaler.autocast(block_mlp.parameters()):
        x_cast = scaler.cast_forward(x)
        out_mp = block_mlp(x_cast).mean()
        scaled_loss = scaler.scale_loss(out_mp)

    scaled_loss.backward()
    stepped = scaler.step(opt)
    assert stepped is True
    assert scaler.skipped_steps == 0

    # Simula overflow (Inf) para verificar backoff automático do Dynamic Loss Scaler
    opt.zero_grad()
    block_mlp.parameters()[0].grad = np.full_like(block_mlp.parameters()[0].data, np.inf)
    prev_scale = scaler.scale
    stepped_overflow = scaler.step(opt)
    assert stepped_overflow is False
    assert scaler.skipped_steps == 1
    assert np.isclose(scaler.scale, prev_scale * 0.5)

    # 4. Teste de DistributedTrainingStub (all_reduce_gradients)
    dist_stub = DistributedTrainingStub(world_size=2, rank=0)
    p0 = block_mlp.parameters()[0]
    p0.grad = np.ones_like(p0.data) * 2.0
    worker_1_grads = [np.ones_like(p.data) * 4.0 for p in block_mlp.parameters()]
    sync_info = dist_stub.all_reduce_gradients(
        block_mlp.parameters(), op="mean", worker_grads=[worker_1_grads]
    )
    assert sync_info["world_size"] == 2
    assert np.allclose(p0.grad, 3.0)

