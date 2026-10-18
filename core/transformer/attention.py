"""
Scaled Dot-Product Attention e Multi-Head Causal Self-Attention com KV-Cache
da arquitetura VeronicaTransformer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

from core.neural.tensor import Tensor
from core.neural.layers import Linear, Module
from core.transformer.positional import RotaryPositionalEmbedding


def create_causal_mask(seq_q: int, seq_k: int, offset: int = 0) -> np.ndarray:
    """
    Cria máscara booleana onde True indica posições futuras proibidas (j > offset + i).
    """
    q_indices = np.arange(seq_q)[:, np.newaxis] + offset
    k_indices = np.arange(seq_k)[np.newaxis, :]
    return k_indices > q_indices


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal_mask: Optional[np.ndarray] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + M) @ V
    q, k, v possuem shape (B, n_heads, T, head_dim).
    Retorna (context, attn_probs).
    """
    d_k = q.shape[-1]
    scale = 1.0 / math.sqrt(float(d_k))
    scores = (q @ k.swapaxes(-1, -2)) * scale
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask, -1e9)
    attn_probs = scores.softmax(axis=-1)
    context = attn_probs @ v
    return context, attn_probs


class MultiHeadCausalAttention(Module):
    """
    Multi-Head Causal Self-Attention com suporte a RoPE, máscara causal,
    KV-Cache para inferência incremental e captura de mapas de atenção.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope: Optional[RotaryPositionalEmbedding] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, bias=True, seed=seed)
        self.k_proj = Linear(d_model, d_model, bias=True, seed=seed + 1)
        self.v_proj = Linear(d_model, d_model, bias=True, seed=seed + 2)
        self.out_proj = Linear(d_model, d_model, bias=True, seed=seed + 3)

        self.last_attention_stats: Dict[str, Any] = {}

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_cache: bool = False,
        capture_diagnostics: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[np.ndarray, np.ndarray]]]:
        bsz, seq_len, _ = x.shape
        offset = int(past_kv[0].shape[2]) if past_kv is not None else 0

        q = self.q_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose((0, 2, 1, 3))
        k = self.k_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose((0, 2, 1, 3))
        v = self.v_proj(x).reshape(bsz, seq_len, self.n_heads, self.head_dim).transpose((0, 2, 1, 3))

        if self.rope is not None:
            q = self.rope.apply_rotary(q, offset=offset)
            k = self.rope.apply_rotary(k, offset=offset)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = Tensor(np.concatenate([past_k, k.data], axis=2), requires_grad=False)
            v = Tensor(np.concatenate([past_v, v.data], axis=2), requires_grad=False)

        present_kv = (k.data.copy(), v.data.copy()) if use_cache else None

        total_k_len = k.shape[2]
        mask = create_causal_mask(seq_len, total_k_len, offset=offset)
        context, attn_probs = scaled_dot_product_attention(q, k, v, causal_mask=mask)

        if capture_diagnostics:
            probs_np = attn_probs.data
            entropy = -np.sum(probs_np * np.log(np.maximum(probs_np, 1e-12)), axis=-1)
            self.last_attention_stats = {
                "mean_entropy": float(np.mean(entropy)),
                "max_attention": float(np.max(probs_np)),
                "head_entropies": [float(np.mean(entropy[:, h])) for h in range(self.n_heads)],
                "sample_matrix": probs_np[0, 0, : min(16, seq_len), : min(16, total_k_len)].tolist(),
            }

        merged = context.transpose((0, 2, 1, 3)).reshape(bsz, seq_len, self.d_model)
        output = self.out_proj(merged)
        return output, present_kv
