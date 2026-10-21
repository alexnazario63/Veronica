"""
Codificação Posicional (Sinusoidal, Treinável e Rotary Position Embedding - RoPE)
da arquitetura VeronicaTransformer.
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np

from core.neural.tensor import Tensor
from core.neural.layers import Module, Parameter
from core.neural.init import normal_


class SinusoidalPositionalEncoding(Module):
    """Codificação posicional trigonométrica clássica (Vaswani et al.)."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        position = np.arange(max_seq_len, dtype=np.float64)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float64) * (-math.log(10000.0) / d_model)
        )
        pe = np.zeros((max_seq_len, d_model), dtype=np.float64)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.pe = pe

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        slice_pe = self.pe[offset : offset + seq_len]
        return x + Tensor(slice_pe, requires_grad=False)


class LearnedPositionalEncoding(Module):
    """Embeddings posicionais treináveis."""

    def __init__(self, max_seq_len: int, d_model: int, seed: Optional[int] = None) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        self.weight = Parameter(
            np.empty((max_seq_len, d_model), dtype=np.float64),
            name="pos_weight",
        )
        normal_(self.weight, mean=0.0, std=0.02, rng=rng)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        pos_slice = self.weight[offset : offset + seq_len]
        return x + pos_slice


class RotaryPositionalEmbedding(Module):
    """
    RoPE (Rotary Position Embedding) com gradiente analítico exato:
    Rotaciona pares de dimensões (2i, 2i+1) nas cabeças de Query e Key pelo ângulo m * theta_i.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 1024, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (
            base ** (np.arange(0, head_dim, 2, dtype=np.float64) / float(head_dim))
        )
        t = np.arange(max_seq_len, dtype=np.float64)
        freqs = np.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
        self.cos_cached = np.cos(freqs)
        self.sin_cached = np.sin(freqs)

    def apply_rotary(self, x: Tensor, offset: int = 0) -> Tensor:
        """
        Aplica rotação RoPE a um tensor `x` de shape (B, n_heads, T, head_dim).
        Como a rotação é ortogonal R(theta), sua transposta no passo backward é R(-theta).
        """
        seq_len = x.shape[-2]
        cos = self.cos_cached[offset : offset + seq_len][np.newaxis, np.newaxis, :, :]
        sin = self.sin_cached[offset : offset + seq_len][np.newaxis, np.newaxis, :, :]

        x_even = x.data[..., 0::2]
        x_odd = x.data[..., 1::2]

        out_data = np.empty_like(x.data)
        out_data[..., 0::2] = x_even * cos - x_odd * sin
        out_data[..., 1::2] = x_even * sin + x_odd * cos

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _children=(x,) if x.requires_grad else (),
            _op="rope",
        )

        def _backward() -> None:
            if out.grad is not None and x.requires_grad:
                x._ensure_grad()
                g_even = out.grad[..., 0::2]
                g_odd = out.grad[..., 1::2]
                dx = np.empty_like(out.grad)
                dx[..., 0::2] = g_even * cos + g_odd * sin
                dx[..., 1::2] = -g_even * sin + g_odd * cos
                x.grad += dx

        out._backward = _backward
        return out
