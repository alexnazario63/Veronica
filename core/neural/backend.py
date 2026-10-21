"""Array backend selection and optional compiled neural kernels.

The Autograd implementation remains NumPy based.  This module provides an
explicit array namespace for inference kernels and keeps NumPy as the reliable
CPU/gradcheck default; CuPy is selected only when requested and available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Numba is an acceleration, never a runtime requirement.
    from numba import njit
except Exception:  # pragma: no cover - depends on optional runtime
    njit = None


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    compiled: bool


def get_backend(name: str = "numpy") -> ArrayBackend:
    """Return a backend namespace; ``auto`` chooses CuPy only when usable."""
    choice = name.lower()
    if choice not in {"numpy", "cupy", "auto"}:
        raise ValueError("backend must be 'numpy', 'cupy', or 'auto'")
    if choice in {"cupy", "auto"}:
        try:
            import cupy as cp  # type: ignore[reportMissingImports]

            if cp.cuda.runtime.getDeviceCount() > 0:
                return ArrayBackend("cupy", cp, njit is not None)
        except Exception:
            if choice == "cupy":
                raise RuntimeError("CuPy requested but no usable CUDA device is available")
    return ArrayBackend("numpy", np, njit is not None)


def set_backend(name: str = "numpy") -> ArrayBackend:
    """Set the process default backend and return it."""
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = get_backend(name)
    return _DEFAULT_BACKEND


_DEFAULT_BACKEND = ArrayBackend("numpy", np, njit is not None)


def array_namespace() -> Any:
    return _DEFAULT_BACKEND.xp


def _rmsnorm_numpy(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def _layernorm_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _kernel(fn: Any) -> Any:
    return njit(cache=False)(fn) if njit is not None else fn


@_kernel
def _rmsnorm_kernel(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rows = x.reshape((-1, x.shape[-1]))
    out = np.empty_like(rows)
    for row in range(rows.shape[0]):
        mean_sq = 0.0
        for col in range(rows.shape[1]):
            mean_sq += rows[row, col] * rows[row, col]
        scale = 1.0 / np.sqrt(mean_sq / rows.shape[1] + eps)
        for col in range(rows.shape[1]):
            out[row, col] = rows[row, col] * scale * weight[col]
    return out.reshape(x.shape)


@_kernel
def _layernorm_kernel(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    rows = x.reshape((-1, x.shape[-1]))
    out = np.empty_like(rows)
    for row in range(rows.shape[0]):
        mean = 0.0
        for col in range(rows.shape[1]):
            mean += rows[row, col]
        mean /= rows.shape[1]
        variance = 0.0
        for col in range(rows.shape[1]):
            delta = rows[row, col] - mean
            variance += delta * delta
        scale = 1.0 / np.sqrt(variance / rows.shape[1] + eps)
        for col in range(rows.shape[1]):
            out[row, col] = (rows[row, col] - mean) * scale * weight[col] + bias[col]
    return out.reshape(x.shape)


@_kernel
def _gelu_kernel(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    k = np.sqrt(2.0 / np.pi)
    for i in range(x.size):
        value = x.flat[i]
        u = k * (value + 0.044715 * value * value * value)
        out.flat[i] = 0.5 * value * (1.0 + np.tanh(u))
    return out


@_kernel
def _rope_kernel(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[-2]):
        for j in range(x.shape[-1] // 2):
            even, odd = x[..., i, 2 * j], x[..., i, 2 * j + 1]
            out[..., i, 2 * j] = even * cos[i, j] - odd * sin[i, j]
            out[..., i, 2 * j + 1] = even * sin[i, j] + odd * cos[i, j]
    return out


@_kernel
def _attention_kernel(q: np.ndarray, k: np.ndarray, v: np.ndarray, offset: int) -> np.ndarray:
    out = np.zeros_like(q)
    scale = 1.0 / np.sqrt(q.shape[-1])
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for i in range(q.shape[2]):
                end = min(k.shape[2], offset + i + 1)
                maximum = -1e300
                for j in range(end):
                    score = 0.0
                    for d in range(q.shape[3]):
                        score += q[b, h, i, d] * k[b, h, j, d]
                    maximum = max(maximum, score * scale)
                denom = 0.0
                for j in range(end):
                    score = 0.0
                    for d in range(q.shape[3]):
                        score += q[b, h, i, d] * k[b, h, j, d]
                    denom += np.exp(score * scale - maximum)
                for j in range(end):
                    score = 0.0
                    for d in range(q.shape[3]):
                        score += q[b, h, i, d] * k[b, h, j, d]
                    prob = np.exp(score * scale - maximum) / denom
                    for d in range(v.shape[3]):
                        out[b, h, i, d] += prob * v[b, h, j, d]
    return out


def rms_norm(x: Any, weight: Any, eps: float = 1e-8) -> Any:
    if not type(x).__module__.startswith("cupy"):
        return _rmsnorm_kernel(np.asarray(x), np.asarray(weight), eps)
    return get_backend("cupy").xp.asarray(x) / get_backend("cupy").xp.sqrt(get_backend("cupy").xp.mean(get_backend("cupy").xp.asarray(x) ** 2, axis=-1, keepdims=True) + eps) * get_backend("cupy").xp.asarray(weight)


def layer_norm(x: Any, weight: Any, bias: Any, eps: float = 1e-5) -> Any:
    if not type(x).__module__.startswith("cupy"):
        return _layernorm_kernel(np.asarray(x), np.asarray(weight), np.asarray(bias), eps)
    xp = get_backend("cupy" if type(x).__module__.startswith("cupy") else "numpy").xp
    mean = xp.mean(x, axis=-1, keepdims=True)
    var = xp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / xp.sqrt(var + eps) * weight + bias


def gelu(x: Any) -> Any:
    if not type(x).__module__.startswith("cupy"):
        return _gelu_kernel(np.asarray(x))
    xp = get_backend("cupy" if type(x).__module__.startswith("cupy") else "numpy").xp
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + xp.tanh(c * (x + 0.044715 * x**3)))


def rope(x: Any, cos: Any, sin: Any) -> Any:
    """Apply rotary position embedding to (..., T, D) arrays."""
    if not type(x).__module__.startswith("cupy"):
        return _rope_kernel(np.asarray(x), np.squeeze(np.asarray(cos)), np.squeeze(np.asarray(sin)))
    xp = get_backend("cupy").xp
    x = xp.asarray(x)
    out = xp.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return out


def causal_self_attention(q: Any, k: Any, v: Any, offset: int = 0) -> Any:
    """Scaled causal attention for (B, H, T, D) inputs."""
    if not type(q).__module__.startswith("cupy"):
        return _attention_kernel(np.asarray(q), np.asarray(k), np.asarray(v), offset)
    xp = get_backend("cupy").xp
    scores = xp.matmul(q, xp.swapaxes(k, -1, -2)) / np.sqrt(q.shape[-1])
    qi = xp.arange(q.shape[-2])[:, None] + offset
    ki = xp.arange(k.shape[-2])[None, :]
    scores = xp.where(ki > qi, -1e9, scores)
    scores = scores - xp.max(scores, axis=-1, keepdims=True)
    probs = xp.exp(scores)
    probs /= xp.sum(probs, axis=-1, keepdims=True)
    return xp.matmul(probs, v)


def gradcheck(function: Any, x: np.ndarray, analytic: np.ndarray, eps: float = 1e-6,
              atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    """Compare supplied analytic gradient to central finite differences.

    ``function`` must return a scalar for an ndarray argument.
    """
    x = np.asarray(x, dtype=np.float64)
    numeric = np.empty_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        numeric[index] = (float(function(plus)) - float(function(minus))) / (2 * eps)
    return bool(np.allclose(numeric, analytic, atol=atol, rtol=rtol))
