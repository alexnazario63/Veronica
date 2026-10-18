"""
Gradient Clipping e inspeção de normas/finitude de gradientes da Veronica.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List
import numpy as np
from core.neural.layers import Parameter


def compute_grad_norm(params: Iterable[Parameter]) -> float:
    total_norm_sq = 0.0
    for p in params:
        if p.requires_grad and p.grad is not None:
            total_norm_sq += float(np.sum(p.grad ** 2))
    return math.sqrt(total_norm_sq)


def has_non_finite_gradients(params: Iterable[Parameter]) -> bool:
    for p in params:
        if p.requires_grad and p.grad is not None:
            if not np.all(np.isfinite(p.grad)):
                return True
    return False


def clip_grad_norm_(params: Iterable[Parameter], max_norm: float, eps: float = 1e-6) -> float:
    """
    Limita a norma L2 global dos gradientes para evitar explosão de gradientes.
    Retorna a norma total antes do corte.
    """
    param_list: List[Parameter] = [p for p in params if p.requires_grad and p.grad is not None]
    total_norm = compute_grad_norm(param_list)
    if max_norm > 0.0 and total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in param_list:
            if p.grad is not None:
                p.grad *= clip_coef
    return total_norm


def clip_grad_value_(params: Iterable[Parameter], clip_value: float) -> None:
    """Limita cada componente individual de gradiente ao intervalo [-clip_value, +clip_value]."""
    bound = abs(float(clip_value))
    for p in params:
        if p.requires_grad and p.grad is not None:
            np.clip(p.grad, -bound, bound, out=p.grad)


def per_parameter_grad_norms(named_params: Dict[str, Parameter]) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    for name, p in named_params.items():
        if p.requires_grad and p.grad is not None:
            norms[name] = float(np.linalg.norm(p.grad))
        else:
            norms[name] = 0.0
    return norms
