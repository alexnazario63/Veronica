"""
Funções de Perda (Losses) da Veronica com estabilidade numérica e derivadas analíticas exatas.
Inclui MSELoss, CrossEntropyLoss e BinaryCrossEntropyLoss.
"""

from __future__ import annotations

import numpy as np
from core.neural.tensor import Tensor


class MSELoss:
    """Erro Quadrático Médio: L = mean((pred - target)^2)."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        sq = diff ** 2
        if self.reduction == "mean":
            return sq.mean()
        if self.reduction == "sum":
            return sq.sum()
        return sq


class CrossEntropyLoss:
    """
    Entropia Cruzada para classificação e previsão do próximo token (Language Modeling).
    Aceita logits de shape (N, C) ou (B, T, C) e targets inteiros de shape (N,) ou (B, T),
    com suporte a ignore_index (ex.: padding).
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean") -> None:
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        orig_shape = logits.shape
        num_classes = orig_shape[-1]
        logits_2d = logits.data.reshape(-1, num_classes)
        targets_1d = np.asarray(targets, dtype=np.int64).reshape(-1)

        if logits_2d.shape[0] != targets_1d.shape[0]:
            raise ValueError(
                f"CrossEntropyLoss: número de amostras nos logits ({logits_2d.shape[0]}) "
                f"difere dos targets ({targets_1d.shape[0]})."
            )

        valid_mask = targets_1d != self.ignore_index
        valid_count = int(np.sum(valid_mask))

        # Log-Sum-Exp numericamente estável
        max_logits = np.max(logits_2d, axis=-1, keepdims=True)
        shifted = logits_2d - max_logits
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        log_probs = shifted - np.log(sum_exp)
        probs = exp_shifted / sum_exp

        safe_targets = np.where(valid_mask, targets_1d, 0)
        nll = -log_probs[np.arange(logits_2d.shape[0]), safe_targets]
        nll = np.where(valid_mask, nll, 0.0)

        if self.reduction == "mean":
            loss_val = float(np.sum(nll) / max(valid_count, 1))
        elif self.reduction == "sum":
            loss_val = float(np.sum(nll))
        else:
            loss_val = nll.reshape(orig_shape[:-1])

        out = Tensor(
            loss_val,
            requires_grad=logits.requires_grad,
            _children=(logits,) if logits.requires_grad else (),
            _op="cross_entropy",
        )

        def _backward() -> None:
            if out.grad is None or not logits.requires_grad:
                return
            logits._ensure_grad()
            grad_2d = probs.copy()
            grad_2d[np.arange(logits_2d.shape[0]), safe_targets] -= 1.0
            grad_2d[~valid_mask] = 0.0

            if self.reduction == "mean":
                grad_2d /= max(valid_count, 1)
                grad_2d *= float(out.grad.item())
            elif self.reduction == "sum":
                grad_2d *= float(out.grad.item())
            else:
                grad_2d *= out.grad.reshape(-1, 1)

            logits.grad += grad_2d.reshape(orig_shape)

        out._backward = _backward
        return out


class BinaryCrossEntropyLoss:
    """Entropia Cruzada Binária sobre probabilidades em (0, 1)."""

    def __init__(self, eps: float = 1e-12, reduction: str = "mean") -> None:
        self.eps = eps
        self.reduction = reduction

    def __call__(self, probs: Tensor, targets: Tensor) -> Tensor:
        p = np.clip(probs.data, self.eps, 1.0 - self.eps)
        y = targets.data
        bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        if self.reduction == "mean":
            loss_val = float(np.mean(bce))
        elif self.reduction == "sum":
            loss_val = float(np.sum(bce))
        else:
            loss_val = bce

        out = Tensor(
            loss_val,
            requires_grad=probs.requires_grad,
            _children=(probs,) if probs.requires_grad else (),
            _op="binary_cross_entropy",
        )

        def _backward() -> None:
            if out.grad is None or not probs.requires_grad:
                return
            probs._ensure_grad()
            grad = -(y / p) + ((1.0 - y) / (1.0 - p))
            if self.reduction == "mean":
                grad = grad * (float(out.grad.item()) / max(probs.size, 1))
            elif self.reduction == "sum":
                grad = grad * float(out.grad.item())
            else:
                grad = grad * out.grad
            probs.grad += grad

        out._backward = _backward
        return out
