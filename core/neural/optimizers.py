"""
Otimizadores matemáticos da Veronica: SGD (com momentum e weight decay) e AdamW
(com decaimento de peso desacoplado e correção de viés).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List
import numpy as np
from core.neural.layers import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-3) -> None:
        self.params: List[Parameter] = list(params)
        self.lr: float = float(lr)
        self.step_count: int = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {"lr": self.lr, "step_count": self.step_count}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.lr = float(state.get("lr", self.lr))
        self.step_count = int(state.get("step_count", self.step_count))


class SGD(Optimizer):
    """
    Stochastic Gradient Descent com Momentum e Weight Decay:
    v_t = mu * v_{t-1} + (g_t + wd * theta_t)
    theta_{t+1} = theta_t - lr * v_t
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.velocities: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.step_count += 1
        for idx, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.copy()
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data
            if self.momentum != 0.0:
                if idx not in self.velocities:
                    self.velocities[idx] = grad.copy()
                else:
                    self.velocities[idx] = self.momentum * self.velocities[idx] + grad
                update = self.velocities[idx]
            else:
                update = grad
            p.data -= self.lr * update


class AdamW(Optimizer):
    """
    AdamW (Adaptive Moment Estimation with Decoupled Weight Decay):
    theta_t = theta_{t-1} - lr * wd * theta_{t-1}
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ^ 2)
    m_hat = m_t / (1 - beta1 ^ t)
    v_hat = v_t / (1 - beta2 ^ t)
    theta_t = theta_t - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(params, lr=lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}

    def step(self) -> None:
        self.step_count += 1
        b1_corr = 1.0 - (self.beta1 ** self.step_count)
        b2_corr = 1.0 - (self.beta2 ** self.step_count)

        for idx, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue
            g = p.grad
            if idx not in self.m:
                self.m[idx] = np.zeros_like(p.data, dtype=np.float64)
                self.v[idx] = np.zeros_like(p.data, dtype=np.float64)

            # Decaimento de peso desacoplado (aplicado apenas a pesos >= 2D por convenção ou geral)
            if self.weight_decay != 0.0 and p.ndim >= 2:
                p.data -= self.lr * self.weight_decay * p.data

            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[idx] / b1_corr
            v_hat = self.v[idx] / b2_corr

            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "step_count": self.step_count,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": {str(k): v.copy() for k, v in self.m.items()},
            "v": {str(k): v.copy() for k, v in self.v.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.beta1 = float(state.get("beta1", self.beta1))
        self.beta2 = float(state.get("beta2", self.beta2))
        self.eps = float(state.get("eps", self.eps))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        if "m" in state:
            self.m = {int(k): np.array(v, dtype=np.float64) for k, v in state["m"].items()}
        if "v" in state:
            self.v = {int(k): np.array(v, dtype=np.float64) for k, v in state["v"].items()}


class CosineWarmupScheduler:
    """Agendador de taxa de aprendizado com aquecimento linear e decaimento em cosseno."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.total_steps = max(int(total_steps), 1)

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.max_lr * float(step + 1) / float(self.warmup_steps)
        if step >= self.total_steps:
            return self.min_lr
        decay_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = float(step - self.warmup_steps) / float(decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine

    def step(self, current_step: int) -> float:
        lr = self.get_lr(current_step)
        self.optimizer.lr = lr
        return lr
