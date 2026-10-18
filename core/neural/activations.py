"""
Funções de ativação da Veronica: ReLU, GELU, Sigmoid, Tanh, Softmax e LogSoftmax.
"""

from __future__ import annotations

from core.neural.tensor import Tensor


def relu(x: Tensor) -> Tensor:
    return x.relu()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.softmax(axis=axis)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return x.log_softmax(axis=axis)


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.gelu()


class Sigmoid:
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh:
    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax:
    def __init__(self, axis: int = -1) -> None:
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)
