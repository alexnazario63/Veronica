"""
Motor Matemático e de Diferenciação Automática (Autograd) da Veronica.
Implementado desde os fundamentos matemáticos sobre arrays NumPy,
com cálculo analítico exato de gradientes e suporte a tensores N-dimensionais.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

ArrayLike = Union[np.ndarray, float, int, Sequence[Any]]


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Reduz o gradiente de volta para `target_shape` desfazendo broadcasting."""
    if grad.shape == target_shape:
        return grad
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


class Tensor:
    """
    Tensor multidimensional com grafo computacional dinâmico e Backpropagation.
    """

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        name: Optional[str] = None,
        dtype: Any = np.float64,
    ) -> None:
        if isinstance(data, Tensor):
            arr = data.data.astype(dtype, copy=True)
        elif isinstance(data, np.ndarray):
            arr = data.astype(dtype, copy=False)
        else:
            arr = np.array(data, dtype=dtype)

        self.data: np.ndarray = arr
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[np.ndarray] = (
            np.zeros_like(self.data, dtype=np.float64) if self.requires_grad else None
        )
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Tensor"] = set(_children)
        self._op: str = _op
        self.name: Optional[str] = name

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False, name=self.name, dtype=self.dtype)

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64)
        else:
            self.grad = None

    def _ensure_grad(self) -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

    @staticmethod
    def _coerce(other: Union["Tensor", ArrayLike], dtype: Any = np.float64) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other, requires_grad=False, dtype=dtype)

    # ==================== Operações Algébricas Elementwise ====================

    def __add__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._coerce(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data + other_t.data,
            requires_grad=req_grad,
            _children=(self, other_t) if req_grad else (),
            _op="add",
            dtype=np.result_type(self.dtype, other_t.dtype),
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t._ensure_grad()
                other_t.grad += _unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="neg",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._coerce(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data - other_t.data,
            requires_grad=req_grad,
            _children=(self, other_t) if req_grad else (),
            _op="sub",
            dtype=np.result_type(self.dtype, other_t.dtype),
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t._ensure_grad()
                other_t.grad -= _unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._coerce(other, self.dtype).__sub__(self)

    def __mul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._coerce(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data * other_t.data,
            requires_grad=req_grad,
            _children=(self, other_t) if req_grad else (),
            _op="mul",
            dtype=np.result_type(self.dtype, other_t.dtype),
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad * other_t.data, self.shape)
            if other_t.requires_grad:
                other_t._ensure_grad()
                other_t.grad += _unbroadcast(out.grad * self.data, other_t.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._coerce(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data / other_t.data,
            requires_grad=req_grad,
            _children=(self, other_t) if req_grad else (),
            _op="div",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _unbroadcast(out.grad / other_t.data, self.shape)
            if other_t.requires_grad:
                other_t._ensure_grad()
                other_t.grad += _unbroadcast(
                    -out.grad * self.data / (other_t.data ** 2), other_t.shape
                )

        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self._coerce(other, self.dtype).__truediv__(self)

    def __pow__(self, exponent: float) -> "Tensor":
        out = Tensor(
            np.power(self.data, exponent),
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op=f"pow({exponent})",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (exponent * np.power(self.data, exponent - 1.0))

        out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        out = Tensor(
            np.sqrt(self.data),
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="sqrt",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (0.5 / np.maximum(out.data, 1e-12))

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        exp_val = np.exp(np.clip(self.data, -80.0, 80.0))
        out = Tensor(
            exp_val,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="exp",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * exp_val

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        safe_x = np.maximum(self.data, 1e-15)
        out = Tensor(
            np.log(safe_x),
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="log",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad / safe_x

        out._backward = _backward
        return out

    # ==================== Multiplicação Matricial & Forma ====================

    def matmul(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._coerce(other, self.dtype)
        req_grad = self.requires_grad or other_t.requires_grad
        out_data = np.matmul(self.data, other_t.data)
        out = Tensor(
            out_data,
            requires_grad=req_grad,
            _children=(self, other_t) if req_grad else (),
            _op="matmul",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            a = self.data
            b = other_t.data
            g = out.grad

            if a.ndim == 1 and b.ndim == 1:
                if self.requires_grad:
                    self._ensure_grad()
                    self.grad += g * b
                if other_t.requires_grad:
                    other_t._ensure_grad()
                    other_t.grad += g * a
            elif a.ndim == 1:
                a_2d = a[np.newaxis, :]
                g_2d = g[np.newaxis, :]
                if self.requires_grad:
                    self._ensure_grad()
                    da = np.matmul(g_2d, np.swapaxes(b, -1, -2)).squeeze(0)
                    self.grad += _unbroadcast(da, self.shape)
                if other_t.requires_grad:
                    other_t._ensure_grad()
                    db = np.matmul(np.swapaxes(a_2d, -1, -2), g_2d)
                    other_t.grad += _unbroadcast(db, other_t.shape)
            elif b.ndim == 1:
                b_2d = b[:, np.newaxis]
                g_2d = g[..., np.newaxis]
                if self.requires_grad:
                    self._ensure_grad()
                    da = np.matmul(g_2d, np.swapaxes(b_2d, -1, -2))
                    self.grad += _unbroadcast(da, self.shape)
                if other_t.requires_grad:
                    other_t._ensure_grad()
                    db = np.matmul(np.swapaxes(a, -1, -2), g_2d).squeeze(-1)
                    other_t.grad += _unbroadcast(db, other_t.shape)
            else:
                if self.requires_grad:
                    self._ensure_grad()
                    da = np.matmul(g, np.swapaxes(b, -1, -2))
                    self.grad += _unbroadcast(da, self.shape)
                if other_t.requires_grad:
                    other_t._ensure_grad()
                    db = np.matmul(np.swapaxes(a, -1, -2), g)
                    other_t.grad += _unbroadcast(db, other_t.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        return self.matmul(other)

    def transpose(self, axes: Optional[Sequence[int]] = None) -> "Tensor":
        if axes is None:
            if self.ndim < 2:
                return self
            axes_list = list(range(self.ndim))
            axes_list[-2], axes_list[-1] = axes_list[-1], axes_list[-2]
            axes = tuple(axes_list)
        else:
            axes = tuple(axes)

        out = Tensor(
            np.transpose(self.data, axes=axes),
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="transpose",
            dtype=self.dtype,
        )

        inv_axes = tuple(np.argsort(axes))

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += np.transpose(out.grad, axes=inv_axes)

        out._backward = _backward
        return out

    def swapaxes(self, axis1: int, axis2: int) -> "Tensor":
        axes = list(range(self.ndim))
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
        return self.transpose(axes)

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(int(s) for s in shape)

        orig_shape = self.shape
        out = Tensor(
            self.data.reshape(target_shape),
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="reshape",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(orig_shape)

        out._backward = _backward
        return out

    def __getitem__(self, idx: Any) -> "Tensor":
        out = Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="slice",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # ==================== Reduções ====================

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="sum",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self._ensure_grad()
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                for ax in sorted(axes):
                    g = np.expand_dims(g, axis=ax)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        if axis is None:
            count = self.size
        elif isinstance(axis, int):
            count = self.shape[axis]
        else:
            count = int(np.prod([self.shape[a] for a in axis]))

        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / max(count, 1))

    def masked_fill(self, mask: np.ndarray, value: float) -> "Tensor":
        bool_mask = np.asarray(mask, dtype=bool)
        out_data = np.where(bool_mask, value, self.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="masked_fill",
            dtype=self.dtype,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += np.where(bool_mask, 0.0, out.grad)

        out._backward = _backward
        return out

    # ==================== Funções de Ativação Analíticas ====================

    def relu(self) -> "Tensor":
        mask = self.data > 0.0
        out = Tensor(
            np.where(mask, self.data, 0.0),
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="relu",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * mask

        out._backward = _backward
        return out

    def gelu(self) -> "Tensor":
        """
        GELU com aproximação Tanh de alta precisão e derivada analítica exata:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        x = self.data
        k = math.sqrt(2.0 / math.pi)
        c = 0.044715
        u = k * (x + c * (x ** 3))
        t = np.tanh(u)
        out_data = 0.5 * x * (1.0 + t)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="gelu",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                du_dx = k * (1.0 + 3.0 * c * (x ** 2))
                dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du_dx
                self.grad += out.grad * dgelu_dx

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        clipped = np.clip(self.data, -80.0, 80.0)
        s = 1.0 / (1.0 + np.exp(-clipped))
        out = Tensor(
            s,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="sigmoid",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="tanh",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (1.0 - t ** 2)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(
            probs,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="softmax",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                dot = np.sum(out.grad * probs, axis=axis, keepdims=True)
                self.grad += probs * (out.grad - dot)

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        sum_exps = np.sum(exps, axis=axis, keepdims=True)
        log_probs = shifted - np.log(sum_exps)
        probs = exps / sum_exps
        out = Tensor(
            log_probs,
            requires_grad=self.requires_grad,
            _children=(self,) if self.requires_grad else (),
            _op="log_softmax",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and self.requires_grad:
                self._ensure_grad()
                sum_grad = np.sum(out.grad, axis=axis, keepdims=True)
                self.grad += out.grad - probs * sum_grad

        out._backward = _backward
        return out

    # ==================== Backpropagation ====================

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Executa a retropropagação (Backpropagation) em ordem topológica reversa.
        """
        if not self.requires_grad:
            raise RuntimeError("Chamada a backward() em um Tensor com requires_grad=False.")

        topo: List["Tensor"] = []
        visited: Set["Tensor"] = set()

        def build_topo(node: "Tensor") -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self._ensure_grad()
        if grad is None:
            if self.size != 1:
                raise RuntimeError(
                    "Gradiente inicial pode ser omitido apenas para saídas escalares (size == 1)."
                )
            self.grad = np.ones_like(self.data, dtype=np.float64)
        else:
            self.grad = np.asarray(grad, dtype=np.float64).reshape(self.shape)

        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        grad_flag = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        op_flag = f", grad_fn=<{self._op}>" if self._op else ""
        return f"Tensor(shape={self.shape}{grad_flag}{op_flag})"
