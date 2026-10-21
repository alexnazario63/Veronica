"""
Neurônios Artificiais Parametrizados e Camadas Neurais da Veronica.
Inclui Parameter, Module, Neuron, Linear, Embedding, LayerNorm, RMSNorm, Dropout e Sequential.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from core.neural.tensor import Tensor
from core.neural.init import xavier_uniform_, normal_


class Parameter(Tensor):
    """Parâmetro treinável de um módulo neural."""

    def __init__(
        self,
        data: Any,
        requires_grad: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(data, requires_grad=requires_grad, name=name)


class Module:
    """Classe base para todos os componentes neurais da Veronica."""

    def __init__(self) -> None:
        self.training: bool = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> Dict[str, Parameter]:
        params: Dict[str, Parameter] = {}
        for attr_name, val in self.__dict__.items():
            full_name = f"{prefix}.{attr_name}" if prefix else attr_name
            if isinstance(val, Parameter):
                params[full_name] = val
            elif isinstance(val, Module):
                params.update(val.named_parameters(prefix=full_name))
            elif isinstance(val, (list, tuple)):
                for idx, item in enumerate(val):
                    item_name = f"{full_name}.{idx}"
                    if isinstance(item, Parameter):
                        params[item_name] = item
                    elif isinstance(item, Module):
                        params.update(item.named_parameters(prefix=item_name))
        return params

    def parameters(self) -> List[Parameter]:
        return list(self.named_parameters().values())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for val in self.__dict__.values():
            if isinstance(val, Module):
                val.train(mode)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def parameter_counts(self) -> Dict[str, int]:
        total = 0
        trainable = 0
        for p in self.parameters():
            total += p.size
            if p.requires_grad:
                trainable += p.size
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
        }

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {name: p.data.copy() for name, p in self.named_parameters().items()}

    def load_state_dict(self, state: Dict[str, np.ndarray], strict: bool = True) -> None:
        named = self.named_parameters()
        if strict:
            missing = {
                k for k in (set(named.keys()) - set(state.keys()))
                if not k.startswith("neurogenesis_circuit")
            }
            unexpected = set(state.keys()) - set(named.keys())
            if missing or unexpected:
                raise KeyError(
                    f"Incompatibilidade em load_state_dict. Faltando: {missing}, Inesperados: {unexpected}"
                )
        for name, param in named.items():
            if name in state:
                if param.shape != state[name].shape:
                    raise ValueError(
                        f"Dimensão incompatível para '{name}': esperado {param.shape}, recebido {state[name].shape}"
                    )
                param.data[...] = state[name]


class Neuron(Module):
    """
    Neurônio artificial individual parametrizado:
    z = x @ w + b, a = activation(z)
    Útil tanto para cálculos fundamentais quanto para inspeção micro no Neural Observatory.
    """

    def __init__(
        self,
        in_features: int,
        activation: str = "relu",
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.activation_name = activation.lower()
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(max(in_features, 1))
        self.weight = Parameter(
            rng.uniform(-bound, bound, size=(in_features, 1)),
            name="weight",
        )
        self.bias: Optional[Parameter] = (
            Parameter(np.zeros((1,), dtype=np.float64), name="bias") if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        z = x @ self.weight
        if self.bias is not None:
            z = z + self.bias
        if self.activation_name == "relu":
            return z.relu()
        if self.activation_name == "gelu":
            return z.gelu()
        if self.activation_name == "sigmoid":
            return z.sigmoid()
        if self.activation_name == "tanh":
            return z.tanh()
        if self.activation_name in ("linear", "none", "identity"):
            return z
        raise ValueError(f"Ativação desconhecida no Neurônio: {self.activation_name}")


class Linear(Module):
    """
    Camada densa (afim) parametrizada: y = x @ W + b
    W possui shape (in_features, out_features) e b possui shape (out_features,).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng(seed)
        self.weight = Parameter(
            np.empty((in_features, out_features), dtype=np.float64),
            name="weight",
        )
        xavier_uniform_(self.weight, rng=rng)
        if bias:
            self.bias: Optional[Parameter] = Parameter(
                np.zeros((out_features,), dtype=np.float64),
                name="bias",
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """
    Tabela de Embeddings treináveis: converte IDs discretos de tokens em vetores densos.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        rng = np.random.default_rng(seed)
        self.weight = Parameter(
            np.empty((num_embeddings, embedding_dim), dtype=np.float64),
            name="weight",
        )
        normal_(self.weight, mean=0.0, std=0.02, rng=rng)

    def forward(self, input_ids: np.ndarray) -> Tensor:
        ids = np.asarray(input_ids, dtype=np.int64)
        out_data = self.weight.data[ids]
        out = Tensor(
            out_data,
            requires_grad=self.weight.requires_grad,
            _children=(self.weight,) if self.weight.requires_grad else (),
            _op="embedding",
        )

        def _backward() -> None:
            if out.grad is not None and self.weight.requires_grad:
                self.weight._ensure_grad()
                np.add.at(self.weight.grad, ids, out.grad)

        out._backward = _backward
        return out


class LayerNorm(Module):
    """
    Layer Normalization sobre a última dimensão:
    y = ((x - E[x]) / sqrt(Var[x] + eps)) * gamma + beta
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Parameter(np.ones((normalized_shape,), dtype=np.float64), name="gamma")
        self.beta = Parameter(np.zeros((normalized_shape,), dtype=np.float64), name="beta")

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered ** 2).mean(axis=-1, keepdims=True)
        std = (var + self.eps).sqrt()
        normed = centered / std
        return normed * self.gamma + self.beta


class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization:
    y = (x / sqrt(mean(x^2) + eps)) * weight
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Parameter(np.ones((normalized_shape,), dtype=np.float64), name="weight")

    def forward(self, x: Tensor) -> Tensor:
        ms = (x ** 2).mean(axis=-1, keepdims=True)
        rms = (ms + self.eps).sqrt()
        return (x / rms) * self.weight


class Dropout(Module):
    """Regularização por Dropout invertido durante o treinamento."""

    def __init__(self, p: float = 0.1, seed: Optional[int] = None) -> None:
        super().__init__()
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (self.rng.random(x.shape) < keep_prob).astype(np.float64) / keep_prob
        return x * mask


class Sequential(Module):
    """Contêiner sequencial de camadas e funções."""

    def __init__(self, *layers: Any) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
