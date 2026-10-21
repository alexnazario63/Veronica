"""
Escalabilidade e Eficiência Computacional do Motor Matemático da Veronica (Seção 3.3):
- MixedPrecisionScaler: precisão mista (fp16 / fp32 em forward pass com pesos mestre e
  acumulação de gradientes em fp64) e Dynamic Loss Scaling para prevenir underflow/overflow.
- activation_checkpoint: Checkpointing de ativações (descarta ativações intermediárias no
  forward pass e recalcula o subgrafo analítico no passo backward() para economizar RAM/VRAM).
- DistributedTrainingStub: interface estruturada preparada para treinamento distribuído
  (all_reduce_gradients e broadcast de parâmetros sobre múltiplos workers/processos).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Set
import numpy as np

from core.neural.tensor import Tensor
from core.neural.layers import Parameter
from core.neural.optimizers import Optimizer


def _resolve_forward_dtype(precision_mode: str) -> Any:
    mode = precision_mode.lower().strip()
    if mode in ("fp16", "float16", "mixed_fp16"):
        return np.float16
    if mode in ("fp32", "float32", "mixed_fp32"):
        return np.float32
    if mode in ("fp64", "float64"):
        return np.float64
    raise ValueError(
        f"Modo de precisão inválido '{precision_mode}'. Use 'fp64', 'fp32' ou 'mixed_fp16'."
    )


class MixedPrecisionScaler:
    """
    Gerenciador de Precisão Mista (fp16/fp32 em forward com acumulação fp64)
    e Dynamic Loss Scaling para evitar underflow numérico de gradientes.
    """

    def __init__(
        self,
        init_scale: float = 1024.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 100,
        precision_mode: str = "mixed_fp16",
        enabled: bool = True,
    ) -> None:
        if init_scale <= 0.0:
            raise ValueError("init_scale deve ser estritamente positivo.")
        if growth_factor <= 1.0:
            raise ValueError("growth_factor deve ser maior que 1.0.")
        if not (0.0 < backoff_factor < 1.0):
            raise ValueError("backoff_factor deve estar entre 0.0 e 1.0.")

        self.precision_mode: str = precision_mode.lower().strip()
        self.forward_dtype: Any = _resolve_forward_dtype(self.precision_mode)
        self.enabled: bool = bool(enabled) and (self.forward_dtype != np.float64)
        self.scale: float = float(init_scale) if self.enabled else 1.0
        self.growth_factor: float = float(growth_factor)
        self.backoff_factor: float = float(backoff_factor)
        self.growth_interval: int = max(int(growth_interval), 1)

        self._growth_tracker: int = 0
        self.skipped_steps: int = 0
        self.total_steps: int = 0
        self._unscaled_in_step: bool = False
        self.master_weights: Dict[int, np.ndarray] = {}

    def cast_forward(self, x: Tensor) -> Tensor:
        """
        Quantiza as ativações de entrada no passo forward para `forward_dtype` (fp16/fp32),
        preservando a retropropagação exata em `fp64` (Straight-Through Cast).
        """
        if not self.enabled or self.forward_dtype == np.float64:
            return x

        quantized_data = x.data.astype(self.forward_dtype).astype(np.float64)
        out = Tensor(
            quantized_data,
            requires_grad=x.requires_grad,
            _children=(x,) if x.requires_grad else (),
            _op=f"autocast({self.precision_mode})",
            dtype=np.float64,
        )

        def _backward() -> None:
            if out.grad is not None and x.requires_grad:
                x._ensure_grad()
                x.grad += out.grad.astype(np.float64, copy=False)

        out._backward = _backward
        return out

    def prepare_master_weights(self, params: Iterable[Parameter]) -> None:
        """
        Salva cópia mestre em `fp64` dos parâmetros e quantiza `p.data` para a precisão
        reduzida (`fp16` ou `fp32`) durante o passo forward.
        """
        if not self.enabled or self.forward_dtype == np.float64:
            return
        for p in params:
            pid = id(p)
            self.master_weights[pid] = p.data.astype(np.float64, copy=True)
            p.data = p.data.astype(self.forward_dtype).astype(np.float64)

    def restore_master_weights(self, params: Iterable[Parameter]) -> None:
        """Restaura os pesos mestre em `fp64` antes da atualização do otimizador."""
        if not self.enabled or not self.master_weights:
            return
        for p in params:
            pid = id(p)
            if pid in self.master_weights:
                p.data = self.master_weights[pid].copy()

    @contextmanager
    def autocast(self, params: Optional[Iterable[Parameter]] = None) -> Iterator[MixedPrecisionScaler]:
        """
        Context manager que executa o forward pass com pesos quantizados em `fp16`/`fp32`
        e restaura automaticamente os pesos mestre `fp64` para o passo de otimização.
        """
        param_list = list(params) if params is not None else []
        if param_list:
            self.prepare_master_weights(param_list)
        try:
            yield self
        finally:
            if param_list:
                self.restore_master_weights(param_list)

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Escala o tensor de perda por `self.scale` para evitar underflow no backward."""
        self._unscaled_in_step = False
        if not self.enabled or self.scale == 1.0:
            return loss
        return loss * self.scale

    def unscale_(self, params: Iterable[Parameter]) -> bool:
        """
        Remove o fator de escala dos gradientes em `fp64` in-place e verifica finitude.
        Retorna `True` se todos os gradientes forem finitos, ou `False` se houver NaN/Inf.
        """
        param_list = list(params)
        inv_scale = (1.0 / self.scale) if (self.enabled and self.scale != 0.0) else 1.0
        all_finite = True

        for p in param_list:
            if not p.requires_grad or p.grad is None:
                continue
            p.grad = p.grad.astype(np.float64, copy=False) * inv_scale
            if not np.all(np.isfinite(p.grad)):
                all_finite = False

        self._unscaled_in_step = True
        return all_finite

    def step(
        self,
        optimizer: Optimizer,
        params: Optional[Iterable[Parameter]] = None,
    ) -> bool:
        """
        Desescala os gradientes (se necessário), restaura os pesos mestre em `fp64`
        e executa `optimizer.step()` caso não haja overflow/NaN. Ajusta `self.scale`
        dinamicamente (dynamic loss scaling).
        """
        target_params = list(params) if params is not None else list(optimizer.params)
        self.restore_master_weights(target_params)

        if not self._unscaled_in_step:
            is_finite = self.unscale_(target_params)
        else:
            is_finite = all(
                np.all(np.isfinite(p.grad))
                for p in target_params
                if p.requires_grad and p.grad is not None
            )

        self.total_steps += 1
        self._unscaled_in_step = False

        if not is_finite:
            self.skipped_steps += 1
            if self.enabled:
                self.scale = max(1.0, self.scale * self.backoff_factor)
            self._growth_tracker = 0
            optimizer.zero_grad()
            return False

        optimizer.step()
        for p in target_params:
            self.master_weights[id(p)] = p.data.astype(np.float64, copy=True)

        if self.enabled:
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, 2.0 ** 24)
                self._growth_tracker = 0

        return True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "precision_mode": self.precision_mode,
            "enabled": self.enabled,
            "scale": self.scale,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "growth_tracker": self._growth_tracker,
            "skipped_steps": self.skipped_steps,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.precision_mode = str(state.get("precision_mode", self.precision_mode))
        self.forward_dtype = _resolve_forward_dtype(self.precision_mode)
        self.enabled = bool(state.get("enabled", self.enabled))
        self.scale = float(state.get("scale", self.scale))
        self.growth_factor = float(state.get("growth_factor", self.growth_factor))
        self.backoff_factor = float(state.get("backoff_factor", self.backoff_factor))
        self.growth_interval = int(state.get("growth_interval", self.growth_interval))
        self._growth_tracker = int(state.get("growth_tracker", self._growth_tracker))
        self.skipped_steps = int(state.get("skipped_steps", self.skipped_steps))
        self.total_steps = int(state.get("total_steps", self.total_steps))


def _release_subgraph(root: Tensor, stop_at: Optional[Tensor] = None) -> int:
    """
    Descarta referências de nós intermediários do subgrafo computacional para
    liberar memória RAM/VRAM imediatamente após o passo forward.
    """
    visited: Set[Tensor] = set()
    freed_nodes = 0

    def _walk(node: Tensor) -> None:
        nonlocal freed_nodes
        if node in visited or node is stop_at:
            return
        visited.add(node)
        if not node._prev:
            return
        children = list(node._prev)
        node._prev.clear()
        node._backward = lambda: None
        freed_nodes += 1
        for child in children:
            _walk(child)

    _walk(root)
    return freed_nodes


def activation_checkpoint(
    module_fn: Callable[..., Any],
    x: Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Checkpointing de ativações (Seção 3.3):
    Executa `module_fn(x, *args, **kwargs)` descartando todas as ativações intermediárias
    do subgrafo durante o passo `forward`, e recalcula o subgrafo sob demanda durante
    o passo `backward()` para obter gradientes analíticos idênticos com menor uso de memória.

    Suporta funções/módulos que retornam um `Tensor` ou uma tupla `(Tensor, ...)`.
    """
    x_probe = Tensor(
        x.data.copy(),
        requires_grad=True,
        dtype=x.dtype,
    )
    raw_out = module_fn(x_probe, *args, **kwargs)

    is_tuple = isinstance(raw_out, tuple)
    out_tensor = raw_out[0] if is_tuple else raw_out
    if not isinstance(out_tensor, Tensor):
        raise TypeError(
            "module_fn em activation_checkpoint deve retornar um Tensor ou uma tupla (Tensor, ...)."
        )

    out_data = out_tensor.data.copy()
    out_dtype = out_tensor.dtype
    # Descarta as ativações intermediárias criadas no forward
    freed_nodes = _release_subgraph(out_tensor, stop_at=x_probe)

    ckpt_tensor = Tensor(
        out_data,
        requires_grad=True,
        _children=(x,),
        _op=f"activation_checkpoint(freed={freed_nodes})",
        dtype=out_dtype,
    )

    def _backward() -> None:
        if ckpt_tensor.grad is None:
            return
        x_recomp = Tensor(
            x.data.copy(),
            requires_grad=x.requires_grad,
            dtype=x.dtype,
        )
        recomp_raw = module_fn(x_recomp, *args, **kwargs)
        recomp_tensor = recomp_raw[0] if isinstance(recomp_raw, tuple) else recomp_raw
        if not isinstance(recomp_tensor, Tensor):
            raise TypeError(
                "module_fn em activation_checkpoint deve retornar um Tensor ou uma tupla (Tensor, ...)."
            )
        if recomp_tensor.requires_grad:
            recomp_tensor.backward(ckpt_tensor.grad)
        if x.requires_grad and x_recomp.grad is not None:
            x._ensure_grad()
            x.grad += x_recomp.grad
        _release_subgraph(recomp_tensor, stop_at=x_recomp)

    ckpt_tensor._backward = _backward

    if is_tuple:
        return (ckpt_tensor, *raw_out[1:])
    return ckpt_tensor


class DistributedTrainingStub:
    """
    Interface estruturada preparada para treinamento distribuído futuro (Data Parallelism),
    suportando `all_reduce_gradients` e `broadcast_parameters` sobre múltiplos workers/processos.
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        backend: str = "local_shm",
    ) -> None:
        if world_size < 1:
            raise ValueError("world_size deve ser >= 1.")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank ({rank}) deve estar em [0, {world_size - 1}].")

        self.world_size: int = int(world_size)
        self.rank: int = int(rank)
        self.backend: str = backend
        self.is_initialized: bool = True
        self.sync_steps: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def all_reduce_gradients(
        self,
        params: Iterable[Parameter],
        op: str = "mean",
        worker_grads: Optional[Sequence[Sequence[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """
        Reduz e sincroniza gradientes entre workers (`mean` ou `sum`).
        Quando `worker_grads` é fornecido (simulação local multi-worker), combina os gradientes
        de todos os workers analiticamente em `fp64`.
        """
        op_norm = op.lower().strip()
        if op_norm not in ("mean", "sum"):
            raise ValueError(f"Operação all_reduce inválida '{op}'. Use 'mean' ou 'sum'.")

        param_list = list(params)
        synced_count = 0

        if worker_grads is not None and len(worker_grads) > 0:
            num_workers = 1 + len(worker_grads)
            for idx, p in enumerate(param_list):
                if not p.requires_grad or p.grad is None:
                    continue
                acc = p.grad.astype(np.float64, copy=True)
                for wg in worker_grads:
                    if idx < len(wg) and wg[idx] is not None:
                        acc += np.asarray(wg[idx], dtype=np.float64)
                if op_norm == "mean":
                    acc /= float(num_workers)
                p.grad = acc
                synced_count += 1
        else:
            for p in param_list:
                if not p.requires_grad or p.grad is None:
                    continue
                if op_norm == "mean" and self.world_size > 1:
                    p.grad = p.grad.astype(np.float64, copy=False) / float(self.world_size)
                synced_count += 1

        self.sync_steps += 1
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "backend": self.backend,
            "op": op_norm,
            "synced_parameters": synced_count,
            "sync_steps": self.sync_steps,
        }

    def broadcast_parameters(
        self,
        params: Iterable[Parameter],
        src_rank: int = 0,
        source_weights: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        """Sincroniza os pesos mestre a partir do worker `src_rank`."""
        if not (0 <= src_rank < self.world_size):
            raise ValueError(f"src_rank inválido: {src_rank}")
        if source_weights is not None:
            for p, w in zip(params, source_weights):
                p.data[...] = np.asarray(w, dtype=np.float64)

    def barrier(self) -> None:
        """Barreira de sincronização entre processos."""
        return None

