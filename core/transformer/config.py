"""
Configuração parametrizada da arquitetura VeronicaTransformer.
Permite escalar de modelos micro (para testes/inspeção individual de neurônios)
a modelos maiores conforme recursos disponíveis de CPU/GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class VeronicaConfig:
    vocab_size: int = 512
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 192
    max_seq_len: int = 256
    norm_type: str = "rmsnorm"        # "rmsnorm" ou "layernorm"
    positional_type: str = "rope"     # "rope", "sinusoidal" ou "learned"
    activation: str = "gelu"          # "gelu" ou "relu"
    dropout: float = 0.0
    seed: int = 42
    use_activation_checkpointing: bool = False
    precision_mode: str = "fp64"      # "fp64", "fp32" ou "mixed_fp16"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) deve ser divisível por n_heads ({self.n_heads})."
            )
        if self.head_dim % 2 != 0 and self.positional_type == "rope":
            raise ValueError(
                f"Para RoPE, head_dim ({self.head_dim}) deve ser par."
            )
        if self.precision_mode not in ("fp64", "fp32", "mixed_fp16"):
            raise ValueError(
                f"precision_mode inválido ({self.precision_mode}). Use 'fp64', 'fp32' ou 'mixed_fp16'."
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
