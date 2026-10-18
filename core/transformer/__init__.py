"""Pacote core.transformer da Veronica."""

from core.transformer.config import VeronicaConfig
from core.transformer.positional import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEmbedding,
)
from core.transformer.attention import (
    create_causal_mask,
    scaled_dot_product_attention,
    MultiHeadCausalAttention,
)
from core.transformer.model import (
    FeedForwardNetwork,
    TransformerBlock,
    VeronicaTransformer,
)

__all__ = [
    "VeronicaConfig",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "RotaryPositionalEmbedding",
    "create_causal_mask",
    "scaled_dot_product_attention",
    "MultiHeadCausalAttention",
    "FeedForwardNetwork",
    "TransformerBlock",
    "VeronicaTransformer",
]
