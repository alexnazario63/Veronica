"""Pacote core.inference da Veronica."""

from core.inference.generator import GenerationConfig, AutoregressiveGenerator, PrefixKVCache
from core.inference.constrained_json import ConstrainedJSONDecoder

__all__ = [
    "GenerationConfig",
    "AutoregressiveGenerator",
    "PrefixKVCache",
    "ConstrainedJSONDecoder",
]
