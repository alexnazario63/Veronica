"""
Inicializadores matemáticos de parâmetros neurais da Veronica.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
from core.neural.tensor import Tensor


def _calculate_fan_in_and_fan_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) < 2:
        fan_in = shape[0] if len(shape) == 1 else 1
        fan_out = shape[0] if len(shape) == 1 else 1
        return fan_in, fan_out
    num_input_fmaps = shape[-2]
    num_output_fmaps = shape[-1]
    receptive_field_size = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    return num_input_fmaps * receptive_field_size, num_output_fmaps * receptive_field_size


def zeros_(tensor: Tensor) -> Tensor:
    tensor.data.fill(0.0)
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    tensor.data.fill(1.0)
    return tensor


def normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    gen = rng or np.random.default_rng()
    tensor.data[...] = gen.normal(loc=mean, scale=std, size=tensor.shape)
    return tensor


def uniform_(
    tensor: Tensor,
    low: float = -0.1,
    high: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    gen = rng or np.random.default_rng()
    tensor.data[...] = gen.uniform(low=low, high=high, size=tensor.shape)
    return tensor


def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor.shape)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound, rng=rng)


def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor.shape)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std, rng=rng)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor.shape)
    gain = math.sqrt(2.0 / (1.0 + a ** 2))
    std = gain / math.sqrt(float(fan_in))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound, rng=rng)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tensor:
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor.shape)
    gain = math.sqrt(2.0 / (1.0 + a ** 2))
    std = gain / math.sqrt(float(fan_in))
    return normal_(tensor, 0.0, std, rng=rng)
