""" Pacote core.neural — Fundação Matemática da Veronica. """

from core.neural.tensor import Tensor
from core.neural.layers import (
    Parameter,
    Module,
    Neuron,
    Linear,
    Embedding,
    LayerNorm,
    RMSNorm,
    Dropout,
    Sequential,
)
from core.neural.activations import (
    relu,
    gelu,
    sigmoid,
    tanh,
    softmax,
    log_softmax,
    ReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
)
from core.neural.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from core.neural.optimizers import Optimizer, SGD, AdamW, CosineWarmupScheduler
from core.neural.init import (
    zeros_,
    ones_,
    normal_,
    uniform_,
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
)
from core.neural.clipping import (
    compute_grad_norm,
    has_non_finite_gradients,
    clip_grad_norm_,
    clip_grad_value_,
    per_parameter_grad_norms,
)
from core.neural.checkpoint import save_checkpoint, load_checkpoint
from core.neural.gradcheck import numerical_gradient, gradcheck
from core.neural.scalability import (
    MixedPrecisionScaler,
    activation_checkpoint,
    DistributedTrainingStub,
)

__all__ = [
    "Tensor",
    "Parameter",
    "Module",
    "Neuron",
    "Linear",
    "Embedding",
    "LayerNorm",
    "RMSNorm",
    "Dropout",
    "Sequential",
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "ReLU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "MSELoss",
    "CrossEntropyLoss",
    "BinaryCrossEntropyLoss",
    "Optimizer",
    "SGD",
    "AdamW",
    "CosineWarmupScheduler",
    "zeros_",
    "ones_",
    "normal_",
    "uniform_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "compute_grad_norm",
    "has_non_finite_gradients",
    "clip_grad_norm_",
    "clip_grad_value_",
    "per_parameter_grad_norms",
    "save_checkpoint",
    "load_checkpoint",
    "numerical_gradient",
    "gradcheck",
    "MixedPrecisionScaler",
    "activation_checkpoint",
    "DistributedTrainingStub",
]
