"""Pacote core.training da Veronica."""

from core.training.trainer import (
    DEFAULT_NETWORK_TRAINING_CORPUS,
    TrainingHyperparameters,
    StepMetricRecord,
    get_real_hardware_telemetry,
    VeronicaTrainer,
)
from core.training.continuous_learning import (
    ContinuousLearningPipeline,
    ValidatedIncidentRecord,
    HumanCorrectionRecord,
)

__all__ = [
    "DEFAULT_NETWORK_TRAINING_CORPUS",
    "TrainingHyperparameters",
    "StepMetricRecord",
    "get_real_hardware_telemetry",
    "VeronicaTrainer",
    "ContinuousLearningPipeline",
    "ValidatedIncidentRecord",
    "HumanCorrectionRecord",
]
