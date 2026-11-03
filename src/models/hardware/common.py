"""Abstract classes for hardware models and the collectors that observe their abstract execution."""

from abc import ABC, abstractmethod
from src.models.noise.noise import NoiseModel

import stim

class ExecutionProxyMetrics:
    """Class for storing proxy information about a single execution of a circuit on a abstract hardware model."""

    hardware_events: dict[str, int]

class HardwareModel(ABC):
    """Class for storing information about a hardware model."""

    @abstractmethod
    def abstract_execution(self, circuit: stim.Circuit, noise_model: NoiseModel) -> tuple[stim.Circuit, ExecutionProxyMetrics]:
        """Executes a circuit abstractly on the hardware model, constructing a noisy circuit and collecting data about the execution proxy.
        """
