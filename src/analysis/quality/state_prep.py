"""State preparation circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

from src.analysis import MetricExtractor
from src.circuits.circuit import QuantumState

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel



class StatePreparationCircuitAnalyzer(MetricExtractor):
    def __init__(self, hardware: HardwareModel, stabilizers: list[stim.PauliString], logical_operators: list[stim.PauliString], state: QuantumState | None = None):
        self.hw = hardware
        self.stabilizers = stabilizers
        self.logical_operators = logical_operators
        if state is None:
            raise AttributeError("A state preparation circuit needs to specify which logical state it prepares.")
        self.state = state

    def analyze(self) -> dict[str, object]:
        return {}