"""Encoder circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

class EncoderCircuitAnalyzer(MetricExtractor):
    def __init__(self, hardware: HardwareModel, stabilizers: list[stim.PauliString], logical_operators: list[stim.PauliString]):
        self.hw = hardware
        self.stabilizers = stabilizers
        self.logical_operators = logical_operators

    def analyze(self) -> dict[str, object]:
        return {}