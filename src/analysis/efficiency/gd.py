"""Metric extractor for the gate distribution of a circuit. It exists to keep the GateDistribution as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel
    from src.structure.gate_distribution import GateDistribution

class GateDistributionAnalyzer(MetricExtractor):
    def __init__(self, gd: GateDistribution,  hardware: HardwareModel):
        self.gd = gd
        self.hw = hardware

    def analyze(self) -> dict[str, object]:
        return {}