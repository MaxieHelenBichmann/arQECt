"""Metric extractor for the weighted interaction graph of a circuit. It exists to keep the InteractionGraph as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel
    from src.structure.interaction_graph import InteractionGraph

class InteractionGraphAnalyzer(MetricExtractor):
    def __init__(self, wig: InteractionGraph, hardware: HardwareModel):
        self.wig = wig
        self.hw = hardware

    def analyze(self) -> dict[str, object]:
        {}