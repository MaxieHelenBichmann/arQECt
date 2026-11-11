"""Metric extractor for the weighted interaction graph of a circuit. It exists to keep the InteractionGraph as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.structure.interaction_graph import InteractionGraph

class InteractionGraphAnalyzer(MetricExtractor):
    def __init__(self, wig: InteractionGraph):
        self.wig = wig

    def analyze(self) -> dict:
        {}