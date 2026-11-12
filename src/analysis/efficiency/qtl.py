"""Metric extractor for the qubit timelines of a circuit. It exists to keep the QubitTimelines as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel
    from src.structure.qubit_timelines import QubitTimelines

class QubitTimelinesAnalyzer(MetricExtractor):
    def __init__(self, qtl: QubitTimelines,  hardware: HardwareModel):
        self.qtl = qtl
        self.hw = hardware

    def analyze(self) -> dict[str, object]:
        return {}