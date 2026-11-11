"""Syndrome extraction circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

class SyndromeExtractionAnalyzer(MetricExtractor):
    def __init__(self, hardware: HardwareModel):
        self.hw = hardware

    def analyze(self) -> dict[str, object]:
        {}