"""Syndrome extraction circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

class SyndromeExtractionAnalyzer(MetricExtractor):
    def __init__(self, hardware: HardwareModel, stabilizers: list[str], logical_operators: list[str]):
        self.hw = hardware
        self.stabilizers = stabilizers
        self.logical_operators = logical_operators

    def analyze(self) -> dict[str, object]:
        return {}