"""Analysis."""

from typing import Literal, Protocol

from src.analysis.processor import GateProcessor, CircuitProcessor

CircuitFunction = Literal["StatePrep", "Encoding", "Syndrome"]


class MetricExtractor(Protocol):
    def analyze(self) -> dict[str, object]:
        ...
