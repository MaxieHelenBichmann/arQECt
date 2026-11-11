"""Metric extractor for the dependency graph of a circuit. It exists to keep the DependencyDAG as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.structure.dependency_dag import DependencyDAG

class DependencyDAGAnalyzer(MetricExtractor):
    def __init__(self, dag: DependencyDAG):
        self.dag = dag

    def analyze(self) -> dict:
        {}