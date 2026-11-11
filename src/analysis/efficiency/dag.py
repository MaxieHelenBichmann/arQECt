"""Metric extractor for the dependency graph of a circuit. It exists to keep the DependencyDAG as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel
    from src.structure.dependency_dag import DependencyDAG

class DependencyDAGAnalyzer(MetricExtractor):
    def __init__(self, dag: DependencyDAG, hardware: HardwareModel):
        self.dag = dag
        self.hw = hardware

    def analyze(self) -> dict[str, object]:
        result : dict[str, object] = {}
        result["depth"] = self.dag.depth
        result["2q_depth"] = self._2q_depth()
        result["nr_gates"] = self._nr_gates()
        return result

    def _2q_depth(self) -> int:
        return 0

    def _nr_gates(self) -> int:
        return 0

    def _average_scheduling_flexibility(self) -> float:
        return 0.0

    def _capacity_constrained_depth_bound(self, capacity: int) -> int:
        return 0