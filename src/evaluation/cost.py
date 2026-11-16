"""Evaluate the suitability of a given circuit on a specific hardware platform and configuration."""

from __future__ import annotations

import stim

from typing import TYPE_CHECKING
from src.circuits.circuit import CircuitFunction
from src.analysis.analyzer import CircuitAnalysisResult
from src.models.hardware import TrappedIonsModel, SuperconductingQubitsModel, NeutralAtomsModel, HardwareModel


class CircuitEvaluator:
        def __init__(self, hardware: HardwareModel):
            self.hardware = hardware

        def evaluate(self, analysis: CircuitAnalysisResult) -> float:
            """Evaluate the extracted circuit properties."""
            return self._cost_function(analysis.compilation_cost_metrics, analysis.runtime_cost_metrics, analysis.noise_cost_metrics)


        def _cost_function(self, runtime_metrics : dict[str, object], compilation_metrics : dict[str, object], noise_metrics : dict[str, object]) -> float:
            if isinstance(self.hardware, TrappedIonsModel):
                return cost_ti(runtime_metrics)
            if isinstance(self.hardware, NeutralAtomsModel):
                return cost_na(runtime_metrics)
            return cost_sc(runtime_metrics)


def cost_ti(metrics : dict[str, object]) -> float:
    return 0.0

def cost_na(metrics : dict[str, object]) -> float:
    return 0.0

def cost_sc(metrics : dict[str, object]) -> float:
    return 0.0
        