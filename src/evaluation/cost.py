"""Evaluate the suitability of a given circuit on a specific hardware platform and configuration."""

from __future__ import annotations

import stim

from typing import TYPE_CHECKING
from src.analysis import CircuitFunction
from src.analysis.analyzer import CircuitAnalysisResult
from src.models.hardware import TrappedIonsModel, SuperconductingQubitsModel, NeutralAtomsModel, HardwareModel


class CircuitEvaluator:
        def __init__(self, hardware: HardwareModel):
            self.hardware = hardware

        def evaluate(self, analysis: CircuitAnalysisResult) -> float:
            """Evaluate the extracted circuit properties."""
            return self._cost_function(analysis.metrics)


        def _cost_function(self, metrics : dict[str, object]) -> float:
            if isinstance(self.hardware, TrappedIonsModel):
                return cost_ti(metrics)
            if isinstance(self.hardware, NeutralAtomsModel):
                return cost_na(metrics)
            return cost_sc(metrics)


def cost_ti(metrics : dict[str, object]) -> float:
    return 0.0

def cost_na(metrics : dict[str, object]) -> float:
    return 0.0

def cost_sc(metrics : dict[str, object]) -> float:
    return 0.0
        