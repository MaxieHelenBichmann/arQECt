"""Evaluate the suitability of a given circuit on a specific hardware platform and configuration."""

from __future__ import annotations

import stim

from typing import TYPE_CHECKING
from src.analysis import CircuitFunction, CircuitAnalyzer
from src.models.hardware import TrappedIonsModel, SuperconductingQubitsModel, NeutralAtomsModel, HardwareModel


class CircuitEvaluator:
        def __init__(self, hardware: HardwareModel):
            self.hardware = hardware
            self.cost: float | None = None
            

        def evaluate(self, circuit: stim.Circuit, function: CircuitFunction):
            # 1.) Analyze the circuit
            analyzer : CircuitAnalyzer = CircuitAnalyzer(self.hardware)
            analyzer.analyze(circuit=circuit, function=function)

            # 2.) Evaluate the extracted metrics
            return self._cost_function(analyzer.metrics)


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
        