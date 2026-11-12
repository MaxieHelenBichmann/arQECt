"""Metric extractor for the gate distribution of a circuit. It exists to keep the GateDistribution as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel
    from src.structure.gate_distribution import GateDistribution

class GateDistributionAnalyzer(MetricExtractor):
    def __init__(self, gd: GateDistribution,  hardware: HardwareModel):
        self.gd = gd
        self.hw = hardware

    def analyze(self) -> dict[str, object]: # assume non-empty circuit with >0 2q-gates, only resets/unitary 1- or 2-qubit gates/measurements, valid hardware parameters (especially gate set has canonical stim gate names)
        result : dict[str, object] = {}

        # composition
        result["nr_gates"] = self._nr_gates()
        result["share_2q_gates"] = self._share_2q_gates()
        result["share_measurements"] = self._share_measurements()
        result["share_resets"] = self._share_resets()

        # distance from native gate set
        result["share_native_gates"] = self._share_native_gates()
        result["share_native_2q_gates"] = self._share_native_2q_gates()
        result["estimated_nr_gates_post_transpilation"] = self._estimated_nr_gates_post_transpilation()
        result["estimated_nr_2q_gates_post_transpilation"] = self._estimated_nr_2q_gates_post_transpilation()
        return result

    def _nr_gates(self) -> int:
        """Total number of gates. Lower is better."""
        return self.gd.gd.total()

    def _share_2q_gates(self) -> float:
        """Fraction of gates acting on two qubits.

        Range [0, 1]. Lower is better usually.
        """
        return sum(count for name, count in self.gd.gd.items() if stim.gate_data(name).is_two_qubit_gate) / self._nr_gates()

    def _share_measurements(self) -> float:
        """Fraction of gates producing a measurement result (including measure-and-reset gates like MR).

        Range [0, 1]. Lower is better usually.
        """
        return sum(count for name, count in self.gd.gd.items() if stim.gate_data(name).produces_measurements) / self._nr_gates()

    def _share_resets(self) -> float:
        """Fraction of gates resetting a qubit (including measure-and-reset gates like MR, which also count as measurements).

        Range [0, 1].
        """
        return sum(count for name, count in self.gd.gd.items() if stim.gate_data(name).is_reset) / self._nr_gates()

    def _share_native_gates(self) -> float:
        """Fraction of gates that are in the hardware's native gate set.

        Range [0, 1]. Higher is better (1 = no decomposition needed before execution).
        """
        return sum(count for name, count in self.gd.gd.items() if name in self.hw.native_gates) / self._nr_gates()

    def _share_native_2q_gates(self) -> float:
        """Fraction of 2q gates that are in the hardware's native gate set.

        Range [0, 1]. Higher is better.
        """
        nr_2q_gates = sum(count for name, count in self.gd.gd.items() if stim.gate_data(name).is_two_qubit_gate)
        return sum(count for name, count in self.gd.gd.items() if stim.gate_data(name).is_two_qubit_gate and name in self.hw.native_gates) / nr_2q_gates

    def _estimated_nr_gates_post_transpilation(self) -> int:
        """Rough number of gates after decomposing each non-native gate on its own into native gates.

        Assumed costs of a non-native gate: SWAP = 3 native 2q gates + 6 1q gates, other 2q gates = 1 native 2q gate + 2 1q gates (e.g. CX = H CZ H),
        measurements / resets = basis change around a native one (e.g. MX = H M H), 1q gates = 2 native 1q gates.
        Range [nr_gates, inf). Lower is better.
        """
        total = 0
        for name, count in self.gd.gd.items():
            data = stim.gate_data(name)
            if name in self.hw.native_gates:
                total += count
            elif name == "SWAP":
                total += 9 * count
            else:
                total +=(3 if data.is_two_qubit_gate or data.produces_measurements or data.is_reset else 2) * count
        return total

    def _estimated_nr_2q_gates_post_transpilation(self) -> int:
        """Rough number of native 2q gates after decomposing each non-native gate on its own: SWAP = 3, every other 2q gate = 1.

        Range [nr_2q_gates, inf). Lower is better.
        """
        return sum(count * (3 if name == "SWAP" and name not in self.hw.native_gates else 1) for name, count in self.gd.gd.items() if stim.gate_data(name).is_two_qubit_gate)
