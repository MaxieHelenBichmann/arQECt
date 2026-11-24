"""Encoder circuits.

This analysis considers noise robustness, and assumes correctness of the original circuit.
It supports encoder circuits of non-subsystem quantum error-correction codes, which map an arbitrary given logical state to the corresponding physical state.

We do not want to choose an arbitrary subset of logical states to verify whether the noisy circuit preserves its functionality, but capture 
information about the whole code space. Thus, the logical qubits qL_1, ... , qL_k corresponding to the logical operators (XL_1, ZL_1), ... , (XL_k, ZL_k)
are entangled with noiseless reference qubits qR_1, ... , qR_k, constructing the fully entangled state.

The ideal observables of this state are the stabilizers on the physical-state subsystem (and identity on the reference subsystem) 
and the 2k correlators with a single X (or Z) on qR_i in the reference subsystem and the corresponding logical operator XL_i (or ZL_i) on the physical qubits.
The stabilizer syndrome is decoded and the recovery operation virtually applied to the physical qubits and evaluated to find out whether 
the half-noisily prepared and error-corrected state actually corresponds to the logical Bell state between the logical and reference qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

from src.analysis.quality._decoder import Decoder

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

class EncoderCircuitAnalyzer(MetricExtractor):
    def __init__(self, noisy_circuit: stim.Circuit, stabilizers: list[stim.PauliString], logical_operators: list[tuple[stim.PauliString, stim.PauliString]], max_shots: int, decoder: Decoder, logical_qubits: list[int] | None = None):
        self.noisy_circuit = noisy_circuit
        self.stabilizers = stabilizers
        self.logical_operators = logical_operators
        if logical_qubits is None or len(logical_qubits) != len(logical_operators):
            raise ValueError("An encoder circuit needs to specify one input qubit per logical qubit.")
        self.logical_qubits = logical_qubits
        self.max_shots = max_shots
        self.decoder = decoder

    def analyze(self) -> dict[str, object]:
        result : dict[str, object] = {}

        # entangle each input qubit with a noiseless reference qubit (Bell pair)
        reference = max(len(self.stabilizers[0]), self.noisy_circuit.num_qubits) # first reference qubit index, after all circuit qubits

        experiment = stim.Circuit()
        for i, q in enumerate(self.logical_qubits):
            experiment.append("H", [reference + i])
            experiment.append("CX", [reference + i, q])

        # run experiment with multiple shots
        experiment += self.noisy_circuit
        existent_measurements = experiment.num_measurements

        targets = []
        correlators = []
        for i, (x, z) in enumerate(self.logical_operators):
            targets += [x, z]
            correlators += [x * stim.PauliString("_" * (reference + i) + "X"), z * stim.PauliString("_" * (reference + i) + "Z")]

        for observable in [*self.stabilizers, *correlators]:
            experiment.append("MPP", list(stim.target_combined_paulis(observable)))

        samples = experiment.compile_sampler().sample(self.max_shots)

        # decode and count logical failures
        g = len(self.stabilizers)

        syndromes = samples[:, existent_measurements:existent_measurements + g]
        logicals = samples[:, existent_measurements + g:]

        failures = 0
        for syndrome, measurements in zip(syndromes, logicals, strict=True):
            correction = self.decoder.decode(syndrome)

            # recovery action -> logical error?
            corrected = [
                bool(m) ^ (not correction.commutes(op)) for m, op in zip(measurements, targets, strict=True)
            ]
            failures += int(any(corrected))

        result["n_failures"] = failures
        result["logical_error_rate"] = failures / self.max_shots

        # 95% Wilson score confidence interval of the logical error rate
        n, z = self.max_shots, 1.96
        center = (failures / n + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z / (1 + z**2 / n) * (failures / n * (1 - failures / n) / n + z**2 / (4 * n**2)) ** 0.5
        result["logical_error_rate_ci_lower"] = center - half
        result["logical_error_rate_ci_upper"] = center + half

        return result
