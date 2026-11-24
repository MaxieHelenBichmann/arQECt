"""Syndrome extraction circuits.

This analysis considers noise robustness, and assumes correctness of the original circuit.
The syndrome extraction circuits need to extract the right syndrome of the physical state by measuring the stabilizers (in an order specified by `syndrome_measurements`)
while preserving the encoded logical information.

Similar to the encoding circuit analysis, we do not want to choose an arbitrary subset of states to analyze the circuit's quality, but capture information about 
all possibilities. By noiselessly measuring the required stabilizers and correlators, we project the state onto the single, fully entangled logical Bell pair
before running the noisy syndrome extraction circuit on it.

Even though this only directly tests whether a physical state belonging to the code space results in the right syndrome under this noise, it also verifies 
the other direction (does a physical state not belonging to the code space result in the right syndrome?) utilizing the fact that we only introduce Pauli noise, 
thus errors flip the measurements linearly.

Only code-space inputs are simulated directly, but since this analysis only supports Clifford circuits with Pauli noise, input errors and circuit faults flip the 
measurement record independently (mod 2). Thus, the probability of extracting a wrong syndrome is therefore independent of the input, so the results are applicable 
to input physical states outside the code space.

The noisily extracted syndrome and logical state are compared to the the results of the preceding and the subsequent round of noiseless measurements to find out 
whether the noisily measured syndrome is indeed the initial perfect syndrome and whether the logical state is preserved after the syndrome extraction circuit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

from src.analysis.quality._decoder import Decoder

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

class SyndromeExtractionAnalyzer(MetricExtractor):
    def __init__(self, noisy_circuit: stim.Circuit, stabilizers: list[stim.PauliString], logical_operators: list[tuple[stim.PauliString, stim.PauliString]], max_shots: int, decoder: Decoder, syndrome_measurements: list[int] | None = None):
        self.noisy_circuit = noisy_circuit
        self.stabilizers = stabilizers
        self.logical_operators = logical_operators
        if syndrome_measurements is None or len(syndrome_measurements) != len(stabilizers):
            raise ValueError("A syndrome extraction circuit needs to specify one measurement per stabilizer.")
        self.syndrome_measurements = syndrome_measurements
        self.max_shots = max_shots
        self.decoder = decoder

    def analyze(self) -> dict[str, object]:
        result : dict[str, object] = {}

        reference = max(len(self.stabilizers[0]), self.noisy_circuit.num_qubits) # first reference qubit index, after all circuit qubits

        targets = []
        correlators = []
        for i, (x, z) in enumerate(self.logical_operators):
            targets += [x, z]
            correlators += [x * stim.PauliString("_" * (reference + i) + "X"), z * stim.PauliString("_" * (reference + i) + "Z")]

        observables = [*self.stabilizers, *correlators]

        # ideal input state: noiseless projection onto stabilizers + logical Bell pairs with reference qubits
        experiment = stim.Circuit()
        for observable in observables:
            experiment.append("MPP", list(stim.target_combined_paulis(observable)))
        existent_measurements = experiment.num_measurements

        # run experiment with multiple shots
        experiment += self.noisy_circuit
        final_measurements = experiment.num_measurements

        for observable in observables:
            experiment.append("MPP", list(stim.target_combined_paulis(observable)))

        samples = experiment.compile_sampler().sample(self.max_shots)

        # compare syndrome extracted by the circuit with the initial perfect syndrome
        g = len(self.stabilizers)

        initial = samples[:, :existent_measurements]
        extracted = samples[:, [existent_measurements + i for i in self.syndrome_measurements]]
        result["syndrome_error_rate"] = int((extracted != initial[:, :g]).any(axis=1).sum()) / self.max_shots

        # decode and count logical failures (differences between initial and final perfect measurements)
        difference = initial ^ samples[:, final_measurements:]
        syndromes = difference[:, :g]
        logicals = difference[:, g:]

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
