"""State preparation circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim
import numpy as np

from src.analysis.quality._decoder import Decoder

from src.analysis import MetricExtractor
from src.circuits.circuit import QuantumState

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel



class StatePreparationCircuitAnalyzer(MetricExtractor):
    def __init__(self, hardware: HardwareModel, stabilizers: list[stim.PauliString], logical_operators: list[tuple[stim.PauliString, stim.PauliString]], max_shots: int, decoder: Decoder, accept_if: dict[int, int] | None = None, state: QuantumState | None = None):
        self.hw = hardware
        self.stabilizers = stabilizers
        match state:
            case "0":
                self.target_logicals = [z for (_,z) in logical_operators]
            case "+":
                self.target_logicals = [x for (x,_) in logical_operators]
            case "1":
                self.target_logicals = [-z for (_,z) in logical_operators]
            case "-":
                self.target_logicals = [-x for (x,_) in logical_operators]
            case _:
                raise ValueError("A state preparation circuit needs to specify which logical state it prepares.")
        self.max_shots = max_shots
        self.accept_if = {} if accept_if is None else accept_if
        self.decoder = decoder

    def analyze(self) -> dict[str, object]:
        result : dict[str, object] = {}

        # run experiment with multiple shots
        experiment = self.hw.noisy_circuit.copy()
        existent_measurements = experiment.num_measurements

        for observable in [*self.stabilizers, *self.target_logicals]:
            experiment.append("MPP", list(stim.target_combined_paulis(observable)))

        samples = experiment.compile_sampler().sample(self.max_shots)

        # discard results that verification does not accept
        accepted_mask = np.ones(self.max_shots, dtype=bool)

        for index, expected in self.accept_if.items():
            accepted_mask &= samples[:, index] == expected

        accepted = samples[accepted_mask]
        result["n_accepted"] = len(accepted)
        result["acceptance_rate"] = len(accepted) / self.max_shots

        # decode and count logical failures
        g = len(self.stabilizers)

        syndromes = accepted[:, existent_measurements:existent_measurements + g]
        logicals = accepted[:, existent_measurements + g:]

        failures = 0
        for syndrome, measurements in zip(syndromes, logicals, strict=True):
            correction = self.decoder.decode(syndrome)

            # recovery action -> logical error?
            corrected = [
                bool(m) ^ (not correction.commutes(op)) for m, op in zip(measurements, self.target_logicals, strict=True)
            ]
            failures += int(any(corrected))

        result["n_failures"] = failures
        result["logical_error_rate"] = failures / len(accepted) if len(accepted) else None

        return result