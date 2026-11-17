"""State preparation circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim
import numpy as np

from ldpc import BpOsdDecoder

from src.models.noise.noise import ErrorChannel

from src.analysis import MetricExtractor
from src.circuits.circuit import QuantumState

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel



class StatePreparationCircuitAnalyzer(MetricExtractor):
    def __init__(self, hardware: HardwareModel, stabilizers: list[stim.PauliString], logical_operators: tuple[stim.PauliString, stim.PauliString], max_shots: int, accept_if: dict[int, int] | None = None, state: QuantumState | None = None):
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
        n = len(self.stabilizers[0])

        syndromes = accepted[:, existent_measurements:existent_measurements + g]
        logicals = accepted[:, existent_measurements + g:]

        decoder = self._decoder()

        failures = 0
        for syndrome, measurements in zip(syndromes, logicals, strict=True):
            error = np.asarray(decoder.decode(syndrome), dtype=np.uint8)
            ez, ex = error[:n], error[n:]

            correction = stim.PauliString("".join("IZXY"[2 * int(x) + int(z)] for x, z in zip(ex, ez, strict=True)))

            # recovery action -> logical error?
            corrected = [
                bool(m) ^ (not correction.commutes(op)) for m, op in zip(measurements, self.target_logicals, strict=True)
            ]
            failures += int(any(corrected))

        result["n_failures"] = failures
        result["logical_error_rate"] = failures / len(accepted) if len(accepted) else None

        return result

    def _decoder(self) -> BpOsdDecoder: # noise-informed decoder
        n = len(self.stabilizers[0])

        x =np.array([[s[q] in (1, 2) for q in range(n)]for s in self.stabilizers], dtype=np.uint8)
        z = np.array([[s[q] in (2, 3) for q in range(n)]for s in self.stabilizers], dtype=np.uint8)

        s = np.hstack([x, z])

        px, pz = self.hw.noise_model.get_aggregated_probabilities()
        px, pz = max(px, 1e-6), max(pz, 1e-6)

        return BpOsdDecoder(s, error_channel=[pz] * n + [px] * n, bp_method="product_sum", max_iter=30, osd_method="osd_cs", osd_order=2, input_vector_type="syndrome")

