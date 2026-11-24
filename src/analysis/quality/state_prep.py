"""State preparation circuits.

This analysis considers noise robustness, and assumes correctness of the original circuit.
The supported state preparation circuits prepare a concretely fixed logical state and can be structured after the following protocols:

A) Normal state preparation (probably "non-fault tolerant state preparation")
    - simple state preparation circuit without any gadgets
    - resulting output state is intended to be the targeted state (no rejection of shots)

B) Verification-based state preparation ("Non-deterministic fault-tolerant state preparation")
    - simple state preparation + subsequent verification circuit
    - verification circuit can either accept (no detected error) or reject (detected error) the prepared state
    - if state rejected, start all over
    - resulting output state is not necessarily intended to be the targeted state (rejection of shots)

C) Correction-based state preparation ("Deterministic fault-tolerant state preparation")
    - simple state preparation + subsequent verification circuit + subsequent correction circuit(s)
    - verification circuit can either accept (no detected error) or reject (detected error) the prepared state
    - if state rejected, correction circuit applies recovery operation based on the verification result
    - resulting output state is intended to be the targeted state (no rejection of shots)

For a non-deterministic state preparation, certain shots (the ones which the verifier rejects, as specified via `accept_if`) are discarded.

After running the state preparation circuit and obtaining a targeted state, the code's stabilizers and targeted logical operators
are measured as observables, the stabilizer syndrome is decoded and the recovery operation virtually applied and evaluated to find out whether 
the noisily prepared and error-corrected state actually corresponds to the right logical state.
"""

# TODO: QECirc circuits dont have correction part of their downloadable det-ft-state-prep circuits 
# (it requires "classical" branches that, when implemented as a larger circuits with more controlled operations, arent really realistic and add more gates and noise than they really would - so they might be slightly pessimistic...
# also, those classically controlled operations rely on classical AND logic, which stim doesnt support)
# so for now I decided to interpret those circuits as their verification-variant, aka as adapting protocol B as well (not perfect but it is how it is)

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
    def __init__(self, noisy_circuit: stim.Circuit, stabilizers: list[stim.PauliString], logical_operators: list[tuple[stim.PauliString, stim.PauliString]], max_shots: int, decoder: Decoder, accept_if: dict[int, int] | None = None, state: QuantumState | None = None):
        self.noisy_circuit = noisy_circuit
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
        experiment = self.noisy_circuit.copy()
        existent_measurements = experiment.num_measurements

        for observable in [*self.stabilizers, *self.target_logicals]:
            experiment.append("MPP", list(stim.target_combined_paulis(observable)))

        samples = experiment.compile_sampler().sample(self.max_shots)

        # discard results that verification does not accept (Protocol B)
        accepted_mask = np.ones(self.max_shots, dtype=bool)

        for index, expected in self.accept_if.items():
            accepted_mask &= samples[:, index] == expected

        accepted = samples[accepted_mask]
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

        result["logical_error_rate"] = failures / len(accepted) if len(accepted) else None

        # 95% Wilson score confidence interval of the logical error rate
        n, z = len(accepted), 1.96
        if n:
            center = (failures / n + z**2 / (2 * n)) / (1 + z**2 / n)
            half = z / (1 + z**2 / n) * (failures / n * (1 - failures / n) / n + z**2 / (4 * n**2)) ** 0.5
            result["logical_error_rate_ci_lower"] = center - half
            result["logical_error_rate_ci_upper"] = center + half
        else:
            result["logical_error_rate_ci_lower"] = None
            result["logical_error_rate_ci_upper"] = None

        return result