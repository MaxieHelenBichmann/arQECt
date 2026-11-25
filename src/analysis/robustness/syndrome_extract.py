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
from itertools import combinations
from math import comb
import numpy as np

from src.analysis.robustness._decoder import Decoder

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

class SyndromeExtractionAnalyzer(MetricExtractor):
    def __init__(self, noisy_circuit: stim.Circuit, stabilizers: list[stim.PauliString], logical_operators: list[tuple[stim.PauliString, stim.PauliString]], max_fault_sets: int, max_rel_tail: int, decoder: Decoder, syndrome_measurements: list[int] | None = None):
        self.noisy_circuit = noisy_circuit
        self.stabilizers = stabilizers
        self.logical_operators = logical_operators
        if syndrome_measurements is None or len(syndrome_measurements) != len(stabilizers):
            raise ValueError("A syndrome extraction circuit needs to specify one measurement per stabilizer.")
        self.syndrome_measurements = syndrome_measurements
        self.max_fault_sets = max_fault_sets
        self.max_rel_tail = max_rel_tail
        self.decoder = decoder

    def analyze(self) -> dict[str, object]:
        result : dict[str, object] = {}

        reference = max(len(self.stabilizers[0]), self.noisy_circuit.num_qubits)
        
        targets = []
        correlators = []
        for i, (x, z) in enumerate(self.logical_operators):
            targets += [x, z]
            correlators += [x * stim.PauliString("_" * (reference + i) + "X"), z * stim.PauliString("_" * (reference + i) + "Z")]

        # add necessary measurements to force / prepare one arbitrary Bell state between reference and logical qubits
        observables = [*self.stabilizers, *correlators]
        experiment = stim.Circuit()
        for obs in observables:
            experiment.append("MPP", list(stim.target_combined_paulis(obs)))
        
        m0 = experiment.num_measurements
        experiment += self.noisy_circuit
        m1 = experiment.num_measurements

        # add necessary measurements to verify the preservation of the Bell state
        for observable in observables:
            experiment.append("MPP", list(stim.target_combined_paulis(observable)))

        # add detectors / observables
        n_meas = experiment.num_measurements
        g = len(self.stabilizers)

        for j in range(g): # D_0..D_(g-1) syndrome (state preservation)
            experiment.append("DETECTOR", [stim.target_rec(j - n_meas), stim.target_rec(m1 + j - n_meas)])
        for j, idx in enumerate(self.syndrome_measurements): # D_g..D_(2g-1) syndrome extraction: initial perfect vs noisily extracted stabilizer measurement
            experiment.append("DETECTOR", [stim.target_rec(j - n_meas), stim.target_rec(m0 + idx - n_meas)])
        for k in range(len(correlators)): # L_0..L_(2k-1) correlator operations (state preservation)
            experiment.append("OBSERVABLE_INCLUDE", [stim.target_rec(g + k - n_meas), stim.target_rec(m1 + g + k - n_meas)], k)

        # correct noiseless original circuit: unchanged stabilizers & correlators, extracted syndrome = initial syndrome
        # --> detector/observable flips <--> changed syndrome bits | wrongly extracted syndrome bits | changed correlator measurement outcomes
        dem = experiment.detector_error_model(approximate_disjoint_errors=True)

        error_mechanisms = []  # (probability of mechanism, flip mask of mechanism): bit i = detector D_i, bit 2g + i = observable L_i
        for inst in dem.flattened():
            if inst.type == "error":
                mask = 0
                for t in inst.targets_copy():
                    if t.is_relative_detector_id():
                        mask ^= 1 << t.val
                    elif t.is_logical_observable_id():
                        mask ^= 1 << (2 * g + t.val)
                error_mechanisms.append((inst.args_copy()[0], mask))
        
        # truncate: only consider fault sets S with at most max_weight fired error mechanisms
        #            ! dynamic max_weight: consider next weight until the not enumerated probability mass (tail) is at most ( max_rel_tail * the smallest measured rate ) instead of a constant (better, because *scaling* uncertainty of LER),
        #                                  or the next weight has more than max_fault_sets fault sets (or none: all fault sets enumerated)
        # --> probability of that exact fault set: P(S) = prod_{i not in S} (1 - p_i) * prod_{i in S} (p_i) = P(no fault at all, aka no mechanism fires) * prod_{i in S} p_i / (1 - p_i)
        no_fault = np.prod([1 - p for p, _ in error_mechanisms])
        outcomes = {} # cache: flip mask of fault set -> (wrongly extracted syndrome, syndrome correction failed because introduced log. operation)
        total, wrong, failed = 0.0, 0.0, 0.0

        max_weight = -1
        while max_weight == -1 or (1 - total > self.max_rel_tail * min(failed, wrong) and 0 < comb(len(error_mechanisms), max_weight + 1) <= self.max_fault_sets):
            max_weight += 1
            for faults in combinations(range(len(error_mechanisms)), max_weight):
                prob, mask = no_fault, 0
                for i in faults:
                    p, m = error_mechanisms[i]
                    prob *= p / (1 - p)
                    mask ^= m

                if mask not in outcomes:
                    correction = self.decoder.decode(np.array([(mask >> j) & 1 for j in range(g)], dtype=np.uint8))
                    outcomes[mask] = (
                        (mask >> g) % (1 << g) != 0,
                        any(((mask >> (2 * g + k)) & 1) ^ (not correction.commutes(op)) for k, op in enumerate(targets))
                    )

                syn, fail = outcomes[mask]
                total += prob
                wrong += prob * syn
                failed += prob * fail

        # not enumerated probability mass (fault sets with > max_weight faults) --> can be wrong / right , failed / successful (unknown) --> worst/best case bounds
        tail = 1 - total

        result["n_mechanisms"] = len(error_mechanisms)
        result["max_weight"] = max_weight
        result["truncation_tail"] = tail
        result["syndrome_error_rate"] = wrong / total
        result["syndrome_error_rate_lower"] = wrong
        result["syndrome_error_rate_upper"] = wrong + tail
        result["logical_error_rate"] = failed / total
        result["logical_error_rate_lower"] = failed
        result["logical_error_rate_upper"] = failed + tail

        return result

