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

from itertools import combinations
from math import comb

import stim
import numpy as np

from src.analysis.quality._decoder import Decoder

from src.analysis import MetricExtractor
from src.circuits.circuit import QuantumState

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

class StatePreparationCircuitAnalyzer(MetricExtractor):
    def __init__(self, noisy_circuit: stim.Circuit, stabilizers: list[stim.PauliString], logical_operators: list[tuple[stim.PauliString, stim.PauliString]], max_fault_sets: int, max_rel_tail: int, decoder: Decoder, accept_if: dict[int, int] | None = None, state: QuantumState | None = None):
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
        self.max_fault_sets = max_fault_sets
        self.max_rel_tail = max_rel_tail
        self.accept_if = {} if accept_if is None else accept_if
        self.decoder = decoder

    def analyze(self) -> dict[str, object]:
        result : dict[str, object] = {}

        # run experiment with multiple shots
        experiment = self.noisy_circuit.copy()
        m0 = experiment.num_measurements

        # add necessary measurements to verify prepared state
        for observable in [*self.stabilizers, *self.target_logicals]:
            experiment.append("MPP", list(stim.target_combined_paulis(observable)))

        # add detectors / observables
        n_meas = experiment.num_measurements
        g = len(self.stabilizers)
        v = len(self.accept_if)

        for j in range(g): # D_0..D_(g-1) syndrome
            experiment.append("DETECTOR", [stim.target_rec(m0 + j - n_meas)])
        for idx in self.accept_if: # D_g..D_(g+v-1) post-selection
            experiment.append("DETECTOR", [stim.target_rec(idx - n_meas)])
        for k in range(len(self.target_logicals)): # L_0..L_k target logical operations
            experiment.append("OBSERVABLE_INCLUDE", [stim.target_rec(m0 + g + k - n_meas)], k)


        # correct noiseless original circuit: trivial syndrome & +1 target logical operations & accepting verification
        # --> detector/observable flips <--> wrong syndrome bits | wrong logical measurement outcomes | verification rejections
        dem = experiment.detector_error_model(approximate_disjoint_errors=True)
    
        error_mechanisms = []  # (probability of mechanism, flip mask of mechanism): bit i = detector D_i, bit g + v + i = observable L_i
        for inst in dem.flattened():
            if inst.type == "error":
                mask = 0
                for t in inst.targets_copy():
                    if t.is_relative_detector_id(): 
                        mask ^= 1 << t.val
                    elif t.is_logical_observable_id(): 
                        mask ^= 1 << (g + v + t.val)
                error_mechanisms.append((inst.args_copy()[0], mask))

        # truncate: only consider fault sets S with at most max_weight fired error mechanisms 
        #            ! dynamic max_weight: consider next weight until the not enumerated probability mass (tail) is at most ( max_rel_tail * the smallest measured rate ) instead of a constant (better, because *scaling* uncertainty of LER),
        #                                  or the next weight has more than max_fault_sets fault sets
        # --> probability of that exact fault set: P(S) = prod_{i not in S} (1 - p_i) * prod_{i in S} (p_i) = P(no fault at all, aka no mechanism fires) * prod_{i in S} p_i / (1 - p_i)
        no_fault = np.prod([1 - p for p, _ in error_mechanisms])
        outcomes = {} # cache: flip mask of fault set -> (physical state accepted by verification, syndrome correction failed because introduced log. operation)
        total, accepted, failed = 0.0, 0.0, 0.0

        max_weight = -1
        while max_weight == -1 or (1 - total > self.max_rel_tail * (min(failed, total - accepted) if v else failed) and 0 < comb(len(error_mechanisms), max_weight + 1) <= self.max_fault_sets):
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
                        (mask >> g) % (1 << v) == 0,
                        any(((mask >> (g + v + k)) & 1) ^ (not correction.commutes(op)) for k, op in enumerate(self.target_logicals))
                    )

                acc, fail = outcomes[mask]
                total += prob
                accepted += prob * acc
                failed += prob * (acc and fail)

        # not enumerated probability mass (fault sets with > max_weight faults) --> can be accepted / rejected , failed / successful (unknown) --> worst/best case bounds
        tail = 1 - total

        result["n_mechanisms"] = len(error_mechanisms)
        result["max_weight"] =  max_weight
        result["truncation_tail"] =  tail
        result["rejection_rate"] =  (total - accepted) / total
        result["rejection_rate_lower"] =  total - accepted
        result["rejection_rate_upper"] =  total - accepted + tail
        result["logical_error_rate"] =  failed / accepted if accepted else None
        result["logical_error_rate_lower"] =  failed / (accepted + tail) if accepted + tail else None
        result["logical_error_rate_upper"] =  (failed + tail) / (accepted + tail) if accepted + tail else None
    
        return result