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
import numpy as np
from itertools import combinations
from math import comb

from src.analysis.quality._decoder import Decoder

from src.analysis import MetricExtractor

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

class EncoderCircuitAnalyzer(MetricExtractor):
    def __init__(self, noisy_circuit: stim.Circuit, stabilizers: list[stim.PauliString], logical_operators: list[tuple[stim.PauliString, stim.PauliString]], max_fault_sets: int, max_rel_tail: int, decoder: Decoder, logical_qubits: list[int] | None = None):
        self.noisy_circuit = noisy_circuit
        self.stabilizers = stabilizers
        self.logical_operators = logical_operators
        if logical_qubits is None or len(logical_qubits) != len(logical_operators):
            raise ValueError("An encoder circuit needs to specify one input qubit per logical qubit.")
        self.logical_qubits = logical_qubits
        self.max_fault_sets = max_fault_sets
        self.max_rel_tail = max_rel_tail
        self.decoder = decoder

    def analyze(self) -> dict[str, object]:
        result : dict[str, object] = {}

        reference = max(len(self.stabilizers[0]), self.noisy_circuit.num_qubits)
        
        experiment = stim.Circuit()
        for i, q in enumerate(self.logical_qubits):
            experiment.append("H", [reference + i])
            experiment.append("CX", [reference + i, q])
    
        experiment += self.noisy_circuit
        m0 = experiment.num_measurements
    
        targets = []
        correlators = []
        for i, (x, z) in enumerate(self.logical_operators):
            targets += [x, z]
            correlators += [x * stim.PauliString("_" * (reference + i) + "X"), z * stim.PauliString("_" * (reference + i) + "Z")]
    
        # add necessary measurements to verify prepared Bell state between reference and logical input qubits
        for obs in [*self.stabilizers, *correlators]:
            experiment.append("MPP", list(stim.target_combined_paulis(obs)))
    
        # add detectors / observables
        n_meas = experiment.num_measurements
        g = len(self.stabilizers)
    
        for j in range(g): # D_0..D_(g-1) syndrome
            experiment.append("DETECTOR", [stim.target_rec(m0 + j - n_meas)])
        for k in range(len(correlators)): # L_0..L_(2k-1) correlator operations
            experiment.append("OBSERVABLE_INCLUDE", [stim.target_rec(m0 + g + k - n_meas)], k)
    
        # correct noiseless original circuit: trivial syndrome & +1 correlators
        # --> detector/observable flips <--> wrong syndrome bits | wrong correlator measurement outcomes
        dem = experiment.detector_error_model(approximate_disjoint_errors=True)
    
        error_mechanisms = []  # (probability of mechanism, flip mask of mechanism): bit i = detector D_i, bit g + i = observable L_i
        for inst in dem.flattened():
            if inst.type == "error":
                mask = 0
                for t in inst.targets_copy():
                    if t.is_relative_detector_id(): 
                        mask ^= 1 << t.val
                    elif t.is_logical_observable_id(): 
                        mask ^= 1 << (g + t.val)
                error_mechanisms.append((inst.args_copy()[0], mask))
    
        # truncate: only consider fault sets S with at most max_weight fired error mechanisms
        #            ! dynamic max_weight: consider next weight until the not enumerated probability mass (tail) is at most ( max_rel_tail * measured error rate ) instead of a constant (better, because *scaling* uncertainty of LER),
        #                                  or the next weight has more than max_fault_sets fault sets
        # --> probability of that exact fault set: P(S) = prod_{i not in S} (1 - p_i) * prod_{i in S} (p_i) = P(no fault at all, aka no mechanism fires) * prod_{i in S} p_i / (1 - p_i)
        no_fault = np.prod([1 - p for p, _ in error_mechanisms])
        outcomes = {} # cache: flip mask of fault set -> syndrome correction failed because introduced log. operation
        total, failed = 0.0, 0.0
    
        max_weight = -1
        while max_weight == -1 or (1 - total > self.max_rel_tail * failed and 0 < comb(len(error_mechanisms), max_weight + 1) <= self.max_fault_sets):
            max_weight += 1
            for faults in combinations(range(len(error_mechanisms)), max_weight):
                prob, mask = no_fault, 0
                for i in faults:
                    p, m = error_mechanisms[i]
                    prob *= p / (1 - p)
                    mask ^= m
    
                if mask not in outcomes:
                    correction = self.decoder.decode(np.array([(mask >> j) & 1 for j in range(g)], dtype=np.uint8))
                    outcomes[mask] = any(((mask >> (g + k)) & 1) ^ (not correction.commutes(op)) for k, op in enumerate(targets))
    
                fail = outcomes[mask]
                total += prob
                failed += prob * fail
    
        # not enumerated probability mass (fault sets with > max_weight faults) --> can be accepted / rejected , failed / successful (unknown) --> worst/best case bounds
        tail = 1 - total
    
        result["n_mechanisms"] = len(error_mechanisms)
        result["max_weight"] = max_weight
        result["truncation_tail"] = tail
        result["logical_error_rate"] = failed / total
        result["logical_error_rate_lower"] = failed
        result["logical_error_rate_upper"] = failed + tail

        return result
