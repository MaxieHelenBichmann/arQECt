
from itertools import combinations

from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim

import numpy as np

from src.models.hardware import HardwareModel
from src.analysis.quality._decoder import Decoder

def run_dem_analysis_sp(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_weight: int) -> dict:
    noisy_circ = noisy_circuit.copy()
    m0 = noisy_circ.num_measurements

    match circuit_data.quantum_state:
        case "0":
            target_logicals = [z for (_,z) in circuit_data.logical_operators]
        case "+":
            target_logicals = [x for (x,_) in circuit_data.logical_operators]
        case "1":
            target_logicals = [-z for (_,z) in circuit_data.logical_operators]
        case "-":
            target_logicals = [-x for (x,_) in circuit_data.logical_operators]

    # add necessary measurements to verify prepared state
    for obs in [*circuit_data.stabilizers, *target_logicals]:
        noisy_circ.append("MPP", stim.target_combined_paulis(obs))

    # add detectors / observables
    n_meas = noisy_circ.num_measurements
    g = len(circuit_data.stabilizers)
    v = len(circuit_data.verification)

    for j in range(g): # D_0..D_(g-1) syndrome
        noisy_circ.append("DETECTOR", [stim.target_rec(m0 + j - n_meas)])
    for idx in circuit_data.verification: # D_g..D_(g+v-1) post-selection
        noisy_circ.append("DETECTOR", [stim.target_rec(idx - n_meas)])
    for k in range(len(target_logicals)): # L_0..L_k target logical operations
        noisy_circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(m0 + g + k - n_meas)], k)

    # correct noiseless original circuit: trivial syndrome & +1 target logical operations & accepting verification
    # --> detector/observable flips <--> wrong syndrome bits | wrong logical measurement outcomes | verification rejections
    dem = noisy_circ.detector_error_model(approximate_disjoint_errors=True)

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
    # --> probability of that exact fault set: P(S) = prod_{i not in S} (1 - p_i) * prod_{i in S} (p_i) = P(no fault at all, aka no mechanism fires) * prod_{i in S} p_i / (1 - p_i)
    no_fault = np.prod([1 - p for p, _ in error_mechanisms])
    outcomes = {} # cache: flip mask of fault set -> (physical state accepted by verification, syndrome correction failed because introduced log. operation)
    total, accepted, failed = 0.0, 0.0, 0.0

    for weight in range(max_weight + 1):
        for faults in combinations(range(len(error_mechanisms)), weight):
            prob, mask = no_fault, 0
            for i in faults:
                p, m = error_mechanisms[i]
                prob *= p / (1 - p)
                mask ^= m

            if mask not in outcomes:
                correction = decoder.decode(np.array([(mask >> j) & 1 for j in range(g)], dtype=np.uint8))
                outcomes[mask] = (
                    (mask >> g) % (1 << v) == 0,
                    any(((mask >> (g + v + k)) & 1) ^ (not correction.commutes(op)) for k, op in enumerate(target_logicals))
                )

            acc, fail = outcomes[mask]
            total += prob
            accepted += prob * acc
            failed += prob * (acc and fail)

    # not enumerated probability mass (fault sets with > max_weight faults) --> can be accepted / rejected , failed / successful --> worst/best case bounds
    tail = 1 - total

    return {
        "n_mechanisms": len(error_mechanisms),
        "truncation_tail": tail,
        "rejection_rate": (total - accepted) / total,
        "rejection_rate_lower": total - accepted,
        "rejection_rate_upper": total - accepted + tail,
        "logical_error_rate": failed / accepted,
        "logical_error_rate_lower": failed / (accepted + tail),
        "logical_error_rate_upper": (failed + tail) / (accepted + tail),
    }

def run_dem_analysis_en(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_weight: int) -> dict:
    reference = max(len(circuit_data.stabilizers[0]), noisy_circuit.num_qubits)

    experiment = stim.Circuit()
    for i, q in enumerate(circuit_data.logical_qubits):
        experiment.append("H", [reference + i])
        experiment.append("CX", [reference + i, q])

    experiment += noisy_circuit
    m0 = experiment.num_measurements

    targets = []
    correlators = []
    for i, (x, z) in enumerate(circuit_data.logical_operators):
        targets += [x, z]
        correlators += [x * stim.PauliString("_" * (reference + i) + "X"), z * stim.PauliString("_" * (reference + i) + "Z")]

    # add necessary measurements to verify prepared Bell state between reference and logical input qubits
    for obs in [*circuit_data.stabilizers, *correlators]:
        experiment.append("MPP", list(stim.target_combined_paulis(obs)))

    # add detectors / observables
    n_meas = experiment.num_measurements
    g = len(circuit_data.stabilizers)

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
    # --> probability of that exact fault set: P(S) = prod_{i not in S} (1 - p_i) * prod_{i in S} (p_i) = P(no fault at all, aka no mechanism fires) * prod_{i in S} p_i / (1 - p_i)
    no_fault = np.prod([1 - p for p, _ in error_mechanisms])
    outcomes = {} # cache: flip mask of fault set -> syndrome correction failed because introduced log. operation
    total, failed = 0.0, 0.0

    for weight in range(max_weight + 1):
        for faults in combinations(range(len(error_mechanisms)), weight):
            prob, mask = no_fault, 0
            for i in faults:
                p, m = error_mechanisms[i]
                prob *= p / (1 - p)
                mask ^= m

            if mask not in outcomes:
                correction = decoder.decode(np.array([(mask >> j) & 1 for j in range(g)], dtype=np.uint8))
                outcomes[mask] = any(((mask >> (g + k)) & 1) ^ (not correction.commutes(op)) for k, op in enumerate(targets))

            fail = outcomes[mask]
            total += prob
            failed += prob * fail

    # not enumerated probability mass (fault sets with > max_weight faults) --> can be accepted / rejected , failed / successful --> worst/best case bounds
    tail = 1 - total

    return {
        "n_mechanisms": len(error_mechanisms),
        "truncation_tail": tail,
        "logical_error_rate": failed / total,
        "logical_error_rate_lower": failed,
        "logical_error_rate_upper": failed + tail,
    }

def run_dem_analysis_se(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_weight: int) -> dict:
    reference = max(len(circuit_data.stabilizers[0]), noisy_circuit.num_qubits)
    
    targets = []
    correlators = []
    for i, (x, z) in enumerate(circuit_data.logical_operators):
        targets += [x, z]
        correlators += [x * stim.PauliString("_" * (reference + i) + "X"), z * stim.PauliString("_" * (reference + i) + "Z")]

    # add necessary measurements to force / prepare one arbitrary Bell state between reference and logical qubits
    observables = [*circuit_data.stabilizers, *correlators]
    experiment = stim.Circuit()
    for obs in observables:
        experiment.append("MPP", list(stim.target_combined_paulis(obs)))
    
    m0 = experiment.num_measurements
    experiment += noisy_circuit
    m1 = experiment.num_measurements

    # add necessary measurements to verify the preservation of the Bell state
    for observable in observables:
        experiment.append("MPP", list(stim.target_combined_paulis(observable)))

    # add detectors / observables
    n_meas = experiment.num_measurements
    g = len(circuit_data.stabilizers)

    for j in range(g): # D_0..D_(g-1) syndrome (state preservation)
        experiment.append("DETECTOR", [stim.target_rec(j - n_meas), stim.target_rec(m1 + j - n_meas)])
    for j, idx in enumerate(circuit_data.syndrome_measurements): # D_g..D_(2g-1) syndrome extraction: initial perfect vs noisily extracted stabilizer measurement
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
    # --> probability of that exact fault set: P(S) = prod_{i not in S} (1 - p_i) * prod_{i in S} (p_i) = P(no fault at all, aka no mechanism fires) * prod_{i in S} p_i / (1 - p_i)
    no_fault = np.prod([1 - p for p, _ in error_mechanisms])
    outcomes = {} # cache: flip mask of fault set -> (wrongly extracted syndrome, syndrome correction failed because introduced log. operation)
    total, wrong, failed = 0.0, 0.0, 0.0

    for weight in range(max_weight + 1):
        for faults in combinations(range(len(error_mechanisms)), weight):
            prob, mask = no_fault, 0
            for i in faults:
                p, m = error_mechanisms[i]
                prob *= p / (1 - p)
                mask ^= m

            if mask not in outcomes:
                correction = decoder.decode(np.array([(mask >> j) & 1 for j in range(g)], dtype=np.uint8))
                outcomes[mask] = (
                    (mask >> g) % (1 << g) != 0,
                    any(((mask >> (2 * g + k)) & 1) ^ (not correction.commutes(op)) for k, op in enumerate(targets))
                )

            syn, fail = outcomes[mask]
            total += prob
            wrong += prob * syn
            failed += prob * fail

    # not enumerated probability mass (fault sets with > max_weight faults) --> can be wrong / right , failed / successful --> worst/best case bounds
    tail = 1 - total

    return {
        "n_mechanisms": len(error_mechanisms),
        "truncation_tail": tail,
        "syndrome_error_rate": wrong / total,
        "syndrome_error_rate_lower": wrong,
        "syndrome_error_rate_upper": wrong + tail,
        "logical_error_rate": failed / total,
        "logical_error_rate_lower": failed,
        "logical_error_rate_upper": failed + tail,
    }


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_DETECTOR_ERROR_MODEL : dict[CircuitFunction, Callable[[QECCircuit, stim.Circuit, Decoder, int], dict]] = {
    "StatePrep": run_dem_analysis_sp,
    "Encoding": run_dem_analysis_en,
    "Syndrome": run_dem_analysis_se
}
