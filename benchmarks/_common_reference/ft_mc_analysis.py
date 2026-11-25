
from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim
import numpy as np

from src.analysis.quality.encoder import EncoderCircuitAnalyzer
from src.analysis.quality.state_prep import StatePreparationCircuitAnalyzer
from src.analysis.quality.syndrome_extract import SyndromeExtractionAnalyzer
from src.models.hardware import HardwareModel
from src.analysis.quality._decoder import Decoder


def _confidence_interval(successes: int, trials: int) -> tuple[float | None, float | None]: # 95% Wilson score CI
    z = 1.96
    if trials:
        center = (successes / trials + z**2 / (2 * trials)) / (1 + z**2 / trials)
        half = z / (1 + z**2 / trials) * (successes / trials * (1 - successes / trials) / trials + z**2 / (4 * trials**2)) ** 0.5
        return center - half, center + half
    return None, None

def run_mc_analysis_sp(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_shots: int) -> dict:
    result : dict[str, object] = {}

    match circuit_data.quantum_state:
        case "0":
            target_logicals = [z for (_,z) in circuit_data.logical_operators]
        case "+":
            target_logicals = [x for (x,_) in circuit_data.logical_operators]
        case "1":
            target_logicals = [-z for (_,z) in circuit_data.logical_operators]
        case "-":
            target_logicals = [-x for (x,_) in circuit_data.logical_operators]

    # run experiment with multiple shots
    experiment = noisy_circuit.copy()
    existent_measurements = experiment.num_measurements

    for observable in [*circuit_data.stabilizers, *target_logicals]:
        experiment.append("MPP", list(stim.target_combined_paulis(observable)))

    samples = experiment.compile_sampler().sample(max_shots)

    # discard results that verification does not accept (Protocol B)
    accepted_mask = np.ones(max_shots, dtype=bool)

    for index, expected in circuit_data.verification.items():
        accepted_mask &= samples[:, index] == expected

    accepted = samples[accepted_mask]
    rejected = max_shots - len(accepted)
    result["rejection_rate"] = rejected / max_shots
    result["rejection_rate_ci_lower"], result["rejection_rate_ci_upper"] = _confidence_interval(rejected, max_shots)

    # decode and count logical failures
    g = len(circuit_data.stabilizers)

    syndromes = accepted[:, existent_measurements:existent_measurements + g]
    logicals = accepted[:, existent_measurements + g:]

    failures = 0
    for syndrome, measurements in zip(syndromes, logicals, strict=True):
        correction = decoder.decode(syndrome)

        # recovery action -> logical error?
        corrected = [
            bool(m) ^ (not correction.commutes(op)) for m, op in zip(measurements, target_logicals, strict=True)
        ]
        failures += int(any(corrected))

    result["logical_error_rate"] = failures / len(accepted) if len(accepted) else None
    result["logical_error_rate_ci_lower"], result["logical_error_rate_ci_upper"] = _confidence_interval(failures, len(accepted))

    return result


def run_mc_analysis_en(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_shots: int) -> dict:
    result : dict[str, object] = {}
    
    # entangle each input qubit with a noiseless reference qubit (Bell pair)
    reference = max(len(circuit_data.stabilizers[0]), noisy_circuit.num_qubits) # first reference qubit index, after all circuit qubits

    experiment = stim.Circuit()
    for i, q in enumerate(circuit_data.logical_qubits):
        experiment.append("H", [reference + i])
        experiment.append("CX", [reference + i, q])

    # run experiment with multiple shots
    experiment += noisy_circuit
    existent_measurements = experiment.num_measurements

    targets = []
    correlators = []
    for i, (x, z) in enumerate(circuit_data.logical_operators):
        targets += [x, z]
        correlators += [x * stim.PauliString("_" * (reference + i) + "X"), z * stim.PauliString("_" * (reference + i) + "Z")]

    for observable in [*circuit_data.stabilizers, *correlators]:
        experiment.append("MPP", list(stim.target_combined_paulis(observable)))

    samples = experiment.compile_sampler().sample(max_shots)

    # decode and count logical failures
    g = len(circuit_data.stabilizers)

    syndromes = samples[:, existent_measurements:existent_measurements + g]
    logicals = samples[:, existent_measurements + g:]

    failures = 0
    for syndrome, measurements in zip(syndromes, logicals, strict=True):
        correction = decoder.decode(syndrome)

        # recovery action -> logical error?
        corrected = [
            bool(m) ^ (not correction.commutes(op)) for m, op in zip(measurements, targets, strict=True)
        ]
        failures += int(any(corrected))

    result["logical_error_rate"] = failures / max_shots
    result["logical_error_rate_ci_lower"], result["logical_error_rate_ci_upper"] = _confidence_interval(failures, max_shots)

    return result

def run_mc_analysis_se(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_shots: int) -> dict:
    result : dict[str, object] = {}
    
    reference = max(len(circuit_data.stabilizers[0]), noisy_circuit.num_qubits) # first reference qubit index, after all circuit qubits

    targets = []
    correlators = []
    for i, (x, z) in enumerate(circuit_data.logical_operators):
        targets += [x, z]
        correlators += [x * stim.PauliString("_" * (reference + i) + "X"), z * stim.PauliString("_" * (reference + i) + "Z")]

    observables = [*circuit_data.stabilizers, *correlators]

    # ideal input state: noiseless projection onto stabilizers + logical Bell pairs with reference qubits
    experiment = stim.Circuit()
    for observable in observables:
        experiment.append("MPP", list(stim.target_combined_paulis(observable)))
    existent_measurements = experiment.num_measurements

    # run experiment with multiple shots
    experiment += noisy_circuit
    final_measurements = experiment.num_measurements

    for observable in observables:
        experiment.append("MPP", list(stim.target_combined_paulis(observable)))

    samples = experiment.compile_sampler().sample(max_shots)

    # compare syndrome extracted by the circuit with the initial perfect syndrome
    g = len(circuit_data.stabilizers)

    initial = samples[:, :existent_measurements]
    extracted = samples[:, [existent_measurements + i for i in circuit_data.syndrome_measurements]]
    wrong = int((extracted != initial[:, :g]).any(axis=1).sum())
    result["syndrome_error_rate"] = wrong / max_shots
    result["syndrome_error_rate_ci_lower"],result["syndrome_error_rate_ci_upper"] = _confidence_interval(wrong, max_shots)

    # decode and count logical failures (differences between initial and final perfect measurements)
    difference = initial ^ samples[:, final_measurements:]
    syndromes = difference[:, :g]
    logicals = difference[:, g:]

    failures = 0
    for syndrome, measurements in zip(syndromes, logicals, strict=True):
        correction = decoder.decode(syndrome)

        # recovery action -> logical error?
        corrected = [
            bool(m) ^ (not correction.commutes(op)) for m, op in zip(measurements, targets, strict=True)
        ]
        failures += int(any(corrected))

    result["logical_error_rate"] = failures / max_shots
    result["logical_error_rate_ci_lower"], result["logical_error_rate_ci_upper"] = _confidence_interval(failures, max_shots)

    return result


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_MONTE_CARLO : dict[CircuitFunction, Callable[[QECCircuit, stim.Circuit, Decoder, int], dict]] = {
    "StatePrep": run_mc_analysis_sp,
    "Encoding": run_mc_analysis_en,
    "Syndrome": run_mc_analysis_se
}