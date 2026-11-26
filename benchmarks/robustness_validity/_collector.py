"""Collect analysis results (and run times) from the Detector-Error-Model-based approach and the Monte-Carlo-Sampling-based reference."""

import csv
import time
from collections import Counter
from pathlib import Path

import stim
import yaml

from src.circuits.circuit import QECCircuit, CircuitFunction
from src.models.hardware import HardwareModel
from src.analysis.robustness._decoder import BeliefPropOrdStat

from benchmarks._common_reference.ft_mc_analysis import _RUN_MONTE_CARLO
from benchmarks._common_reference.ft_dem_analysis import _RUN_DETECTOR_ERROR_MODEL
from benchmarks._common_reference.circuit_preparation import _PREPARE_CIRCUIT
from benchmarks._common_reference.architecture_preparation import _PREPARE_HARDWARE
from benchmarks._common_reference.abstract_execution import _RUN_ABSTRACTION

def collect(functions: list[CircuitFunction], circuit_dir: Path, config_dir: Path, results_dir: Path, max_qubits: int, max_shots: int, batch_size: int, max_rel_ci: float, max_rel_tail: float, max_fault_sets: int) -> None:
    # collect different circuits (with their task group folder)
    circuits : dict[CircuitFunction, dict[str, tuple[Path, stim.Circuit]]] = {"StatePrep": {}, "Encoding": {}, "Syndrome": {}}
    for circuit_file in sorted(circuit_dir.glob("*/*/*.stim")):
        circuit = stim.Circuit.from_file(circuit_file)

        if circuit.num_qubits <= max_qubits:
            task = circuit_file.parent.name
            if task == "syndrome_extraction":
                function = "Syndrome"
            elif task == "encoding":
                function = "Encoding"
                if circuit_file.parent.parent.name in ["9-1-3", "16-1-4-bacon-shor", "49-9-4"]: # subsystem codes
                    continue
            elif task.endswith("preparation"):
                function = "StatePrep"
            else:
                continue
            circuits[function][circuit_file.stem] = (circuit_file, circuit)
    print(f"Collected {sum(len(c) for c in circuits.values())} circuits...")

    # collect different hardware configs
    configs = []
    for config_file in sorted(config_dir.glob("*.yaml")):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        configs.append(config)
    print(f"Collected {len(configs)} hardware configurations...")

    # run the analyses for each circuit
    for function in functions:
        for circuit_name, (circuit_file, circuit) in circuits[function].items():

            qec_circuit = _PREPARE_CIRCUIT[function](circuit_file, circuit)

            for config in configs:
                raw_file = results_dir / "raw" / function.lower() / f"{config['name']}__{circuit_name}.csv"
                if raw_file.exists():
                    print(f"[{function}] Skipping {circuit_name} on {config['name']}: raw result already exists")
                    continue

                abstract_hardware, _ = _PREPARE_HARDWARE[config["platform"]](parameters=config["architecture"], num_qubits=circuit.num_qubits)
                _, _, noisy_circuit = _RUN_ABSTRACTION[config["platform"]](hardware=abstract_hardware, circuit=circuit)

                decoder = BeliefPropOrdStat(abstract_hardware, qec_circuit.stabilizers)

                # encoders: analyze the real noisy circuit (noisy input) and a version without noise on the input qubits before their first gate (clean input)
                variants = {function: noisy_circuit}
                if function == "Encoding":
                    clean_input_circuit = stim.Circuit()
                    started = set()
                    for op in noisy_circuit.flattened():
                        gate = stim.gate_data(op.name)
                        if gate.is_noisy_gate and not gate.produces_measurements:
                            targets = op.targets_copy()
                            step = 2 if gate.is_two_qubit_gate else 1
                            kept = []
                            for i in range(0, len(targets), step):
                                if not any(t.value in qec_circuit.logical_qubits and t.value not in started for t in targets[i:i + step]):
                                    kept += targets[i:i + step]
                            if kept:
                                clean_input_circuit.append(op.name, kept, op.gate_args_copy(), tag=op.tag)
                        else:
                            clean_input_circuit.append(op)
                            started |= {t.value for t in op.targets_copy() if t.is_qubit_target}
                    variants["EncodingCleanInput"] = clean_input_circuit

                for variant, variant_circuit in variants.items():
                    print(f"[{variant}] Running monte carlo analysis of noisy {circuit_name} on {config["name"]}...")
                    mc_metrics = _RUN_MONTE_CARLO[function](circuit_data=qec_circuit, noisy_circuit=variant_circuit, decoder=decoder, max_shots=max_shots, batch_size=batch_size, max_rel_ci=max_rel_ci)

                    print(f"[{variant}] Running detector error model analysis of noisy {circuit_name} on {config["name"]}...")
                    dem_metrics = _RUN_DETECTOR_ERROR_MODEL[function](circuit_data=qec_circuit, noisy_circuit=variant_circuit, decoder=decoder, max_rel_tail=max_rel_tail, max_fault_sets=max_fault_sets)

                    save_raw_results(results_dir / "raw" / variant.lower() / f"{config['name']}__{circuit_name}.csv", circuit_file, qec_circuit, variant, config, max_shots, batch_size, max_rel_ci, max_rel_tail, max_fault_sets, mc_metrics, dem_metrics)


def save_raw_results(file: Path, circuit_file: Path, circuit: QECCircuit, function: str, hw_config: dict,
                     max_shots: int, batch_size: int, max_rel_ci: float, max_rel_tail: float, max_fault_sets: int, mc_metrics: dict, dem_metrics: dict) -> None:
    if not circuit.verification: # nothing to reject: rejection rate trivially 0 for both methods, not saved
        mc_metrics = {k: v for k, v in mc_metrics.items() if not k.startswith("rejection_rate")}
        dem_metrics = {k: v for k, v in dem_metrics.items() if not k.startswith("rejection_rate")}

    row = {
        # circuit
        "circuit": circuit_file.stem,
        "code": circuit_file.parent.parent.name,
        "function": function, # circuit function, or "EncodingCleanInput" for encoders without noise on the input before their first gate
        "state": circuit.quantum_state,
        "ft": "_ft_preparation" in circuit_file.parent.name and "non_ft" not in circuit_file.parent.name, # fault-tolerant state preparation (task folder)
        "num_qubits": circuit.circuit.num_qubits,
        "num_logical_qubits": len(circuit.logical_operators),
        "num_stabilizers": len(circuit.stabilizers),
        "num_gates": sum(len(op.targets_copy()) // (2 if stim.gate_data(op.name).is_two_qubit_gate else 1) for op in circuit.circuit), # 2-qubit gates count once per pair
        # hardware config
        "config": hw_config["name"],
        "platform": hw_config["platform"],
        # analyses
        "max_shots": max_shots,
        "batch_size": batch_size,
        "max_rel_ci": max_rel_ci,
        "mc_shots": mc_metrics["shots"], # dynamically reached number of shots of the MC
        "max_rel_tail": max_rel_tail,
        "max_fault_sets": max_fault_sets,
        "max_weight": dem_metrics["max_weight"], # dynamically reached truncation weight of the DEM
        **{f"{method}_{metric}": metrics.get(metric) for method, metrics in [("mc", mc_metrics), ("dem", dem_metrics)]
           for metric in ["logical_error_rate", "logical_error_rate_ci_lower", "logical_error_rate_ci_upper", "logical_error_rate_lower", "logical_error_rate_upper",
                          "rejection_rate", "rejection_rate_ci_lower", "rejection_rate_ci_upper", "rejection_rate_lower", "rejection_rate_upper",
                          "syndrome_error_rate", "syndrome_error_rate_ci_lower", "syndrome_error_rate_ci_upper", "syndrome_error_rate_lower", "syndrome_error_rate_upper", "truncation_tail", "n_mechanisms", "wall_clock_s"]},
    }

    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
