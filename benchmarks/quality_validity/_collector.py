"""Collect analysis results from the Monte-Carlo-Sampling-based approach and the Detector-Error-Model-based approach."""

import csv
from pathlib import Path

import stim
import yaml

from src.circuits.circuit import QECCircuit, CircuitFunction

from benchmarks._common_reference.ft_mc_analysis import _RUN_MONTE_CARLO
from benchmarks._common_reference.ft_dem_analysis import _RUN_DETECTOR_ERROR_MODEL
from benchmarks._common_reference.circuit_preparation import _PREPARE_CIRCUIT

def collect(functions: list[CircuitFunction], circuit_dir: Path, results_dir: Path, max_qubits: int) -> None:
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

    # run the analyses for each circuit
    for function in functions:
        for circuit_name, (circuit_file, circuit) in circuits[function].items():
            raw_file = results_dir / "raw" / function.lower() / f"__{circuit_name}.csv"
            if raw_file.exists():
                print(f"[{function}] Skipping {circuit_name}: raw result already exists")
                continue
            
            qec_circuit = _PREPARE_CIRCUIT[function](circuit_file, circuit)

            print(f"[{function}] Running monte carlo analysis of {circuit_name}...")
            mc_metrics = _RUN_MONTE_CARLO[function](circuit=qec_circuit)

            print(f"[{function}] Running detector error model analysis of {circuit_name}...")
            dem_metrics = _RUN_DETECTOR_ERROR_MODEL[function](circuit=qec_circuit)

            save_raw_results(raw_file, qec_circuit, mc_metrics, dem_metrics)


def save_raw_results(file: Path, circuit: QECCircuit, mc_metrics: dict, dem_metrics: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
