"""Collect the hardware events of different circuits by running abstract executions and actual compilations on them."""

import csv
from collections import Counter
from pathlib import Path

import stim
import yaml

from benchmarks._common_reference.compilation import  _RUN_COMPILER 
from benchmarks._common_reference.abstract_execution import _RUN_ABSTRACTION, _RUN_BASELINE
from benchmarks._common_reference.architecture_preparation import _PREPARE_HARDWARE


def collect(platforms: list[str], circuit_dir: Path, config_dir: Path, results_dir: Path, max_qubits: int, max_compile_time: int) -> None:
    # collect different circuits (flattened from data/circuits/<code family>/<task>/)
    circuits = {}
    for circuit_file in sorted(circuit_dir.glob("*/*/*.stim")):
        name = circuit_file.stem
        circuit = stim.Circuit.from_file(circuit_file)
        if circuit.num_qubits <= max_qubits:
            circuits[name] = circuit
    print(f"Collected {len(circuits)} circuits...")

    # collect different hardware configs
    configs = []
    for config_file in sorted(config_dir.glob("*.yaml")):
        with open(config_file) as f:
                config = yaml.safe_load(f)
        if platforms is None or config["platform"] in platforms:
            configs.append(config)
    print(f"Collected {len(configs)} hardware configurations...")

    # run baseline and my abstract execution once for each hardware config and circuit, and the compilation for each compiler
    for config in configs:
        for circuit_name, circuit in circuits.items():
            compilers = {}
            for compiler, run_compiler in _RUN_COMPILER[config["platform"]].items():
                raw_file = results_dir / config["platform"].lower() / "raw" / f"{config['name']}__{circuit_name}__{compiler}.csv"
                if raw_file.exists():
                    print(f"[{config["platform"]}] Skipping {circuit_name} on ({config["name"]}) with {compiler}: raw result already exists")
                else:
                    compilers[compiler] = (run_compiler, raw_file)
            if not compilers:
                continue

            anchors = Counter(t.value for op in circuit for group in op.target_groups() for t in group
                                if stim.gate_data(op.name).is_two_qubit_gate or stim.gate_data(op.name).produces_measurements
                                or stim.gate_data(op.name).is_reset)

            abstract_hardware, concrete_hardwares = _PREPARE_HARDWARE[config["platform"]](parameters=config["architecture"], num_qubits=circuit.num_qubits)

            print(f"[{config["platform"]}] Running baseline execution of {circuit_name} on ({config["name"]})...")
            baseline_hardware_events, baseline_details = _RUN_BASELINE[config["platform"]](circuit)
            print(f"[{config["platform"]}] Running abstract execution of {circuit_name} on ({config["name"]})...")
            abstract_hardware_events, abstraction_details, _ = _RUN_ABSTRACTION[config["platform"]](hardware=abstract_hardware, circuit=circuit)

            for compiler, (run_compiler, raw_file) in compilers.items():
                if compiler not in concrete_hardwares:
                    print(f"Skipped {config['name']} / {circuit_name} / {compiler}: architecture not supported by {compiler}")
                    continue
                concrete_hardware = concrete_hardwares[compiler]

                print(f"[{config["platform"]}] Running compilation of        {circuit_name} on ({config["name"]}) with {compiler}...")
                try:
                    concrete_hardware_events, compiler_details = run_compiler(hardware=concrete_hardware, circuit=circuit, max_compile_time=max_compile_time)
                except (RuntimeError, TimeoutError) as error:
                    print(f"Skipped {config['name']} / {circuit_name} / {compiler}: {error}")
                    continue

                metadata = {"config": config["name"], "platform": config["platform"], "compiler": compiler, **config["architecture"],
                            "circuit": circuit_name, "num_qubits": circuit.num_qubits}
                save_raw_results(raw_file, metadata,
                                    anchors={qubit: anchors[qubit] for qubit in range(circuit.num_qubits)},
                                    abstract=(abstract_hardware_events, abstraction_details),
                                    compiled=(concrete_hardware_events, compiler_details),
                                    baseline=(baseline_hardware_events, baseline_details))
                
def save_raw_results(file: Path, metadata: dict, anchors: dict[int, int], abstract: tuple, compiled: tuple, baseline: tuple) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*metadata, "source", "metric", "qubit", "anchor", "count", "value"])
        for qubit, num_anchors in anchors.items():
            writer.writerow([*metadata.values(), "circuit", "anchors", qubit, "", "", num_anchors])
        for source, (events, details) in {"abstract": abstract, "compiled": compiled, "per_gate": baseline}.items():
            for metric, locations in vars(events).items():
                for location in locations:  # a location, or a pair of locations (SC virtual_swap: one row per swapped qubit)
                    for qubit, anchor in (location if isinstance(location[0], tuple) else [location]):
                        writer.writerow([*metadata.values(), source, metric, qubit, anchor, 1, ""])
            for metric, value in details.items():
                if isinstance(value, dict):
                    for (qubit, anchor), count in value.items():
                        writer.writerow([*metadata.values(), source, metric, qubit, anchor, count, ""])
                else:
                    writer.writerow([*metadata.values(), source, metric, "", "", "", value])
