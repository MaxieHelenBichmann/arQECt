"""Collect the hardware events and static properties of different circuits by running static analyses and actual compilations on them."""

import csv
from pathlib import Path

import stim
import yaml

from src.models.hardware import PlatformTag

from benchmarks.common_reference.compilation import _RUN_COMPILER
from benchmarks.common_reference.structural_analysis import _RUN_STRUCTURAL_ANALYSIS
from benchmarks.common_reference.architecture_preparation import _PREPARE_HARDWARE
from benchmarks.static_metric_validation._visualizer import visualize


def collect(platforms: list[str], circuit_dir: Path, config_dir: Path, results_dir: Path, max_qubits: int, max_compile_time: int) -> None:
    # collect different circuits (with their task group folder)
    circuits = {}
    for circuit_file in sorted(circuit_dir.glob("*/*/*.stim")):
        circuit = stim.Circuit.from_file(circuit_file)
        if circuit.num_qubits <= max_qubits:
            circuits[circuit_file.stem] = (circuit_file.parent.relative_to(circuit_dir).as_posix(), circuit)
    print(f"Collected {len(circuits)} circuits...")

    # collect different hardware configs
    configs = []
    for config_file in sorted(config_dir.glob("*.yaml")):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        if platforms is None or config["platform"] in platforms:
            configs.append(config)
    print(f"Collected {len(configs)} hardware configurations...")

    # run the structural analysis once for each hardware config and circuit, and the compilation for each compiler
    for config in configs:
        platform = config["platform"]
        for circuit_name, (group, circuit) in circuits.items():
            compilers = {}
            for compiler, run_compiler in _RUN_COMPILER[platform].items():
                raw_file = results_dir / "raw" / platform.lower() / f"{config['name']}__{circuit_name}__{compiler}.csv"
                if raw_file.exists():
                    print(f"[{platform}] Skipping {circuit_name} on ({config['name']}) with {compiler}: raw result already exists")
                else:
                    compilers[compiler] = (run_compiler, raw_file)
            if not compilers:
                continue

            abstract_hardware, concrete_hardwares = _PREPARE_HARDWARE[platform](parameters=config["architecture"], num_qubits=circuit.num_qubits)

            print(f"[{platform}] Running structural analysis of {circuit_name} on ({config['name']})...")
            metrics, _ = _RUN_STRUCTURAL_ANALYSIS[platform](hardware=abstract_hardware, circuit=circuit)

            for compiler, (run_compiler, raw_file) in compilers.items():
                if compiler not in concrete_hardwares:
                    print(f"Skipped {config['name']} / {circuit_name} / {compiler}: architecture not supported by {compiler}")
                    continue
                concrete_hardware = concrete_hardwares[compiler]
                metadata = {"config": config["name"], "platform": platform, "compiler": compiler, "circuit": circuit_name, "group": group,
                            "num_qubits": circuit.num_qubits,
                            "N_2q": sum(len(op.target_groups()) for op in circuit if stim.gate_data(op.name).is_two_qubit_gate)}

                print(f"[{platform}] Running compilation of     {circuit_name} on ({config['name']}) with {compiler}...")
                try:
                    events, details = run_compiler(hardware=concrete_hardware, circuit=circuit, max_compile_time=max_compile_time)
                except (RuntimeError, TimeoutError) as error:
                    print(f"Skipped {config['name']} / {circuit_name} / {compiler}: {error}")
                    continue

                # relevant values = aggregated number of each event and detail
                compiled = {name: len(locations) for name, locations in vars(events).items()}
                compiled.update({name: sum(value.values()) if isinstance(value, dict) else value for name, value in details.items()})

                save_raw_results(raw_file, metadata, metrics, compiled)


def save_raw_results(file: Path, metadata: dict, metrics: dict, compiled: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*metadata, "source", "metric", "value"])
        for source, values in {"structural": metrics, "compiled": compiled}.items():
            for metric, value in values.items():
                writer.writerow([*metadata.values(), source, metric, "" if value is None else value])
