"""Validate the static efficiency metrics against the runtime resource costs (not compilation costs) of actual compilations."""

import csv
from pathlib import Path

import stim
import yaml

from src.models.hardware import PlatformTag

from benchmarks.common_reference.compilation import _RUN_COMPILER
from benchmarks.common_reference.structural_analysis import _RUN_STRUCTURAL_ANALYSIS
from benchmarks.common_reference.architecture_preparation import _PREPARE_HARDWARE
from benchmarks.static_metric_validation.visualization import visualize

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "configs"
_CIRCUIT_DIR = Path(__file__).resolve().parents[2] / "data" / "circuits"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "static_metric_validation"


def run(platforms: list[PlatformTag] | None = None, max_qubits: int | None = None, max_compile_time: float = 600.0, append: bool = False,
        only_visual: bool = False) -> None:
    if only_visual:
        visualize(_RESULTS_DIR)
        return

    # collect different circuits (with their task group folder)
    circuits = {}
    for circuit_file in sorted(_CIRCUIT_DIR.glob("*/*/*.stim")):
        circuit = stim.Circuit.from_file(circuit_file)
        if max_qubits is None or circuit.num_qubits <= max_qubits:
            circuits[circuit_file.stem] = (circuit_file.parent.relative_to(_CIRCUIT_DIR).as_posix(), circuit)
    print(f"Collected {len(circuits)} circuits...")

    # collect different hardware configs
    configs = []
    for config_file in sorted(_CONFIG_DIR.glob("*.yaml")):
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
                raw_file = _RESULTS_DIR / "raw" / platform.lower() / f"{config['name']}__{circuit_name}__{compiler}.csv"
                if append and raw_file.exists():
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

    # evaluate the results
    visualize(_RESULTS_DIR)


def save_raw_results(file: Path, metadata: dict, metrics: dict, compiled: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*metadata, "source", "metric", "value"])
        for source, values in {"structural": metrics, "compiled": compiled}.items():
            for metric, value in values.items():
                writer.writerow([*metadata.values(), source, metric, "" if value is None else value])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the static efficiency metrics against compiled resource costs.")
    parser.add_argument("--platform", nargs="+", choices=["TI", "NA", "SC"], default=None,
                        help="only run configs of these platforms (default: all)")
    parser.add_argument("--max-qubits", type=int, default=None, help="skip circuits with more qubits (default: no limit)")
    parser.add_argument("--max-compile-time", type=float, default=600.0, help="time limit per compilation in seconds (default: 600)")
    parser.add_argument("--append", action="store_true", help="skip circuit/config/compiler combinations whose raw result file already exists")
    parser.add_argument("--only-visual", action="store_true", help="skip running, only visualize the existing raw results")
    args = parser.parse_args()

    run(platforms=args.platform, max_qubits=args.max_qubits, max_compile_time=args.max_compile_time, append=args.append,
        only_visual=args.only_visual)
