"""Validate the static efficiency metrics against the resource costs of actual compilations."""

import csv
from collections import Counter
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
_REUSE_DIR = Path(__file__).resolve().parents[2] / "results" / "model_accuracy"  # compilations of the model accuracy benchmark


def run(platforms: list[PlatformTag] | None = None, max_qubits: int | None = None, max_compile_time: float = 600.0, append: bool = False,
        only_reuse: bool = False, only_visual: bool = False) -> None:
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

    # run the structural analysis once for each hardware config and circuit, and the compilation for each compiler (or reuse it)
    for config in configs:
        platform = config["platform"]
        for circuit_name, (group, circuit) in circuits.items():
            abstract_hardware, concrete_hardwares = _PREPARE_HARDWARE[platform](parameters=config["architecture"], num_qubits=circuit.num_qubits)
            metrics, _ = _RUN_STRUCTURAL_ANALYSIS[platform](hardware=abstract_hardware, circuit=circuit)
            workload = {"N_gate": metrics["nr_gates"],
                        "N_2q": sum(len(op.target_groups()) for op in circuit if stim.gate_data(op.name).is_two_qubit_gate),
                        "N_qop": sum(len(group) for op in circuit if op.name != "QUBIT_COORDS" for group in op.target_groups())}

            for compiler, run_compiler in _RUN_COMPILER[platform].items():
                raw_file = _RESULTS_DIR / "raw" / platform.lower() / f"{config['name']}__{circuit_name}__{compiler}.csv"
                reuse_file = _REUSE_DIR / platform.lower() / "raw" / f"{config['name']}__{circuit_name}__{compiler}.csv"
                if append and raw_file.exists():
                    continue

                totals = Counter()
                if reuse_file.exists():
                    with reuse_file.open(newline="") as f:
                        for row in csv.DictReader(f):
                            if row["source"] == "compiled" and row["count"]:
                                totals[row["metric"]] += float(row["count"])
                    status = "reused"
                elif compiler not in concrete_hardwares:
                    status = "unsupported"
                elif only_reuse:
                    continue
                else:
                    print(f"[{platform}] Running compilation of {circuit_name} on ({config['name']}) with {compiler}...")
                    try:
                        events, details = run_compiler(hardware=concrete_hardwares[compiler], circuit=circuit, max_compile_time=max_compile_time)
                        totals.update({name: len(events_list) for name, events_list in vars(events).items()})
                        totals.update({name: sum(value.values()) for name, value in details.items() if isinstance(value, dict)})
                        status = "success"
                    except Exception as error:  # IonShuttler (linear) reports its timeout as a RuntimeError
                        status = "timeout" if isinstance(error, TimeoutError) or "TIMEOUT" in str(error) else "error"
                        print(f"Failed {config['name']} / {circuit_name} / {compiler}: {error}")

                # compiled costs: moves per 2q gate (SC swaps are recorded for both qubits) and zone changes per qubit operation
                targets = {"routing": {"TI": totals["shuttles"], "NA": totals["transports"], "SC": totals["swaps"] / 2}[platform] / workload["N_2q"]}
                if platform != "SC":
                    zones = totals["interaction_zone_entry"] + totals["interaction_zone_eviction"] if platform == "TI" else totals["zone_changes"]
                    targets["zones"] = zones / workload["N_qop"]
                metadata = {"config": config["name"], "platform": platform, "compiler": compiler, "circuit": circuit_name, "group": group,
                            "num_qubits": circuit.num_qubits}
                save_raw_results(raw_file, metadata, status, workload, metrics, targets if status in ("success", "reused") else {})

    # evaluate the results
    visualize(_RESULTS_DIR)


def save_raw_results(file: Path, metadata: dict, status: str, workload: dict, metrics: dict, targets: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*metadata, "source", "metric", "value"])
        writer.writerow([*metadata.values(), "run", "status", status])
        for source, values in {"circuit": workload, "structural": metrics, "compiled": targets}.items():
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
    parser.add_argument("--only-reuse", action="store_true", help="do not compile, only reuse the compilations of the model accuracy benchmark")
    parser.add_argument("--only-visual", action="store_true", help="skip running, only visualize the existing raw results")
    args = parser.parse_args()

    run(platforms=args.platform, max_qubits=args.max_qubits, max_compile_time=args.max_compile_time, append=args.append,
        only_reuse=args.only_reuse, only_visual=args.only_visual)
