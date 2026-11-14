
"""Run the model accuracy experiments by comparing hardware events of different circuits on the abstract executions and actual compilations."""

import csv
import shutil
from collections import Counter
from pathlib import Path

import stim
import yaml
from collections.abc import Callable

from src.models.hardware import HardwareModel, HardwareEvents, PlatformTag

from benchmarks.common_reference.compilation import  _RUN_COMPILER 
from benchmarks.common_reference.abstract_execution import _RUN_ABSTRACTION, _RUN_BASELINE
from benchmarks.common_reference.architecture_preparation import _PREPARE_HARDWARE
from benchmarks.model_accuracy.visualization import visualize

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "model_accuracy" / "configs"
_CIRCUIT_DIR = Path(__file__).resolve().parents[2] / "data" / "model_accuracy" / "circuits"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "model_accuracy"


def run(platforms: list[PlatformTag] | None = None, max_qubits: int | None = None, max_compile_time: float = 600.0, clean: bool = False,
        append: bool = False, only_visual: bool = False) -> None:
    if only_visual:
        visualize(platforms or ["TI", "NA", "SC"], _RESULTS_DIR, max_qubits)
        return

    # empty the raw result directories of the platforms that are run (so only results of this run remain)
    if clean:
        for platform in platforms or ["TI", "NA", "SC"]:
            raw_dir = _RESULTS_DIR / platform.lower() / "raw"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
                print(f"Emptied {raw_dir}...")

    # collect different circuits
    circuits = {}
    for circuit_file in sorted(_CIRCUIT_DIR.glob("*.stim")):
        name = circuit_file.stem
        circuit = stim.Circuit.from_file(circuit_file)
        if max_qubits is None or circuit.num_qubits <= max_qubits:
            circuits[name] = circuit
    print(f"Collected {len(circuits)} circuits...")

    # collect different hardware configs
    configs = []
    for config_file in sorted(_CONFIG_DIR.glob("*.yaml")):
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
                raw_file = _RESULTS_DIR / config["platform"].lower() / "raw" / f"{config['name']}__{circuit_name}__{compiler}.csv"
                if append and raw_file.exists():
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
            abstract_hardware_events, abstraction_details = _RUN_ABSTRACTION[config["platform"]](hardware=abstract_hardware, circuit=circuit)

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
    
    # evaluate the results
    visualize(platforms or ["TI", "NA", "SC"], _RESULTS_DIR, max_qubits)


def save_raw_results(file: Path, metadata: dict, anchors: dict[int, int], abstract: tuple, compiled: tuple, baseline: tuple) -> None:
    """Store one comparison (one hardware config, one circuit) as CSV: the metadata, then source, metric, qubit, anchor, count, value.
    Sources: the circuit, the abstract model, the compiler and the naive per-gate model ("per_gate").

    - circuit anchors: one row per qubit (source "circuit"), with its number of anchor operations in `value`
    - event lists (e.g. interaction_zone_entry): one row per event, in list (= chronological) order, count 1
    - {(qubit, anchor): count} details (e.g. shuttles, swaps): one row per entry
    - other details (e.g. wall_clock_s): one row with the number in `value`
    """
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare abstract executions and compilations of circuits on hardware configs.")
    parser.add_argument("--platform", nargs="+", choices=["TI", "NA", "SC"], default=None,
                        help="only run configs of these platforms (default: all)")
    parser.add_argument("--max-qubits", type=int, default=None, help="skip circuits with more qubits (default: no limit)")
    parser.add_argument("--max-compile-time", type=float, default=600.0, help="time limit per compilation in seconds (default: 600)")
    parser.add_argument("--clean", action="store_true",
                        help="delete all files in the raw result directories of the run platforms before running")
    parser.add_argument("--append", action="store_true",
                        help="skip circuit/config combinations whose raw result file already exists")
    parser.add_argument("--only-visual", action="store_true",
                        help="skip running, only visualize the existing raw results of the selected platforms")
    args = parser.parse_args()

    run(platforms=args.platform, max_qubits=args.max_qubits, max_compile_time=args.max_compile_time, clean=args.clean,
        append=args.append, only_visual=args.only_visual)

