
"""Run the model accuracy experiments by comparing hardware events of different circuits on the abstract executions and actual compilations."""

import csv
import shutil
from collections import Counter
from pathlib import Path

import stim
import yaml

from benchmarks.model_accuracy.compilation import run_ti_compiler, run_na_compiler, run_sc_compiler, run_ti_abstraction
from benchmarks.model_accuracy.architecture_preparation import prepare_ti_hardware, prepare_na_hardware, prepare_sc_hardware
from benchmarks.model_accuracy.visualization import visualize

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "model_accuracy" / "configs"
_CIRCUIT_DIR = Path(__file__).resolve().parents[2] / "data" / "model_accuracy" / "circuits"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "model_accuracy"

def run(platforms: list[str] | None = None, circuit_pattern: str = "*", config_pattern: str = "*",
        max_qubits: int | None = None, max_compile_time: float = 600.0, visualize_true: bool = True, clean: bool = False) -> None:
    # empty the raw result directories of the platforms that are run (so only results of this run remain)
    if clean:
        for platform in platforms or ["TI", "NA", "SC"]:
            raw_dir = _RESULTS_DIR / "raw" / platform.lower()
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
                print(f"Emptied {raw_dir}...")

    # collect different circuits
    circuits = {}
    for circuit_file in sorted(_CIRCUIT_DIR.glob(f"{circuit_pattern}.stim")):
        name = circuit_file.stem
        circuit = stim.Circuit.from_file(circuit_file)
        if max_qubits is None or circuit.num_qubits <= max_qubits:
            circuits[name] = circuit
    print(f"Collected {len(circuits)} circuits...")

    # collect different hardware configs
    configs = []
    for config_file in sorted(_CONFIG_DIR.glob(f"{config_pattern}.yaml")):
        with open(config_file) as f:
                config = yaml.safe_load(f)
        if platforms is None or config["platform"] in platforms:
            configs.append(config)
    print(f"Collected {len(configs)} hardware configurations...")

    # run abstract execution and compilation for each hardware config and circuit
    for config in configs:
        if config["platform"] == "TI":
            for circuit_name, circuit in circuits.items():
                # anchors per qubit: 2q gates, measurements and resets (MR counts as one anchor, as in TrappedIonsModel)
                anchors = Counter(t.value for op in circuit for group in op.target_groups() for t in group
                                  if stim.gate_data(op.name).is_two_qubit_gate or stim.gate_data(op.name).produces_measurements
                                  or stim.gate_data(op.name).is_reset)
                abstract_hardware, concrete_hardware = prepare_ti_hardware(parameters=config["architecture"], num_qubits=circuit.num_qubits)

                print(f"[{config["platform"]}] Running abstract execution of {circuit_name} on ({config["name"]})...")
                abstract_hardware_events, abstraction_details = run_ti_abstraction(hardware=abstract_hardware, circuit=circuit)
                print(f"[{config["platform"]}] Running compilation of        {circuit_name} on ({config["name"]})...")
                try:
                    concrete_hardware_events, compiler_details = run_ti_compiler(hardware=concrete_hardware, circuit=circuit, max_compile_time=max_compile_time)
                except (RuntimeError, TimeoutError) as error:
                    print(f"Skipped {config['name']} / {circuit_name}: {error}")
                    continue

                metadata = {"config": config["name"], "platform": config["platform"], **config["architecture"],
                            "circuit": circuit_name, "num_qubits": circuit.num_qubits}
                save_raw_results(_RESULTS_DIR / "ti" / "raw" / f"{config['name']}__{circuit_name}.csv", metadata,
                                 anchors={qubit: anchors[qubit] for qubit in range(circuit.num_qubits)},
                                 abstract=(abstract_hardware_events, abstraction_details),
                                 compiled=(concrete_hardware_events, compiler_details))

        elif config["platform"] == "NA":
            pass
        elif config["platform"] == "SC":
            pass
        else:
            print(f"Unsupported platform: {config["platform"]}")

    # evaluate the results
    if visualize_true:
        visualize(platforms or ["TI", "NA", "SC"], _RESULTS_DIR, max_qubits)


def save_raw_results(file: Path, metadata: dict, anchors: dict[int, int], abstract: tuple, compiled: tuple) -> None:
    """Store one comparison (one hardware config, one circuit) as CSV: the metadata, then source, metric, qubit, anchor, count, value.

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
        for source, (events, details) in {"abstract": abstract, "compiled": compiled}.items():
            for metric, locations in vars(events).items():
                for qubit, anchor in locations:
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
    parser.add_argument("--circuits", default="*", help="glob pattern for circuit file names, without .stim (default: all)")
    parser.add_argument("--configs", default="*", help="glob pattern for config file names, without .yaml (default: all)")
    parser.add_argument("--max-qubits", type=int, default=None, help="skip circuits with more qubits (default: no limit)")
    parser.add_argument("--max-compile-time", type=float, default=600.0, help="time limit per compilation in seconds (default: 600)")
    parser.add_argument("--no-visualize", action="store_true", help="only write the raw results, no plots")
    parser.add_argument("--clean", action="store_true",
                        help="delete all files in the raw result directories of the run platforms before running")
    args = parser.parse_args()

    run(platforms=args.platform, circuit_pattern=args.circuits, config_pattern=args.configs, max_qubits=args.max_qubits,
        max_compile_time=args.max_compile_time, visualize_true=not args.no_visualize, clean=args.clean)

