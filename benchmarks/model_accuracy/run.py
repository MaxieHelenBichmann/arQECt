
"""Run the model accuracy experiments by comparing hardware events of different circuits on the abstract executions and actual compilations."""

import stim
import yaml
from pathlib import Path

from benchmarks.model_accuracy.compiler import run_ti_compiler, run_na_compiler, run_sc_compiler, run_ti_abstraction
from benchmarks.model_accuracy.architecture import prepare_ti_hardware, prepare_na_hardware, prepare_sc_hardware

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "model_accuracy" / "configs"
_CIRCUIT_DIR = Path(__file__).resolve().parents[2] / "data" / "model_accuracy" / "circuits"

def run() -> None:
    # collect different circuits
    circuits = []
    circuits.append(stim.Circuit.from_file("data/circuits/steane_example.stim"))

    # collect different hardware configs
    configs = []
    for config_file in sorted(_CONFIG_DIR.glob("*.yaml")):
        with open(config_file) as f:
                configs.append(yaml.safe_load(f))

    # run abstract execution and compilation for each hardware config and circuit
    for config in configs:
        if config["platform"] == "TI":
            for circuit in circuits:
                abstract_hardware, concrete_hardware = prepare_ti_hardware(parameters=config["architecture"], num_qubits=circuit.num_qubits)

                abstract_hardware_events = run_ti_abstraction(hardware=abstract_hardware, circuit=circuit)
                concrete_hardware_events, compiler_details = run_ti_compiler(hardware=concrete_hardware, circuit=circuit)

                # TODO: compare hardware events
                # A) How well do i approximate the actual recorded hardware events with my abstract execution (eg IZ crossings)
                # B) How good of a proxy are these recorded hardware events (eg IZ crossings) to the actual relevant metric (eg shuttling operations)
                print(f"===== {config} =====")
                print("abstract:", abstract_hardware_events)
                print("compiled:", concrete_hardware_events)
                print("compile time:", f"{compiler_details['wall_clock_s']:.2f}s")
                print("architecture:", compiler_details["architecture"])
                print("IZ entries (abstract vs compiled):", len(abstract_hardware_events.interaction_zone_entry), "vs", len(concrete_hardware_events.interaction_zone_entry))
                print("IZ evictions (abstract vs compiled):", len(abstract_hardware_events.interaction_zone_eviction), "vs", len(concrete_hardware_events.interaction_zone_eviction))
                print("shuttles:", compiler_details["num_shuttles"])
                print("makespan (timesteps):", compiler_details["num_timesteps"])
                if "shuttles_per_qubit" in compiler_details:  # ring / grid (multi_shuttler)
                    print("shuttles per qubit:", compiler_details["shuttles_per_qubit"])
                if "result" in compiler_details:  # linear: full compiled schedule (Shuttle(ion, src, dst), PhysicalSwap(...), Rzz(ion_a, ion_b))
                    print("action counts:", compiler_details["action_counts"])
                print()

                # TODO: write the data somewhere to visualize
        elif config["platform"] == "NA":
            pass
        elif config["platform"] == "SC":
            pass
        else:
            print(f"Unsupported platform: {config["platform"]}")
