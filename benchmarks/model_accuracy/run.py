
"""Run the model accuracy experiments by comparing hardware events of different circuits on the abstract executions and actual compilations."""

import stim

from benchmarks.model_accuracy.compiler import run_ti_compiler, run_na_compiler, run_sc_compiler, run_ti_abstraction
from benchmarks.model_accuracy.architecture import prepare_ti_hardware, prepare_na_hardware, prepare_sc_hardware
from src.models.hardware import TrappedIonsModel, NeutralAtomsModel, SuperconductingQubitsModel

# TODO: this is one concrete example, make it read input config files? 

def run() -> None:
    # define different circuits
    circuit = stim.Circuit.from_file("data/circuits/steane_example.stim")

    # define different hardware configs -> named TI architectures (the abstract model parameters are derived from them)
    ti_architectures = [
        {"architecture": "linear", "nr_zones": 1, "zone_capacity": 3},  # Quantinuum H1 style
        {"architecture": "ring", "nr_zones": 2},  # Quantinuum Helios style (storage ring + 2 gate-zone legs)
        {"architecture": "grid", "nr_zones": 4, "grid_size": (3, 3)},  # 2D QCCD
    ]

    # run abstract execution and compilation for each hardware config and circuit
    for parameters in ti_architectures:
        abstract_hardware, concrete_hardware = prepare_ti_hardware(parameters=parameters)

        abstract_hardware_events = run_ti_abstraction(parameters=parameters, circuit=circuit)
        concrete_hardware_events, compiler_details = run_ti_compiler(parameters=parameters, circuit=circuit)

        # TODO: compare hardware events
        # A) How well do i approximate the actual recorded hardware events with my abstract execution (eg IZ crossings)
        # B) How good of a proxy are these recorded hardware events (eg IZ crossings) to the actual relevant metric (eg shuttling operations)
        print(f"===== {parameters} =====")
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


def read_config() -> dict:
    pass

def read_circuit() -> dict:
    pass