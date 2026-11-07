
"""Run the model accuracy experiments by comparing hardware events of different circuits on the abstract executions and actual compilations."""

import stim

from bm.model_accuracy.compiler import run_ti_compiler, run_na_compiler, run_sc_compiler, run_ti_abstraction
from src.models.hardware import TrappedIonsModel, NeutralAtomsModel, SuperconductingQubitsModel

# TODO: this is one concrete example, make it read input config files? 

def run() -> None:
    # define different circuits
    circuit = stim.Circuit.from_file("data/circuits/steane_example.stim")

    # define different hardware configs -> my HW models and compilers Architectures
    parameters = {"interaction_zone_capacity": 3, "nr_interaction_sites": 1}

    # run abstract execution and compilation for each hardware config and circuit
    abstract_hardware_events = run_ti_abstraction(parameters=parameters, circuit=circuit)
    concrete_hardware_events, compiler_details = run_ti_compiler(parameters=parameters, circuit=circuit)

    # TODO: "extract" hardware events from both and compare them
    # A) How well do i approximate the actual recorded hardware events with my abstract execution (eg IZ crossings)
    # B) How good of a proxy are these recorded hardware events (eg IZ crossings) to the actual relevant metric (eg shuttling operations)
    print(abstract_hardware_events)
    print()
    print(concrete_hardware_events)
    print(compiler_details)
