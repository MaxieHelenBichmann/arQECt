"""Compiler for quantum circuits targeting specific hardware."""

import stim

from src.models.noise.noise import NoiseModel

from mqt.ionshuttler.linear import (
    Architecture as TIArchitecture, LinearCompiler, CompilationStatus
)
from mqt.ionshuttler.linear.actions import Shuttle, PhysicalSwap

from mqt.qmap.na.zoned import (
    ZonedNeutralAtomArchitecture,
    RoutingAgnosticCompiler,
    RoutingAwareCompiler,
)

from mqt.qmap.plugins.qiskit.sc import compile_
from mqt.qmap.sc import Architecture as SCArchitecture, Method

from src.models.hardware import ( TrappedIonsModel, TrappedIonsHardwareEvents, NeutralAtomsModel, NeutralAtomsHardwareEvents, SuperconductingQubitsModel, SuperconductingQubitsHardwareEvents)



def run_ti_compiler(parameters: dict, circuit: stim.Circuit) -> tuple[TrappedIonsHardwareEvents, dict]: # (proxy information, details)
    """Run the trapped ion compiler on the given circuit."""
    pass


def run_na_compiler(parameters: dict, circuit: stim.Circuit) -> tuple[NeutralAtomsHardwareEvents, dict]:
    """Run the neutral atom compiler on the given circuit."""
    pass



def run_sc_compiler(parameters: dict, circuit: stim.Circuit) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
    """Run the superconducting qubit compiler on the given circuit."""
    pass


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_ti_abstraction(parameters: dict, circuit: stim.Circuit) -> TrappedIonsHardwareEvents:
    """Run the trapped ion abstract execution on the given circuit."""
    abstract_model = TrappedIonsModel(interaction_zone_capacity=parameters["interaction_zone_capacity"], nr_interaction_sites=parameters["nr_interaction_sites"])

    for op in circuit:
        for group in op.target_groups():
            abstract_model.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))

    return abstract_model.hardware_events


def run_na_abstraction(parameters: dict, circuit: stim.Circuit) -> TrappedIonsHardwareEvents:
    """Run the neutral atom abstract execution on the given circuit."""
    abstract_model = NeutralAtomsModel(entanglement_zone_capacity=parameters["entanglement_zone_capacity"], readout_zone_capacity=parameters["readout_zone_capacity"])

    for op in circuit:
        for group in op.target_groups():
            abstract_model.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))

    return abstract_model.hardware_events

def run_sc_abstraction(parameters: dict, circuit: stim.Circuit) -> TrappedIonsHardwareEvents:
    """Run the superconducting qubit abstract execution on the given circuit."""
    abstract_model = SuperconductingQubitsModel(number_of_qubits=parameters["number_of_qubits"], gate_durations=parameters["gate_durations"], topology=parameters["topology"])

    for op in circuit:
        for group in op.target_groups():
            abstract_model.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))

    return abstract_model.hardware_events