"""Constructs specific hardware."""

from src.models.hardware import ( TrappedIonsModel, NeutralAtomsModel, SuperconductingQubitsModel)
from mqt.ionshuttler.linear import Architecture as TIArchitecture
from mqt.qmap.sc import Architecture as SCArchitecture
from mqt.qmap.na.zoned import ZonedNeutralAtomArchitecture

def prepare_ti_hardware(parameters: dict) -> tuple[TrappedIonsModel, TIArchitecture]:
    capacity = parameters["zone_capacity"] if parameters["architecture"] == "linear" else 2
    return TrappedIonsModel(interaction_zone_capacity=capacity, nr_interaction_sites=parameters["nr_zones"]),


def prepare_na_hardware(parameters: dict) -> tuple[NeutralAtomsModel, ZonedNeutralAtomArchitecture]:
    return NeutralAtomsModel(entanglement_zone_capacity=parameters["entanglement_zone_capacity"], readout_zone_capacity=parameters["readout_zone_capacity"]),



def prepare_sc_hardware(parameters: dict) -> tuple[SuperconductingQubitsModel, SCArchitecture]:
    return SuperconductingQubitsModel(number_of_qubits=parameters["number_of_qubits"], gate_durations=parameters["gate_durations"], topology=parameters["topology"]), 
