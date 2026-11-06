"""Compiler for quantum circuits targeting specific hardware."""

import stim

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

from src.models.hardware import TrappedIonsHardwareEvents, NeutralAtomsHardwareEvents, SuperconductingQubitsHardwareEvents



def run_ti_compiler(parameters: dict, circuit: stim.Circuit) -> TrappedIonsHardwareEvents:
    """Run the trapped ion compiler on the given circuit."""
    pass

def run_na_compiler(parameters: dict, circuit: stim.Circuit) -> NeutralAtomsHardwareEvents:
    """Run the neutral atom compiler on the given circuit."""
    pass


def run_sc_compiler(parameters: dict, circuit: stim.Circuit) -> SuperconductingQubitsHardwareEvents:
    """Run the superconducting qubit compiler on the given circuit."""
    pass