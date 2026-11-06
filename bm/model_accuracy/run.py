
"""Run the model accuracy experiments by comparing hardware events of different circuits on the abstract executions and actual compilations."""

import stim

from bm.model_accuracy.compiler import run_ti_compiler, run_na_compiler, run_sc_compiler
from src.models.hardware import TrappedIonsModel, NeutralAtomsModel, SuperconductingQubitsModel

def run() -> None:
    circuit = stim.Circuit.from_file("data/circuits/steane_example.stim")

    # TODO: define different circuits
    # TODO: define different hardware configs -> my HW models and compilers Architectures
    # TODO: run abstract execution and compilation for each hardware config and circuit
    # TODO: "extract" hardware events from both and compare them
