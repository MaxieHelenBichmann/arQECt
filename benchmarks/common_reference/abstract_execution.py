"""Abstract execution of quantum circuits targeting specific hardware."""

import contextlib
import io
import math
import multiprocessing
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict

import stim

from src.models.hardware import HardwareModel, PlatformTag, HardwareEvents, TrappedIonsHardwareEvents, TrappedIonsModel, SuperconductingQubitsHardwareEvents, SuperconductingQubitsModel, NeutralAtomsHardwareEvents, NeutralAtomsModel
from collections.abc import Callable

from benchmarks.common_reference.architecture_preparation import HardwareArchitecture


def run_ti_abstraction(hardware: TrappedIonsModel, circuit: stim.Circuit) -> tuple[TrappedIonsHardwareEvents, dict]:
    """Run the trapped ion abstract execution on the given circuit."""
    start = time.perf_counter()
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))
    wall_clock_s = time.perf_counter() - start

    return hardware.hardware_events, {"wall_clock_s": wall_clock_s}


def run_na_abstraction(hardware: NeutralAtomsModel, circuit: stim.Circuit) -> tuple[NeutralAtomsHardwareEvents, dict]:
    """Run the neutral atom abstract execution on the given circuit."""
    start = time.perf_counter()
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))
    wall_clock_s = time.perf_counter() - start

    return hardware.hardware_events, {"wall_clock_s": wall_clock_s}

def run_sc_abstraction(hardware: SuperconductingQubitsModel, circuit: stim.Circuit) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
    """Run the superconducting qubit abstract execution on the given circuit."""
    start = time.perf_counter()
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))
    wall_clock_s = time.perf_counter() - start

    return hardware.hardware_events, {"wall_clock_s": wall_clock_s}


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_ti_baseline(circuit: stim.Circuit) -> tuple[TrappedIonsHardwareEvents, dict]:
    """Naive per-gate model for comparison: the qubits of every gate enter an interaction zone before and leave it after the gate."""
    hardware_events = TrappedIonsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # qubit -> number of completed anchor operations
    for op in circuit:
        data = stim.gate_data(op.name)
        if not (data.is_single_qubit_gate or data.is_two_qubit_gate):
            continue
        for group in op.target_groups():
            qubits = [t.value for t in group]
            hardware_events.interaction_zone_entry.extend((qubit, qubit_steps[qubit]) for qubit in qubits)
            if data.is_two_qubit_gate or data.produces_measurements or data.is_reset:
                for qubit in qubits:
                    qubit_steps[qubit] += 1
            hardware_events.interaction_zone_eviction.extend((qubit, qubit_steps[qubit]) for qubit in qubits)

    return hardware_events, {}

def run_na_baseline(circuit: stim.Circuit) -> tuple[NeutralAtomsHardwareEvents, dict]:
    """Naive per-gate model for comparison: the qubits of every 2-qubit gate (or measurement) enter the entanglement (or readout) zone from storage before and return after the gate."""
    hardware_events = NeutralAtomsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # qubit -> number of completed anchor operations
    for op in circuit:
        data = stim.gate_data(op.name)
        zone_changes = (hardware_events.storage_entanglement if data.is_two_qubit_gate
                        else hardware_events.storage_readout if data.produces_measurements else None)
        for group in op.target_groups():
            qubits = [t.value for t in group]
            if zone_changes is not None:
                zone_changes.extend((qubit, qubit_steps[qubit]) for qubit in qubits)
            if data.is_two_qubit_gate or data.produces_measurements or data.is_reset:
                for qubit in qubits:
                    qubit_steps[qubit] += 1
            if zone_changes is not None:
                zone_changes.extend((qubit, qubit_steps[qubit]) for qubit in qubits)

    return hardware_events, {}

def run_sc_baseline(circuit: stim.Circuit) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
    """Naive per-gate model for comparison: one virtual swap before and one after every 2-qubit gate (recorded on the two gate qubits)."""
    hardware_events = SuperconductingQubitsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # qubit -> number of completed anchor operations
    for op in circuit:
        data = stim.gate_data(op.name)
        for group in op.target_groups():
            qubits = [t.value for t in group]
            if data.is_two_qubit_gate:
                hardware_events.virtual_swap.append(((qubits[0], qubit_steps[qubits[0]]), (qubits[1], qubit_steps[qubits[1]])))
            if data.is_two_qubit_gate or data.produces_measurements or data.is_reset:
                for qubit in qubits:
                    qubit_steps[qubit] += 1
            if data.is_two_qubit_gate:
                hardware_events.virtual_swap.append(((qubits[0], qubit_steps[qubits[0]]), (qubits[1], qubit_steps[qubits[1]])))

    return hardware_events, {}

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_BASELINE : dict[PlatformTag, Callable[[stim.Circuit], tuple[HardwareModel, HardwareEvents | dict]]] = {
    "TI": run_ti_baseline,
    "NA": run_na_baseline,
    "SC": run_sc_baseline
}

_RUN_ABSTRACTION : dict[PlatformTag, Callable[[HardwareModel,  stim.Circuit], tuple[HardwareModel, HardwareEvents | dict]]] = {
    "TI": run_ti_abstraction,
    "NA": run_na_abstraction,
    "SC": run_sc_abstraction
}