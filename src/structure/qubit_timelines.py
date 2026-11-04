"""Representation of a quantum circuit focusing on the gates regarding each qubit over time. 

This representation is useful for analyzing the transformation acting on each individual qubit over time, and is not fit for capturing their concrete interactions or dependency structure.

It is a qubit-centered representation, where each qubit has a timeline of gates acting on it and partners it interacts with, represented as entries in the timeline.
"""

from src.analysis.analyzer import GateProcessor
import stim
from dataclasses import dataclass
from typing import Literal
from collections import defaultdict

GateKind = Literal["SN", "MS", "EN"]

@dataclass(frozen=True, slots=True)
class TimelineEntry:
    kind: GateKind
    partner: int | None = None  

class QubitTimelines(GateProcessor):
    """A class to store the timelines of qubits in a circuit."""

    def __init__(self) -> None:
        self.timelines: defaultdict[int, list[TimelineEntry]] = defaultdict(list)

    def process(self, gate: stim.CircuitInstruction) -> None:
        """Processes a gate by adding to the timelines of the qubits involved."""
        data = stim.gate_data(gate.name)
        qubits = [t.value for t in gate.targets_copy()]

        if data.is_two_qubit_gate:
            src, tgt = qubits
            self.timelines[src].append(TimelineEntry("EN", tgt))
            self.timelines[tgt].append(TimelineEntry("EN", src))
        elif data.produces_measurements:
            self.timelines[qubits[0]].append(TimelineEntry("MS"))
        else:  # single-qubit gate or reset
            self.timelines[qubits[0]].append(TimelineEntry("SN"))
