"""Representation of a quantum circuit focusing on the gates regarding each qubit over time. 

This representation is useful for analyzing the transformation acting on each individual qubit over time, and is not fit for capturing their concrete interactions or dependency structure.

It is a qubit-centered representation, where each qubit has a timeline of gates acting on it and partners it interacts with, represented as entries in the timeline.
"""

from src.analysis import GateProcessor, QubitTimelinesAnalyzer
import stim
from dataclasses import dataclass
from collections import defaultdict

from src.models.common import GateKind

@dataclass(frozen=True, slots=True)
class TimelineGate:
    kind: GateKind
    partner: int | None = None  

class QubitTimelines(GateProcessor):
    """A class to store the timelines of qubits in a circuit."""

    def __init__(self) -> None:
        self.timelines: defaultdict[int, list[TimelineGate]] = defaultdict(list)

    def process(self, gate: stim.CircuitInstruction) -> None:
        """Processes a gate by adding to the timelines of the qubits involved."""
        data = stim.gate_data(gate.name)
        qubits = [t.value for t in gate.targets_copy()]

        if data.is_two_qubit_gate:
            src, tgt = qubits
            self.timelines[src].append(TimelineGate("EN", tgt))
            self.timelines[tgt].append(TimelineGate("EN", src))
        elif data.produces_measurements:
            self.timelines[qubits[0]].append(TimelineGate("MS"))
        else:  # single-qubit gate or reset
            self.timelines[qubits[0]].append(TimelineGate("SN"))

    def finalize(self) -> None:
        pass

    def extractor(self, function) -> list[QubitTimelinesAnalyzer]:
        return [QubitTimelinesAnalyzer(qtl=self)]

    def __repr__(self) -> str:
        return f"QubitTimelines(nr_qubits={len(self.timelines)})"

    def __str__(self) -> str:
        lines = [f"QubitTimelines for {len(self.timelines)} qubits (SN: single-qubit/reset, MS: measurement, EN(partner): entangling):"]
        for qubit in sorted(self.timelines):
            entries = " ".join(e.kind if e.partner is None else f"{e.kind}({e.partner})" for e in self.timelines[qubit])
            lines.append(f"  {qubit:>3}: {entries}")
        return "\n".join(lines)
