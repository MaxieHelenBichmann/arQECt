"""Representation of a quantum circuit as a directed acyclic graph (DAG) of immediate gate dependencies.

This representation is useful for analyzing the the dependency structure of a quantum circuit, which influences gate scheduling,
but is not fit for capturing the overall connectivity or focusing on one specific qubit.

It is a gate-centered representation, where each gate is a node in the graph, and directed edges represent the immediate dependencies between gates.
"""

# TODO: optimize DAG representation, potentially storing in layers

from src.analysis.processor import GateProcessor
import stim
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class DAGGate:
    qubits: tuple[int, ...]
    readout: bool = False
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)
    schedule_interval: tuple[int | None, int | None] = field(default_factory=lambda: (None, None))  # (asap schedule time, alap schedule time)

class DependencyDAG(GateProcessor):
    """A dependency DAG that tracks the dependencies between gates in a circuit."""

    def __init__(self) -> None:
        self.dag: defaultdict[int, DAGGate] = {}
        self.depth: int = 0
        self._index = 0

        self._last_gate: dict[int, int] = {}

    def process(self, gate: stim.CircuitInstruction) -> None:
        qubits = tuple(t.value for t in gate.targets_copy())
        predecessors = {self._last_gate[q] for q in qubits if q in self._last_gate}
        for p in predecessors:
            self.dag[p].successors.add(self._index)

        asap_time = max([self.dag[p].schedule_interval[0] for p in predecessors]) + 1 if predecessors else 1
        self.depth = max(self.depth, asap_time)

        self.dag[self._index] = DAGGate(qubits=qubits, readout=gate.produces_measurements, predecessors=predecessors, schedule_interval=(asap_time, None))
        for q in qubits:
            self._last_gate[q] = self._index
        self._index += 1

    def finalize(self) -> None:
        for i in sorted(self.dag.keys(), reverse=True):
            alap_time = min([self.dag[s].schedule_interval[1] for s in self.dag[i].successors]) - 1 if self.dag[i].successors else self.depth
            self.dag[i].schedule_interval = (self.dag[i].schedule_interval[0], alap_time)

    def __repr__(self) -> str:
        return f"DependencyDAG(nr_gates={len(self.dag)})"

    def __str__(self) -> str:
        lines = [f"DependencyDAG with {len(self.dag)} gates (index: qubits | predecessors --> successors):"]
        for index, node in self.dag.items():
            predecessors = ", ".join(map(str, sorted(node.predecessors))) or "-"
            successors = ", ".join(map(str, sorted(node.successors))) or "-"
            lines.append(f"  {index:>3}: {node.qubits} | {predecessors} --> {successors}")
        return "\n".join(lines)
    