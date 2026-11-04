"""Representation of a quantum circuit as a directed acyclic graph (DAG) of immediate gate dependencies.

This representation is useful for analyzing the the dependency structure of a quantum circuit, which influences gate scheduling,
but is not fit for capturing the overall connectivity or focusing on one specific qubit.

It is a gate-centered representation, where each gate is a node in the graph, and directed edges represent the immediate dependencies between gates.
"""

# TODO: optimize DAG representation, potentially storing in layers

from src.analysis.analyzer import GateProcessor
import stim
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class DAGGate:
    qubits: tuple[int, ...]
    predecessors: set[int] = field(default_factory=set)
    successors: set[int] = field(default_factory=set)

class DependencyDAG(GateProcessor):
    """A dependency DAG that tracks the dependencies between gates in a circuit."""

    def __init__(self) -> None:
        self.dag: defaultdict[int, DAGGate] = {}
        self._index = 0

        self._last_gate: dict[int, int] = {} 

    def process(self, gate: stim.CircuitInstruction) -> None:
        qubits = tuple(t.value for t in gate.targets_copy())
        predecessors = {self._last_gate[q] for q in qubits if q in self._last_gate}
        for p in predecessors:
            self.dag[p].successors.add(self._index)
        self.dag[self._index] = DAGGate(qubits=qubits, predecessors=predecessors)
        for q in qubits:
            self._last_gate[q] = self._index
        self._index += 1
