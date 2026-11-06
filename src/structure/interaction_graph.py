"""Representation of a quantum circuit focusing the interactions between qubits over the whole circuit.

This representation is useful for analyzing the overall connectivity and interaction patterns in a quantum circuit, which influence qubit mapping and routing, but it does not provide information about the temporal sequence of operations.

It is a structure-centered representation, where each qubit is a node in the graph, and edges represent interactions between qubits, weighted by the number of interactions.
"""

from src.analysis.analyzer import GateProcessor
import stim
from collections import defaultdict

class InteractionGraph(GateProcessor):
    """A weighted undirected interaction graph that tracks the interactions between qubits in a circuit."""

    def __init__(self) -> None:
        self.graph: defaultdict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))

    def process(self, gate: stim.CircuitInstruction) -> None:
        """Processes a gate by updating the weighted interaction graph."""
        if stim.gate_data(gate.name).is_two_qubit_gate:
            src, tgt = (t.value for t in gate.targets_copy())
            self.graph[src][tgt] += 1
            self.graph[tgt][src] += 1

    def finalize(self) -> None:
        pass

    def __repr__(self) -> str:
        nr_edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        return f"InteractionGraph(nr_qubits={len(self.graph)}, nr_edges={nr_edges})"

    def __str__(self) -> str:
        edges = sorted((a, b, weight) for a, neighbors in self.graph.items() for b, weight in neighbors.items() if a < b)
        lines = [f"InteractionGraph with {len(self.graph)} qubits and {len(edges)} edges (qubit - qubit: weight):"]
        lines += [f"  {a:>3} - {b:<3}: {weight}" for a, b, weight in edges]
        return "\n".join(lines)