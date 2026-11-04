"""Representation of a quantum circuit as a directed acyclic graph (DAG) of immediate gate dependencies.

This representation is useful for analyzing the the dependency structure of a quantum circuit, which influences gate scheduling,
but is not fit for capturing the overall connectivity or focusing on one specific qubit.

It is a gate-centered representation, where each gate is a node in the graph, and directed edges represent the immediate dependencies between gates.
"""


from src.analysis.analyzer import GateProcessor
import stim
from collections import defaultdict

class DependencyDAG(GateProcessor):
    """A dependency DAG that tracks the dependencies between gates in a circuit."""

    def __init__(self) -> None:
        self.dag = {}  # type: dict[int, set[int]]

    def process(self, gate: stim.CircuitInstruction) -> None:
        raise NotImplementedError("Dependency DAG is not implemented yet.")