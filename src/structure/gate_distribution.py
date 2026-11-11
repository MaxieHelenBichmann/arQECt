"""Representation of a quantum circuit as a directed acyclic graph (DAG) of immediate gate dependencies.

This representation is useful for analyzing the the dependency structure of a quantum circuit, which influences gate scheduling,
but is not fit for capturing the overall connectivity or focusing on one specific qubit.

It is a gate-centered representation, where each gate is a node in the graph, and directed edges represent the immediate dependencies between gates.
"""

from src.analysis.processor import GateProcessor
import stim

from collections import Counter

class GateDistribution(GateProcessor):
    """A dependency DAG that tracks the dependencies between gates in a circuit."""

    def __init__(self) -> None:
        self.gd :  Counter[str] = Counter()

    def process(self, gate: stim.CircuitInstruction) -> None:
        self.gd[gate.name] += 1 

    def finalize(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"GateDistribution()"

    def __str__(self) -> str:
        lines = [f"GateDistribution with {self.gd.total()} gates ({len(self.gd)} types):"]
        lines += [f"  {name}: {count}x" for name, count in self.gd.most_common()]
        return "\n".join(lines)
    