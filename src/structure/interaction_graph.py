from src.analysis.analyzer import GateProcessor
import stim

class InteractionGraph(GateProcessor):
    """An interaction graph that tracks the interactions between qubits in a circuit without the temporal dimension."""

    def __init__(self) -> None:
        self.graph: dict[int, set[int]] = {}
            
    def process(self, gate: stim.CircuitInstruction) -> None:
        raise NotImplementedError("Interaction graph is not implemented yet.")