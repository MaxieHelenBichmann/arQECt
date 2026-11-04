from src.analysis.analyzer import GateProcessor
import stim

class InteractionGraph(GateProcessor):
    def process(self, gate: stim.CircuitInstruction, index :int) -> None:
        raise NotImplementedError("Interaction graph is not implemented yet.")