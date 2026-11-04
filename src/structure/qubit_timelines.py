from src.analysis.analyzer import GateProcessor
import stim

class QubitTimelines(GateProcessor):
    def process(self, gate: stim.CircuitInstruction) -> None:
        raise NotImplementedError("Qubit timelines are not implemented yet.")