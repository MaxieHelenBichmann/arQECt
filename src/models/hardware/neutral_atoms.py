from src.analysis.analyzer import GateProcessor
import stim

class NeutralAtomsModel(GateProcessor):
    def process(self, gate: stim.CircuitInstruction, _ :int) -> None:
        raise NotImplementedError("Neutral atoms model is not implemented yet.")