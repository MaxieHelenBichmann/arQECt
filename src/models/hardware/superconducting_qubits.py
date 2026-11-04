from src.analysis.analyzer import GateProcessor
import stim

class SuperconductingQubitsModel(GateProcessor):
    def process(self, gate: stim.CircuitInstruction) -> None:
        raise NotImplementedError("Superconducting qubits model is not implemented yet.")