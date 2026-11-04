from src.analyzer import GateProcessor
import stim

class SuperconductingQubitsModel(GateProcessor):
    def process(self, gate: stim.CircuitInstruction, _ :int) -> None:
        raise NotImplementedError("Superconducting qubits model is not implemented yet.")