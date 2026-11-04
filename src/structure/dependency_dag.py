from src.analysis.analyzer import GateProcessor
import stim

class DependencyDAG(GateProcessor):
    def process(self, gate: stim.CircuitInstruction, index :int) -> None:
        raise NotImplementedError("Dependency DAG is not implemented yet.")