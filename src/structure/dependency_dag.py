from src.analysis.analyzer import GateProcessor
import stim

class DependencyDAG(GateProcessor):
    """A dependency DAG that tracks the dependencies between gates in a circuit."""

    def __init__(self) -> None:
        self.dag = {}  # type: dict[int, set[int]]

    def process(self, gate: stim.CircuitInstruction) -> None:
        raise NotImplementedError("Dependency DAG is not implemented yet.")