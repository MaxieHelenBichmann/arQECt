from typing import Protocol
import stim


class GateProcessor(Protocol):
    def process(self, gate: stim.CircuitInstruction) -> None:
        ...

    def finalize(self) -> None:
        ...


class CircuitAnalyzer:
    def __init__(self, *processors: GateProcessor):
        self.processors = processors

    def run(self, circuit: stim.Circuit): 
        """Analyzes a circuit by processing each gate simultaneously with the provided processors."""
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                raise ValueError("unsupported instruction: REPEAT blocks")
            if op.name == "QUBIT_COORDS":  # annotation
                continue
            gate_data = stim.gate_data(op.name)
            for group in op.target_groups(): # only 1q gates and 2q gates on plain qubits
                if not (gate_data.is_single_qubit_gate or gate_data.is_two_qubit_gate) or (
                    gate_data.is_noisy_gate and not gate_data.produces_measurements
                ) or not all(target.is_qubit_target for target in group):
                    raise ValueError(f"unsupported instruction: {op.name} on targets {group}")
                
                gate = stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag)
                for processor in self.processors:
                    processor.process(gate)

        for processor in self.processors:
            processor.finalize()