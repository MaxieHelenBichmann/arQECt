
from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim

from src.analysis.quality.encoder import EncoderCircuitAnalyzer
from src.analysis.quality.state_prep import StatePreparationCircuitAnalyzer
from src.analysis.quality.syndrome_extract import SyndromeExtractionAnalyzer
from src.models.hardware import HardwareModel
from src.analysis.quality._decoder import BeliefPropOrdStat

def run_mc_analysis_sp(circuit: QECCircuit, hw: HardwareModel, max_shots: int) -> dict:
    analyzer = StatePreparationCircuitAnalyzer(
        hw, 
        circuit.stabilizers, 
        circuit.logical_operators, 
        max_shots=max_shots, 
        decoder=BeliefPropOrdStat(hw, circuit.stabilizers), 
        accept_if=circuit.verification, 
        state=circuit.quantum_state
    )
    return analyzer.analyze()


def run_mc_analysis_en(circuit: QECCircuit, hw: HardwareModel, max_shots: int) -> dict:
    analyzer = EncoderCircuitAnalyzer(
        hw, 
        circuit.stabilizers, 
        circuit.logical_operators, 
        max_shots=max_shots, 
        decoder=BeliefPropOrdStat(hw, circuit.stabilizers), 
        logical_qubits=circuit.logical_qubits
    )
    return analyzer.analyze()

def run_mc_analysis_se(circuit: QECCircuit, hw: HardwareModel, max_shots: int) -> dict:
    analyzer = SyndromeExtractionAnalyzer(
        hw, 
        circuit.stabilizers, 
        circuit.logical_operators, 
        max_shots=max_shots, 
        decoder=BeliefPropOrdStat(hw, circuit.stabilizers), 
        syndrome_measurements=circuit.syndrome_measurements
    ) 
    return analyzer.analyze()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_MONTE_CARLO : dict[CircuitFunction, Callable[[QECCircuit, HardwareModel, int], dict]] = {
    "StatePrep": run_mc_analysis_sp,
    "Encoding": run_mc_analysis_en,
    "Syndrome": run_mc_analysis_se
}