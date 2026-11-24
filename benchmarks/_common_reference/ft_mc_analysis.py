
from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim

from src.analysis.quality.encoder import EncoderCircuitAnalyzer
from src.analysis.quality.state_prep import StatePreparationCircuitAnalyzer
from src.analysis.quality.syndrome_extract import SyndromeExtractionAnalyzer
from src.models.hardware import HardwareModel
from src.analysis.quality._decoder import Decoder

def run_mc_analysis_sp(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_shots: int) -> dict:
    analyzer = StatePreparationCircuitAnalyzer(
        noisy_circuit, 
        circuit_data.stabilizers, 
        circuit_data.logical_operators, 
        max_shots=max_shots, 
        decoder=decoder, 
        accept_if=circuit_data.verification, 
        state=circuit_data.quantum_state
    )
    return analyzer.analyze()


def run_mc_analysis_en(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_shots: int) -> dict:
    analyzer = EncoderCircuitAnalyzer(
        noisy_circuit, 
        circuit_data.stabilizers, 
        circuit_data.logical_operators, 
        max_shots=max_shots, 
        decoder=decoder, 
        logical_qubits=circuit_data.logical_qubits
    )
    return analyzer.analyze()

def run_mc_analysis_se(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_shots: int) -> dict:
    analyzer = SyndromeExtractionAnalyzer(
        noisy_circuit, 
        circuit_data.stabilizers, 
        circuit_data.logical_operators, 
        max_shots=max_shots, 
        decoder=decoder, 
        syndrome_measurements=circuit_data.syndrome_measurements
    ) 
    return analyzer.analyze()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_MONTE_CARLO : dict[CircuitFunction, Callable[[QECCircuit, stim.Circuit, Decoder, int], dict]] = {
    "StatePrep": run_mc_analysis_sp,
    "Encoding": run_mc_analysis_en,
    "Syndrome": run_mc_analysis_se
}