
from itertools import combinations
from math import comb

from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim

import time

from src.analysis.robustness.encoder import EncoderCircuitAnalyzer
from src.analysis.robustness.state_prep import StatePreparationCircuitAnalyzer
from src.analysis.robustness.syndrome_extract import SyndromeExtractionAnalyzer

import numpy as np

from src.models.hardware import HardwareModel
from src.analysis.robustness._decoder import Decoder

def run_dem_analysis_sp(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_rel_tail: float, max_fault_sets: int) -> dict:
    analyzer = StatePreparationCircuitAnalyzer(
        noisy_circuit,
        stabilizers=circuit_data.stabilizers,
        logical_operators=circuit_data.logical_operators,
        max_fault_sets=max_fault_sets,
        max_rel_tail=max_rel_tail,
        decoder=decoder,
        accept_if=circuit_data.verification,
        state=circuit_data.quantum_state,
    )
    start = time.perf_counter()
    result = analyzer.analyze()
    t = time.perf_counter() - start
    result["wall_clock_s"] = t
    return result
    

def run_dem_analysis_en(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_rel_tail: float, max_fault_sets: int) -> dict:
    analyzer = EncoderCircuitAnalyzer(
            noisy_circuit,
            stabilizers=circuit_data.stabilizers,
            logical_operators=circuit_data.logical_operators,
            max_fault_sets=max_fault_sets,
            max_rel_tail=max_rel_tail,
            decoder=decoder,
            logical_qubits=circuit_data.logical_qubits
        )
    start = time.perf_counter()
    result = analyzer.analyze()
    t = time.perf_counter() - start
    result["wall_clock_s"] = t
    return result

def run_dem_analysis_se(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_rel_tail: float, max_fault_sets: int) -> dict:
    analyzer = SyndromeExtractionAnalyzer(
                noisy_circuit,
                stabilizers=circuit_data.stabilizers,
                logical_operators=circuit_data.logical_operators,
                max_fault_sets=max_fault_sets,
                max_rel_tail=max_rel_tail,
                decoder=decoder,
                syndrome_measurements=circuit_data.syndrome_measurements
            )
    start = time.perf_counter()
    result = analyzer.analyze()
    t = time.perf_counter() - start
    result["wall_clock_s"] = t
    return result    


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_DETECTOR_ERROR_MODEL : dict[CircuitFunction, Callable[[QECCircuit, stim.Circuit, Decoder, float, int], dict]] = {
    "StatePrep": run_dem_analysis_sp,
    "Encoding": run_dem_analysis_en,
    "Syndrome": run_dem_analysis_se
}
