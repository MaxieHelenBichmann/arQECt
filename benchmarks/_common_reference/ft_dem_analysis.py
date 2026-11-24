
from itertools import combinations

from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim

import numpy as np

from src.models.hardware import HardwareModel
from src.analysis.quality._decoder import Decoder

def run_dem_analysis_sp(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_weight: int) -> dict:
    noisy_circ = noisy_circuit.copy()

def run_dem_analysis_en(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_weight: int) -> dict:
    return {}

def run_dem_analysis_se(circuit_data: QECCircuit, noisy_circuit: stim.Circuit, decoder: Decoder, max_weight: int) -> dict:
    return {}


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_DETECTOR_ERROR_MODEL : dict[CircuitFunction, Callable[[QECCircuit, stim.Circuit, Decoder, int], dict]] = {
    "StatePrep": run_dem_analysis_sp,
    "Encoding": run_dem_analysis_en,
    "Syndrome": run_dem_analysis_se
}
