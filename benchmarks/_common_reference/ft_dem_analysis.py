
from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim


def run_dem_analysis_sp(circuit: QECCircuit) -> dict:
    return {}

def run_dem_analysis_en(circuit: QECCircuit) -> dict:
    return {}

def run_dem_analysis_se(circuit: QECCircuit) -> dict:
    return {}


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_DETECTOR_ERROR_MODEL : dict[CircuitFunction, Callable[[QECCircuit], dict]] = {
    "StatePrep": run_dem_analysis_sp,
    "Encoding": run_dem_analysis_en,
    "Syndrome": run_dem_analysis_se
}