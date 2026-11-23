
from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim


def run_mc_analysis_sp(circuit: QECCircuit) -> dict:
    return {}

def run_mc_analysis_en(circuit: QECCircuit) -> dict:
    return {}

def run_mc_analysis_se(circuit: QECCircuit) -> dict:
    return {}


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_MONTE_CARLO : dict[CircuitFunction, Callable[[QECCircuit], dict]] = {
    "StatePrep": run_mc_analysis_sp,
    "Encoding": run_mc_analysis_en,
    "Syndrome": run_mc_analysis_se
}