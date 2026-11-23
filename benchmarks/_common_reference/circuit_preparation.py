
from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim


def prepare_sp_circ(name: str, circuit: stim.Circuit) -> QECCircuit:
    stabilizers = []
    logical_op = []
    state = "0"
    verification = {i: int(v) for i, v in enumerate(circuit.reference_sample())}
    return QECCircuit(
        circuit=circuit,
        function="StatePrep",
        stabilizers=stabilizers,
        logical_operators=logical_op,
        quantum_state=state,
        verification=verification,
        logical_qubits=None,
        syndrome_measurements=None
    )

def prepare_en_circ(name: str, circuit: stim.Circuit) -> QECCircuit:
    stabilizers = []
    logical_op = []
    log_qubits = []
    return QECCircuit(
        circuit=circuit,
        function="Encoding",
        stabilizers=stabilizers,
        logical_operators=logical_op,
        quantum_state=None,
        verification=None,
        logical_qubits=log_qubits,
        syndrome_measurements=None
    )

def prepare_se_circ(name: str, circuit: stim.Circuit) -> QECCircuit:
    stabilizers = []
    logical_op = []
    syndrome_meas = []
    return QECCircuit(
        circuit=circuit,
        function="Syndrome",
        stabilizers=stabilizers,
        logical_operators=logical_op,
        quantum_state=None,
        verification=None,
        logical_qubits=None,
        syndrome_measurements=syndrome_meas
    )


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_PREPARE_CIRCUIT : dict[CircuitFunction, Callable[[str, stim.Circuit], QECCircuit]] = {
    "StatePrep": prepare_sp_circ,
    "Encoding": prepare_en_circ,
    "Syndrome": prepare_se_circ
}