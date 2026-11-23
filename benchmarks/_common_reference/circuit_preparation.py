
from src.circuits.circuit import CircuitFunction, QECCircuit
from collections.abc import Callable
import stim
from pathlib import Path


def _load_code(file: Path) -> tuple[list[stim.PauliString], list[tuple[stim.PauliString, stim.PauliString]]]:
    code_dir = file.parent.parent
    stabilizers = [stim.PauliString(s) for s in open(code_dir / "stabilizers.txt").read().split()]
    logicals = [stim.PauliString(s) for s in open(code_dir / "logicals.txt").read().split()]
    k = len(logicals) // 2
    return stabilizers, list(zip(logicals[:k], logicals[k:]))

def prepare_sp_circ(file: Path, circuit: stim.Circuit) -> QECCircuit:
    stabilizers, logical_op = _load_code(file)

    task_name = file.parent.name
    state = "0" if "zero" in task_name else "+" if "plus" in task_name else "1" if "one" in task_name else "-"
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

def prepare_en_circ(file: Path, circuit: stim.Circuit) -> QECCircuit:
    stabilizers, _ = _load_code(file)

    tableau = stim.Tableau.from_circuit(circuit, ignore_reset=True)
    log_qubits = [q for q in range(len(tableau)) if all(tableau.x_output(q).commutes(s) and tableau.z_output(q).commutes(s) for s in stabilizers)]
    logical_op = [(tableau.x_output(q), tableau.z_output(q)) for q in log_qubits] # exact logical operators relevant and not necessarily consistent with the proposed basis

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

def prepare_se_circ(file: Path, circuit: stim.Circuit) -> QECCircuit:
    _, logical_op = _load_code(file)

    n = len(logical_op[0][0])
    inverse = stim.Tableau.from_circuit(circuit, ignore_reset=True, ignore_measurement=True).inverse()
    measured = [inverse.x_output(t.value) if gate.name == "MX" else inverse.z_output(t.value) for gate in circuit if stim.gate_data(gate.name).produces_measurements for t in gate.targets_copy()]
    stabilizers = [p.sign * p[:n] for p in measured] # exact stabilizer basis and their order relevant and not necessarily consistent with the proposed basis
    syndrome_meas = list(range(len(measured)))

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

_PREPARE_CIRCUIT : dict[CircuitFunction, Callable[[Path, stim.Circuit], QECCircuit]] = {
    "StatePrep": prepare_sp_circ,
    "Encoding": prepare_en_circ,
    "Syndrome": prepare_se_circ
}