import pathlib

import pytest
import stim

from src.analysis.quality._decoder import BeliefPropOrdStat
from src.analysis.quality.encoder import EncoderCircuitAnalyzer
from src.models.common import GateKind
from src.models.hardware import SuperconductingQubitsModel

CODE_DIR = pathlib.Path(__file__).parents[3] / "data/circuits/steane-code"
STABILIZERS = [stim.PauliString(s) for s in (CODE_DIR / "stabilizers.txt").read_text().split()]
ENCODERS = sorted((CODE_DIR / "encoding").glob("*.stim"))
STANDARD = stim.Circuit.from_file(CODE_DIR / "encoding/steane-code--standard-encoding.stim")
DURATIONS: dict[GateKind, float] = {"SN": 1.0, "EN": 2.0, "MS": 5.0}
SHOTS = 500


def logicals(circuit: stim.Circuit) -> tuple[list[int], list[tuple[stim.PauliString, stim.PauliString]]]:
    # same extraction as in sandbox.py: input qubits are those whose X and Z images commute with all stabilizers
    tableau = stim.Tableau.from_circuit(circuit, ignore_reset=True)
    qubits = [q for q in range(len(tableau)) if all(tableau.x_output(q).commutes(s) and tableau.z_output(q).commutes(s) for s in STABILIZERS)]
    return qubits, [(tableau.x_output(q), tableau.z_output(q)) for q in qubits]


def analyze(circuit: stim.Circuit, noisy_circuit: stim.Circuit | None = None) -> dict[str, object]:
    # the analyzer only reads the noisy circuit, so it is set directly to control exactly which errors happen
    hw = SuperconductingQubitsModel(circuit.num_qubits, DURATIONS, {2: circuit.num_qubits}, {"H", "CX", "M", "R"})
    hw.noisy_circuit = circuit if noisy_circuit is None else noisy_circuit
    qubits, operators = logicals(circuit)
    return EncoderCircuitAnalyzer(hw, STABILIZERS, operators, SHOTS, BeliefPropOrdStat(hw, STABILIZERS), qubits).analyze()


def test_missing_or_mismatching_logical_qubits_raise():
    hw = SuperconductingQubitsModel(7, DURATIONS, {2: 7}, {"H", "CX", "M", "R"})
    _, operators = logicals(STANDARD)
    decoder = BeliefPropOrdStat(hw, STABILIZERS)

    with pytest.raises(ValueError):
        EncoderCircuitAnalyzer(hw, STABILIZERS, operators, SHOTS, decoder, None)
    with pytest.raises(ValueError):
        EncoderCircuitAnalyzer(hw, STABILIZERS, operators, SHOTS, decoder, [0, 1])


@pytest.mark.parametrize("path", ENCODERS, ids=lambda p: p.stem)
def test_noiseless_encoder_has_no_failures(path: pathlib.Path):
    assert analyze(stim.Circuit.from_file(path)) == {"n_failures": 0, "logical_error_rate": 0.0}


@pytest.mark.parametrize("error", ["X 3", "Z 5", "Y 0"])
def test_single_qubit_error_after_encoding_is_corrected(error: str):
    assert analyze(STANDARD, STANDARD + stim.Circuit(error))["logical_error_rate"] == 0.0


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_every_logical_error_after_encoding_is_detected(pauli: str):
    # the Bell pair with the reference checks X_L and Z_L at once, so X_L, Y_L and Z_L all fail
    _, [(x, z)] = logicals(STANDARD)
    operator = {"X": x, "Z": z, "Y": x * z}[pauli]
    error = stim.Circuit()
    for q in range(len(operator)):
        if operator[q]:
            error.append("XYZ"[operator[q] - 1], [q])

    assert analyze(STANDARD, STANDARD + error)["logical_error_rate"] == 1.0


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_error_on_input_qubit_before_encoding_is_a_logical_error(pauli: str):
    [q], _ = logicals(STANDARD)

    assert analyze(STANDARD, stim.Circuit(f"{pauli} {q}") + STANDARD)["logical_error_rate"] == 1.0
