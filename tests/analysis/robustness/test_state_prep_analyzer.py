import pathlib

import pytest
import stim

from src.analysis.robustness._decoder import BeliefPropOrdStat
from src.analysis.robustness.state_prep import StatePreparationCircuitAnalyzer
from src.models.common import GateKind
from src.models.hardware import SuperconductingQubitsModel

CODE_DIR = pathlib.Path(__file__).parents[3] / "data/circuits/steane-code"
STABILIZERS = [stim.PauliString(s) for s in (CODE_DIR / "stabilizers.txt").read_text().split()]
LOGICALS = [stim.PauliString(s) for s in (CODE_DIR / "logicals.txt").read_text().split()]
LOGICAL_OPERATORS = [(LOGICALS[0], LOGICALS[1])]
PREP = stim.Circuit.from_file(CODE_DIR / "zero_non_ft_preparation/steane-code--non-ft-zero-opt.stim")
FT_PREP = stim.Circuit.from_file(CODE_DIR / "zero_ft_preparation/steane-code--ft-zero-heuristic-heuristic.stim")
DURATIONS: dict[GateKind, float] = {"SN": 1.0, "EN": 2.0, "MS": 5.0}
MAX_FAULT_SETS = 1_000_000
MAX_REL_TAIL = 0.01


def hardware(noisy_circuit: stim.Circuit) -> SuperconductingQubitsModel:
    # the analyzer only reads the noisy circuit, so it is set directly to control exactly which errors happen
    hw = SuperconductingQubitsModel(noisy_circuit.num_qubits, DURATIONS, {2: noisy_circuit.num_qubits}, {"H", "CX", "M", "R"})
    hw.noisy_circuit = noisy_circuit
    return hw


def analyze(noisy_circuit: stim.Circuit, state="0", accept_if=None) -> dict[str, object]:
    hw = hardware(noisy_circuit)
    return StatePreparationCircuitAnalyzer(hw.noisy_circuit, STABILIZERS, LOGICAL_OPERATORS, MAX_FAULT_SETS, MAX_REL_TAIL, BeliefPropOrdStat(hw, STABILIZERS), accept_if, state).analyze()


def test_missing_state_raises():
    with pytest.raises(ValueError):
        analyze(PREP, state=None)


@pytest.mark.parametrize("circuit", [PREP, FT_PREP])
def test_noiseless_preparation_has_no_failures(circuit: stim.Circuit):
    assert analyze(circuit).items() >= {"rejection_rate": 0.0, "logical_error_rate": 0.0}.items()


def test_plus_state_is_target_of_x_logical():
    # |+_L> = H^7 |0_L> for the Steane code
    assert analyze(PREP + stim.Circuit("H 0 1 2 3 4 5 6"), state="+")["logical_error_rate"] == 0.0
    assert analyze(PREP + stim.Circuit("H 0 1 2 3 4 5 6\nZ 0 1 2 3 4 5 6"), state="-")["logical_error_rate"] == 0.0


# the DEM measures flips relative to the noiseless circuit, so errors have to be noise (deterministic Paulis are part of the reference)
@pytest.mark.parametrize("error", ["X_ERROR(0.1) 3", "Z_ERROR(0.1) 5", "Y_ERROR(0.1) 0"])
def test_single_qubit_error_is_corrected(error: str):
    assert analyze(PREP + stim.Circuit(error))["logical_error_rate"] == pytest.approx(0.0)


def test_logical_error_is_not_corrected():
    assert analyze(PREP + stim.Circuit("E(0.1) X0 X1 X2 X3 X4 X5 X6"))["logical_error_rate"] == pytest.approx(0.1)
    # Z_L does not change |0_L>
    assert analyze(PREP + stim.Circuit("E(0.1) Z0 Z1 Z2 Z3 Z4 Z5 Z6"))["logical_error_rate"] == pytest.approx(0.0)


def test_rejected_shots_are_discarded():
    # with probability 0.3 the ancilla flips and copies X_L onto the data, the verifier sees the flipped ancilla
    flagged_error = PREP + stim.Circuit("X_ERROR(0.3) 7\nCX 7 0 7 1 7 2 7 3 7 4 7 5 7 6\nM 7")

    without_verification = analyze(flagged_error)
    assert without_verification["rejection_rate"] == 0.0
    assert 0.2 < without_verification["logical_error_rate"] < 0.4

    with_verification = analyze(flagged_error, accept_if={0: 0})
    assert 0.2 < with_verification["rejection_rate"] < 0.4
    assert with_verification["logical_error_rate"] == 0.0
