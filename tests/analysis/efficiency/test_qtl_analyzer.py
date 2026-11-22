import pathlib

import pytest
import stim

from src.analysis.efficiency.qtl import QubitTimelinesAnalyzer
from src.analysis.processor import CircuitProcessor
from src.models.common import GateKind
from src.models.hardware import NeutralAtomsModel, SuperconductingQubitsModel, TrappedIonsModel
from src.structure.qubit_timelines import QubitTimelines

STEANE = stim.Circuit((pathlib.Path(__file__).parents[3] / "data/circuits/steane_example.stim").read_text())
NATIVE_GATES: set[str] = {"CX", "H", "M", "R"}
DURATIONS: dict[GateKind, float] = {"SN": 1.0, "EN": 2.0, "MS": 5.0}

# timelines: 0: EN(1) EN(2) EN(1) SN MS SN | 1: EN(0) EN(0) MS | 2: EN(0)
CIRCUIT = "CX 0 1\nCX 0 2\nCX 0 1\nH 0\nM 0\nH 0\nM 1"


def analyze(circuit: str | stim.Circuit, hardware) -> dict[str, object]:
    qtl = QubitTimelines()
    CircuitProcessor(qtl).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit)
    result = QubitTimelinesAnalyzer(qtl, hardware).analyze()
    return {**result["runtime"], **result["compilation"]}


def test_metrics_are_split_into_runtime_and_compilation():
    qtl = QubitTimelines()
    CircuitProcessor(qtl).run(stim.Circuit(CIRCUIT))
    result = QubitTimelinesAnalyzer(qtl, TrappedIonsModel(2, 1, NATIVE_GATES)).analyze()

    assert set(result) == {"runtime", "compilation"}
    assert set(result["compilation"]) == {"average_1q_run_length"}


def test_average_interaction_reuse_distance():
    # qubit 0 meets qubit 1 again after 2 operations, qubit 1 meets qubit 0 again directly -> (2 + 1) / 2
    assert analyze(CIRCUIT, TrappedIonsModel(2, 1, NATIVE_GATES))["average_interaction_reuse_distance"] == 1.5
    assert analyze("CX 0 1\nCX 1 2", TrappedIonsModel(2, 1, NATIVE_GATES))["average_interaction_reuse_distance"] == 0.0


@pytest.mark.parametrize("hardware, expected", [
    (TrappedIonsModel(2, 1, NATIVE_GATES), 0.5),  # local capacity 1: qubit 0's reuse has 1 other partner in between
    (TrappedIonsModel(3, 1, NATIVE_GATES), 1.0),  # local capacity 2: both reuses fit
    (NeutralAtomsModel(1, 1, NATIVE_GATES), 0.5),  # only back-to-back reuses fit
    (SuperconductingQubitsModel(3, DURATIONS, {2: 3}, NATIVE_GATES), 1.0),  # average degree 2
])
def test_capacity_constrained_reuse(hardware, expected: float):
    assert analyze(CIRCUIT, hardware)["capacity_constrained_reuse"] == expected


def test_capacity_constrained_reuse_without_reuse_is_zero():
    assert analyze("CX 0 1\nCX 1 2", TrappedIonsModel(2, 1, NATIVE_GATES))["capacity_constrained_reuse"] == 0.0


def test_partner_switches_per_qubit():
    # switches: qubit 0 to 2 (first meeting), qubit 0 back to 1 and qubit 1 back to 0 (qubit 2 in between on qubit 0's side)
    result = analyze(CIRCUIT, TrappedIonsModel(2, 1, NATIVE_GATES))
    assert result["partner_switches_per_qubit"] == 1.0  # 3 switches / 3 interacting qubits
    assert result["capacity_constrained_partner_switches_per_qubit"] == 1.0  # local capacity 1: all exceed

    result = analyze(CIRCUIT, TrappedIonsModel(3, 1, NATIVE_GATES))
    assert result["capacity_constrained_partner_switches_per_qubit"] == pytest.approx(1 / 3)  # only the first meeting


def test_back_to_back_partner_is_no_switch():
    result = analyze("CX 0 1\nCX 0 1\nCX 1 0", TrappedIonsModel(2, 1, NATIVE_GATES))

    assert result["partner_switches_per_qubit"] == 0.0
    assert result["capacity_constrained_partner_switches_per_qubit"] == 0.0


def test_share_mid_circuit_measurements():
    # M 0 is followed by H 0, M 1 is the last operation on qubit 1
    assert analyze(CIRCUIT, TrappedIonsModel(2, 1, NATIVE_GATES))["share_mid_circuit_measurements"] == 0.5
    assert analyze("H 0", TrappedIonsModel(2, 1, NATIVE_GATES))["share_mid_circuit_measurements"] == 0.0


def test_activity_imbalance():
    # 6, 3, 1 operations -> 6 / (10 / 3)
    assert analyze(CIRCUIT, TrappedIonsModel(2, 1, NATIVE_GATES))["activity_imbalance"] == pytest.approx(1.8)
    assert analyze("CX 0 1\nCX 1 0", TrappedIonsModel(2, 1, NATIVE_GATES))["activity_imbalance"] == 1.0


def test_average_1q_run_length():
    # qubit 0: SN SN EN SN -> runs 2, 1 | qubit 1: EN SN -> run 1
    assert analyze("H 0\nR 0\nCX 0 1\nH 0\nH 1", TrappedIonsModel(2, 1, NATIVE_GATES))["average_1q_run_length"] == pytest.approx(4 / 3)
    assert analyze("CX 0 1", TrappedIonsModel(2, 1, NATIVE_GATES))["average_1q_run_length"] == 0.0


@pytest.mark.parametrize("hardware", [
    TrappedIonsModel(2, 2, NATIVE_GATES), NeutralAtomsModel(2, 2, NATIVE_GATES), SuperconductingQubitsModel(7, DURATIONS, {2: 6, 3: 1}, NATIVE_GATES),
])
def test_steane_metrics_are_within_their_ranges(hardware):
    result = analyze(STEANE, hardware)

    assert 0 <= result["capacity_constrained_reuse"] <= 1
    assert 0 <= result["capacity_constrained_partner_switches_per_qubit"] <= result["partner_switches_per_qubit"]
    assert 0 <= result["share_mid_circuit_measurements"] <= 1
    assert 1 <= result["activity_imbalance"] <= 7
    assert result["average_1q_run_length"] >= 1
