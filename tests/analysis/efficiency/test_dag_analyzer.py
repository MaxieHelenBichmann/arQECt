import pathlib

import pytest
import stim

from src.analysis.efficiency.dag import DependencyDAGAnalyzer
from src.analysis.processor import CircuitProcessor
from src.models.common import GateKind
from src.models.hardware import NeutralAtomsModel, SuperconductingQubitsModel, TrappedIonsModel
from src.structure.dependency_dag import DependencyDAG

STEANE = stim.Circuit((pathlib.Path(__file__).parents[3] / "data/circuits/steane_example.stim").read_text())
NATIVE_GATES: set[str] = {"CX", "H", "M", "R"}
DURATIONS: dict[GateKind, float] = {"SN": 1.0, "EN": 2.0, "MS": 5.0}

# 0: H 0 (1,1) | 1: CX 0 1 (2,2) | 2: H 1 (3,3) | 3: H 1 (4,4) | 4: H 2 (1,3) | 5: CX 2 3 (2,4)   (asap, alap)
CIRCUIT = "H 0\nCX 0 1\nH 1\nH 1\nH 2\nCX 2 3"


def sc() -> SuperconductingQubitsModel:
    return SuperconductingQubitsModel(4, DURATIONS, {2: 4}, NATIVE_GATES)


def analyze(circuit: str | stim.Circuit, hardware) -> dict[str, object]:
    dag = DependencyDAG()
    CircuitProcessor(dag).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit)
    result = DependencyDAGAnalyzer(dag, hardware).analyze()
    return {**result["runtime"], **result["compilation"]}


def test_metrics_are_split_into_runtime_and_compilation():
    dag = DependencyDAG()
    CircuitProcessor(dag).run(stim.Circuit(CIRCUIT))
    result = DependencyDAGAnalyzer(dag, sc()).analyze()

    assert set(result) == {"runtime", "compilation"}
    assert set(result["compilation"]) == {"average_scheduling_slack", "average_weighted_scheduling_slack", "no_slack_fraction"}


def test_depths():
    result = analyze(CIRCUIT, sc())

    assert result["critical_path_depth"] == 4
    assert result["2q_depth"] == 1


def test_2q_depth_counts_chained_2q_gates():
    assert analyze("CX 0 1\nCX 1 2\nH 2\nCX 2 3", sc())["2q_depth"] == 3


def test_critical_path_duration_uses_superconducting_gate_durations():
    # H 0 (1) -> CX 0 1 (2) -> H 1 (1) -> H 1 (1) = 5
    assert analyze(CIRCUIT, sc())["critical_path_duration"] == 5.0
    # R (1) -> CX (2) -> M (5) = 8
    assert analyze("R 0\nCX 0 1\nM 1", sc())["critical_path_duration"] == 8.0


def test_critical_path_duration_is_none_without_gate_durations():
    assert analyze(CIRCUIT, TrappedIonsModel(2, 1, NATIVE_GATES))["critical_path_duration"] is None
    assert analyze(CIRCUIT, NeutralAtomsModel(1, 1, NATIVE_GATES))["critical_path_duration"] is None


def test_scheduling_slack():
    result = analyze(CIRCUIT, sc())

    assert result["average_scheduling_slack"] == pytest.approx(4 / 6)  # slack 2 for gates 4 and 5
    assert result["average_weighted_scheduling_slack"] == pytest.approx((2 * 1 + 2 * 2) / 6)  # gate 5 is a 2q gate
    assert result["no_slack_fraction"] == pytest.approx(4 / 6)


def test_parallelism_saturation_and_capacity_constrained_depth_bound():
    sc_result = analyze(CIRCUIT, sc())
    assert sc_result["parallelism_saturation"] == 0
    assert sc_result["capacity_constrained_depth_bound"] == 4

    # ion slots: 4 1q gates (1 each) + 2 2q gates (2 each) = 8, 1 zone with capacity 2 -> 4 layers
    assert analyze(CIRCUIT, TrappedIonsModel(2, 1, NATIVE_GATES))["parallelism_saturation"] == 4
    assert analyze(CIRCUIT, TrappedIonsModel(4, 2, NATIVE_GATES))["parallelism_saturation"] == 1

    # 2 2q gates, 1 entanglement pair -> 2 layers; 3 readouts, readout capacity 1 -> 3 layers
    na_result = analyze(CIRCUIT + "\nM 0 1 2", NeutralAtomsModel(1, 1, NATIVE_GATES))
    assert na_result["parallelism_saturation"] == 3

    # 12 2q gates (6 per layer) exceed the depth 2 bound: 24 ion slots / (2 * 1) = 12 layers
    ti_result = analyze("CX 0 1 2 3 4 5 6 7 8 9 10 11\nCX 0 1 2 3 4 5 6 7 8 9 10 11", TrappedIonsModel(2, 1, NATIVE_GATES))
    assert ti_result["critical_path_depth"] == 2
    assert ti_result["capacity_constrained_depth_bound"] == 12


def test_window_congestion_is_zero_without_zone_limits():
    result = analyze("H 0 1 2 3 4 5\nH 0 1 2 3 4 5", sc())

    assert result["min_scheduling_window_congestion"] == 0.0
    assert result["max_scheduling_window_congestion"] == 0.0


def test_window_congestion_trapped_ions():
    # 3 rigid 1q gates (3 ion slots) in each of the 2 layers, but only 2 slots
    result = analyze("H 0 1 2\nH 0 1 2", TrappedIonsModel(2, 1, NATIVE_GATES))
    assert result["min_scheduling_window_congestion"] == 1.0
    assert result["max_scheduling_window_congestion"] == 1.0

    # ion slots of must-execute gates never exceed the window capacity, but layers 2, 3, 4 have 5, 4, 3 slots of may-execute gates
    result = analyze(CIRCUIT, TrappedIonsModel(2, 1, NATIVE_GATES))
    assert result["min_scheduling_window_congestion"] == 0.0
    assert result["max_scheduling_window_congestion"] == 0.75

    # 2q gates occupy 2 ion slots: 2 parallel CX need 4 slots
    assert analyze("CX 0 1 2 3\nCX 0 1 2 3", TrappedIonsModel(2, 1, NATIVE_GATES))["min_scheduling_window_congestion"] == 1.0
    assert analyze("CX 0 1 2 3\nCX 0 1 2 3", TrappedIonsModel(2, 2, NATIVE_GATES))["min_scheduling_window_congestion"] == 0.0


def test_window_congestion_neutral_atoms_counts_only_2q_gates_and_readouts():
    # 2 rigid layers each (depth >= 2, see test_depth_one_circuit_qubit_reuse)
    assert analyze("CX 0 1 2 3\nCX 0 1 2 3", NeutralAtomsModel(1, 1, NATIVE_GATES))["min_scheduling_window_congestion"] == 1.0
    assert analyze("CX 0 1 2 3\nCX 0 1 2 3", NeutralAtomsModel(2, 1, NATIVE_GATES))["min_scheduling_window_congestion"] == 0.0
    assert analyze("M 0 1\nM 0 1", NeutralAtomsModel(1, 1, NATIVE_GATES))["max_scheduling_window_congestion"] == 1.0
    assert analyze("H 0 1 2 3\nH 0 1 2 3", NeutralAtomsModel(1, 1, NATIVE_GATES))["max_scheduling_window_congestion"] == 0.0


def test_qubit_reuse():
    result = analyze(CIRCUIT, sc())

    # asap layers: {0, 2}, {0, 1, 2, 3}, {1}, {1} -> reuse 1, 1/4, 1
    assert result["average_asap_qubit_reuse"] == pytest.approx((1 + 0.25 + 1) / 3)
    # alap layers: {0}, {0, 1}, {1, 2}, {1, 2, 3} -> reuse 1, 1/2, 1
    assert result["average_alap_qubit_reuse"] == pytest.approx((1 + 0.5 + 1) / 3)


def test_asap_qubit_interaction():
    # one window over all 4 layers: pairs {0, 1}, {2, 3} out of C(4, 2) = 6 possible pairs
    assert analyze(CIRCUIT, sc())["average_asap_qubit_interaction"] == pytest.approx(2 / 6)
    # a triangle is all-to-all within its window
    assert analyze("CX 0 1\nCX 1 2\nCX 0 2", sc())["average_asap_qubit_interaction"] == 1.0
    # fewer than 3 interacting qubits -> no window counts
    assert analyze("CX 0 1\nCX 0 1", sc())["average_asap_qubit_interaction"] == 0.0


def test_min_idle_steps():
    assert analyze(CIRCUIT, sc())["min_idle_steps"] == 0
    # qubit 0 has to wait for the two H gates on qubit 1 before the second CX
    assert analyze("CX 0 1\nH 1\nH 1\nCX 0 1", sc())["min_idle_steps"] == 2


def test_depth_one_circuit_has_no_qubit_reuse():
    # no consecutive layers to compare
    result = analyze("H 0 1", sc())

    assert result["average_asap_qubit_reuse"] == 0.0
    assert result["average_alap_qubit_reuse"] == 0.0


@pytest.mark.parametrize("hardware", [sc(), TrappedIonsModel(2, 2, NATIVE_GATES), NeutralAtomsModel(2, 2, NATIVE_GATES)])
def test_steane_metrics_are_within_their_ranges(hardware):
    result = analyze(STEANE, hardware)
    depth = result["critical_path_depth"]

    assert 0 <= result["2q_depth"] <= depth
    assert 0 <= result["average_scheduling_slack"] <= depth - 1
    assert 0 <= result["average_weighted_scheduling_slack"] <= 2 * (depth - 1)
    assert result["capacity_constrained_depth_bound"] >= depth
    for key in ["no_slack_fraction", "min_scheduling_window_congestion", "max_scheduling_window_congestion",
                "average_asap_qubit_reuse", "average_alap_qubit_reuse", "average_asap_qubit_interaction"]:
        assert 0 <= result[key] <= 1
    # every gate that must execute in a window may also execute in one of its layers
    assert result["min_scheduling_window_congestion"] == 0 or result["max_scheduling_window_congestion"] > 0
