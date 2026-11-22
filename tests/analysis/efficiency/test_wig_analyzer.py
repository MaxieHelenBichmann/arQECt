import math
import pathlib

import pytest
import stim

from src.analysis.efficiency.wig import InteractionGraphAnalyzer
from src.analysis.processor import CircuitProcessor
from src.models.common import GateKind
from src.models.hardware import NeutralAtomsModel, SuperconductingQubitsModel, TrappedIonsModel
from src.structure.interaction_graph import InteractionGraph

STEANE = stim.Circuit((pathlib.Path(__file__).parents[3] / "data/circuits/steane_example.stim").read_text())
NATIVE_GATES: set[str] = {"CX", "H", "M", "R"}
DURATIONS: dict[GateKind, float] = {"SN": 1.0, "EN": 2.0, "MS": 5.0}

# star around qubit 0: 0-1 (weight 2), 0-2 (1), 0-3 (1)
STAR = "CX 0 1\nCX 0 1\nCX 0 2\nCX 0 3"
TRIANGLE = "CX 0 1\nCX 1 2\nCX 0 2"


def analyze(circuit: str | stim.Circuit, hardware) -> dict[str, object]:
    wig = InteractionGraph()
    CircuitProcessor(wig).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit)
    result = InteractionGraphAnalyzer(wig, hardware).analyze()
    assert result["compilation"] == {}
    return result["runtime"]


def test_no_2q_gates_gives_undefined_metrics():
    result = analyze("H 0\nM 0", TrappedIonsModel(2, 1, NATIVE_GATES))

    assert len(result) == 10
    assert all(value is None for value in result.values())


@pytest.mark.parametrize("hardware", [TrappedIonsModel(2, 1, NATIVE_GATES), NeutralAtomsModel(1, 1, NATIVE_GATES)])
def test_partner_capacity_with_capacity_one(hardware):
    result = analyze(STAR, hardware)

    assert result["fraction_of_partner_capacity_exceeding_qubits"] == 0.25  # only qubit 0 (3 partners)
    assert result["weighted_fraction_of_partner_capacity_exceeding_qubits"] == pytest.approx((1 - 2 / 4) / 4)  # qubit 0 keeps 1 local
    assert result["fraction_of_effective_partner_capacity_exceeding_qubits"] == 0.25  # qubit 0: 2^1.5 effective partners
    assert result["max_partner_capacity_ratio"] == 3.0


def test_partner_capacity_superconducting_assigns_highest_degrees_to_busiest_qubits():
    result = analyze(STAR, SuperconductingQubitsModel(4, DURATIONS, {3: 1, 1: 3}, NATIVE_GATES))

    assert result["fraction_of_partner_capacity_exceeding_qubits"] == 0.0
    assert result["weighted_fraction_of_partner_capacity_exceeding_qubits"] == 0.0
    assert result["fraction_of_effective_partner_capacity_exceeding_qubits"] == 0.0
    assert result["max_partner_capacity_ratio"] == 1.0


def test_interaction_spread():
    result = analyze(STAR, TrappedIonsModel(2, 1, NATIVE_GATES))

    assert result["average_interaction_partner_diversity"] == 0.5  # (3/3 + 1/3 + 1/3 + 1/3) / 4
    assert result["average_interaction_entropy"] == pytest.approx(1.5 / 4)  # qubit 0: H(1/2, 1/4, 1/4) = 1.5
    assert result["average_effective_partner_number"] == pytest.approx((2**1.5 + 3) / 4)


def test_triangle_spread():
    result = analyze(TRIANGLE, TrappedIonsModel(2, 1, NATIVE_GATES))

    assert result["average_interaction_partner_diversity"] == 1.0  # all-to-all
    assert result["average_interaction_entropy"] == 1.0
    assert result["average_effective_partner_number"] == 2.0


def test_statically_servable_interaction_fraction():
    # capacity 1: only the heaviest edge 0-1 can be static
    assert analyze(STAR, TrappedIonsModel(2, 1, NATIVE_GATES))["statically_servable_interaction_fraction"] == 0.5
    # capacity 3: everything static
    assert analyze(STAR, TrappedIonsModel(4, 1, NATIVE_GATES))["statically_servable_interaction_fraction"] == 1.0
    assert analyze(STAR, SuperconductingQubitsModel(4, DURATIONS, {3: 1, 1: 3}, NATIVE_GATES))["statically_servable_interaction_fraction"] == 1.0


def test_capacity_constrained_qubit_partition():
    # capacity 1: groups {0, 1}, {2}, {3} -> edges 0-2 and 0-3 are cut
    assert analyze(STAR, TrappedIonsModel(2, 1, NATIVE_GATES))["capacity_constrained_qubit_partition"] == 0.5
    # capacity 3: one group for all
    assert analyze(STAR, TrappedIonsModel(4, 1, NATIVE_GATES))["capacity_constrained_qubit_partition"] == 0.0
    # superconducting star: qubit 0 is the center and all edges go through it
    assert analyze(STAR, SuperconductingQubitsModel(4, DURATIONS, {3: 1, 1: 3}, NATIVE_GATES))["capacity_constrained_qubit_partition"] == 0.0


def test_superconducting_partition_cuts_edges_not_touching_the_center():
    # triangle with degree-2 hardware: one group with center 0, edge 1-2 does not touch the center
    result = analyze(TRIANGLE, SuperconductingQubitsModel(3, DURATIONS, {2: 3}, NATIVE_GATES))

    assert result["capacity_constrained_qubit_partition"] == pytest.approx(1 / 3)


def test_non_bipartite_interaction_fraction():
    assert analyze(STAR, TrappedIonsModel(2, 1, NATIVE_GATES))["non_bipartite_interaction_fraction"] == 0.0
    assert analyze(TRIANGLE, TrappedIonsModel(2, 1, NATIVE_GATES))["non_bipartite_interaction_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("hardware", [
    TrappedIonsModel(2, 2, NATIVE_GATES), NeutralAtomsModel(2, 2, NATIVE_GATES), SuperconductingQubitsModel(7, DURATIONS, {2: 6, 3: 1}, NATIVE_GATES),
])
def test_steane_metrics_are_within_their_ranges(hardware):
    result = analyze(STEANE, hardware)

    for key in ["fraction_of_partner_capacity_exceeding_qubits", "weighted_fraction_of_partner_capacity_exceeding_qubits",
                "fraction_of_effective_partner_capacity_exceeding_qubits", "average_interaction_partner_diversity",
                "statically_servable_interaction_fraction", "capacity_constrained_qubit_partition"]:
        assert 0 <= result[key] <= 1
    assert 0 <= result["non_bipartite_interaction_fraction"] <= 0.5
    assert 0 <= result["average_interaction_entropy"] <= math.log2(6)
    assert 1 <= result["average_effective_partner_number"] <= 6
    assert result["max_partner_capacity_ratio"] > 0
