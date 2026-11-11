import pathlib

import stim

from src.analysis.analyzer import CircuitAnalyzer
from src.structure.interaction_graph import InteractionGraph

STEANE = stim.Circuit((pathlib.Path(__file__).parents[2] / "data/circuits/steane_example.stim").read_text())


def build(circuit: str | stim.Circuit) -> InteractionGraph:
    graph = InteractionGraph()
    CircuitAnalyzer(graph).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit, "StatePrep")
    return graph


def test_single_qubit_gates_measurements_and_resets_add_no_nodes():
    assert build("H 0\nR 1\nM 2\nMR 3").graph == {}


def test_two_qubit_gate_adds_symmetric_edge_with_weight_one():
    assert build("CX 0 1").graph == {0: {1: 1}, 1: {0: 1}}


def test_repeated_interactions_increase_weight_independent_of_direction_and_gate():
    assert build("CX 0 1\nCX 1 0\nCZ 0 1").graph == {0: {1: 3}, 1: {0: 3}}


def test_multiple_target_pairs_are_separate_interactions():
    assert build("CX 0 1 2 3").graph == {0: {1: 1}, 1: {0: 1}, 2: {3: 1}, 3: {2: 1}}


def test_swap_counts_as_interaction():
    assert build("SWAP 0 1").graph == {0: {1: 1}, 1: {0: 1}}


def test_steane():
    assert build(STEANE).graph == {
        0: {2: 1, 5: 1},
        1: {3: 1},
        2: {0: 1, 3: 1},
        3: {1: 1, 4: 1, 2: 1, 5: 1},
        4: {3: 1, 6: 1},
        5: {0: 1, 6: 1, 3: 1},
        6: {4: 1, 5: 1},
    }


def test_steane_graph_is_symmetric_and_total_weight_is_twice_the_number_of_two_qubit_gates():
    graph = build(STEANE).graph
    nr_two_qubit_gates = sum(len(op.target_groups()) for op in STEANE if stim.gate_data(op.name).is_two_qubit_gate)

    assert all(graph[b][a] == weight for a in graph for b, weight in graph[a].items())
    assert sum(weight for neighbors in graph.values() for weight in neighbors.values()) == 2 * nr_two_qubit_gates
