import pathlib
import random

import pytest
import stim

from src.analysis.analyzer import CircuitAnalyzer
from src.structure.dependency_dag import DependencyDAG

STEANE = stim.Circuit((pathlib.Path(__file__).parents[2] / "data/circuits/steane_example.stim").read_text())


def build(circuit: str | stim.Circuit) -> DependencyDAG:
    dag = DependencyDAG()
    CircuitAnalyzer(dag).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit, "StatePrep")
    return dag


def predecessors(dag: DependencyDAG) -> dict[int, set[int]]:
    return {index: node.predecessors for index, node in dag.dag.items()}


def successors(dag: DependencyDAG) -> dict[int, set[int]]:
    return {index: node.successors for index, node in dag.dag.items()}


def random_circuit(seed: int, nr_qubits: int = 6, nr_gates: int = 40) -> stim.Circuit:
    rng = random.Random(seed)
    circuit = stim.Circuit()
    for _ in range(nr_gates):
        kind = rng.choice(["H", "M", "R", "CX", "CZ"])
        if kind in ("CX", "CZ"):
            circuit.append(kind, rng.sample(range(nr_qubits), 2))
        else:
            circuit.append(kind, [rng.randrange(nr_qubits)])
    return circuit


def test_gates_on_different_qubits_are_independent():
    dag = build("H 0\nH 1")

    assert predecessors(dag) == {0: set(), 1: set()}
    assert successors(dag) == {0: set(), 1: set()}


def test_gates_on_one_qubit_form_a_chain_of_immediate_dependencies():
    dag = build("R 0\nH 0\nM 0")

    assert predecessors(dag) == {0: set(), 1: {0}, 2: {1}}
    assert successors(dag) == {0: {1}, 1: {2}, 2: set()}


def test_two_qubit_gate_depends_on_last_gate_of_both_qubits():
    dag = build("H 0\nH 1\nCX 0 1\nM 0\nM 1")

    assert predecessors(dag) == {0: set(), 1: set(), 2: {0, 1}, 3: {2}, 4: {2}}
    assert successors(dag) == {0: {2}, 1: {2}, 2: {3, 4}, 3: set(), 4: set()}


def test_nodes_store_qubits_of_individual_gates():
    dag = build("CX 0 1 2 3\nH 1")

    assert [node.qubits for node in dag.dag.values()] == [(0, 1), (2, 3), (1,)]


def test_annotations_create_no_nodes():
    dag = build("QUBIT_COORDS(0, 0) 0\nH 0\nTICK\nH 0")

    assert predecessors(dag) == {0: set(), 1: {0}}


def test_dependency_on_each_qubit_is_kept_even_if_implied_by_another_path():
    # gate 0 precedes gate 2 also via gate 1 (qubit 1, then qubit 2), but is its direct predecessor on qubit 0
    dag = build("CX 0 1\nCX 1 2\nCX 0 2")

    assert predecessors(dag)[2] == {0, 1}


def test_steane():
    # 0-6: R 0..6, 7-10: H 0 3 4 5, 11-18: CX (3,1) (0,2) (4,3) (5,0) (4,6) (2,3) (5,6) (3,5)
    assert predecessors(build(STEANE)) == {
        **{i: set() for i in range(7)},
        7: {0},
        8: {3},
        9: {4},
        10: {5},
        11: {8, 1},
        12: {7, 2},
        13: {9, 11},
        14: {10, 12},
        15: {13, 6},
        16: {12, 13},
        17: {14, 15},
        18: {16, 17},
    }


@pytest.mark.parametrize("circuit", [STEANE, *(random_circuit(seed) for seed in range(20))])
def test_dag_invariants(circuit: stim.Circuit):
    dag = build(circuit).dag

    for index, node in dag.items():
        for p in node.predecessors:
            assert p < index  # acyclic: dependencies point backwards in the circuit
            assert index in dag[p].successors  # predecessors and successors are consistent
            assert set(dag[p].qubits) & set(node.qubits)  # dependencies only via shared qubits
        for s in node.successors:
            assert index in dag[s].predecessors
        assert len(node.predecessors) <= len(node.qubits)  # at most one immediate predecessor per qubit


@pytest.mark.parametrize("seed", range(20))
def test_predecessors_are_the_last_earlier_gates_on_each_qubit(seed: int):
    circuit = random_circuit(seed)
    dag = build(circuit).dag
    nodes = list(dag.values())

    for index, node in enumerate(nodes):
        expected = set()
        for qubit in node.qubits:
            earlier_on_qubit = [i for i in range(index) if qubit in nodes[i].qubits]
            if earlier_on_qubit:
                expected.add(earlier_on_qubit[-1])
        assert node.predecessors == expected


def build_scheduled(circuit: str | stim.Circuit) -> DependencyDAG:
    return build(circuit)  # CircuitAnalyzer.run calls finalize()


def schedule(dag: DependencyDAG) -> dict[int, tuple[int | None, int | None]]:
    return {index: node.schedule_interval for index, node in dag.dag.items()}


def test_empty_circuit_has_depth_zero():
    dag = build_scheduled("")

    assert dag.depth == 0
    assert schedule(dag) == {}


def test_independent_gates_are_scheduled_in_first_layer():
    dag = build_scheduled("H 0\nH 1\nH 2")

    assert dag.depth == 1
    assert schedule(dag) == {0: (1, 1), 1: (1, 1), 2: (1, 1)}


def test_chain_on_one_qubit_is_fixed():
    dag = build_scheduled("R 0\nH 0\nM 0")

    assert dag.depth == 3
    assert schedule(dag) == {0: (1, 1), 1: (2, 2), 2: (3, 3)}


def test_depth_is_maximum_over_all_gates_not_last_gate():
    # last gate H 2 has asap 1, but the circuit depth is 3
    dag = build_scheduled("CX 0 1\nH 0\nH 0\nH 2")

    assert dag.depth == 3
    assert schedule(dag) == {0: (1, 1), 1: (2, 2), 2: (3, 3), 3: (1, 3)}


def test_gate_without_successors_can_be_delayed_to_the_end():
    dag = build_scheduled("H 0\nCX 0 1\nH 1\nH 1\nH 2\nCX 2 3")

    assert dag.depth == 4
    assert schedule(dag) == {
        0: (1, 1),
        1: (2, 2),
        2: (3, 3),
        3: (4, 4),
        4: (1, 3),
        5: (2, 4),
    }


def test_steane_schedule():
    dag = build_scheduled(STEANE)

    assert dag.depth == 7
    assert schedule(dag) == {
        0: (1, 2),
        1: (1, 2),
        2: (1, 3),
        3: (1, 1),
        4: (1, 2),
        5: (1, 3),
        6: (1, 4),
        7: (2, 3),
        8: (2, 2),
        9: (2, 3),
        10: (2, 4),
        11: (3, 3),
        12: (3, 4),
        13: (4, 4),
        14: (4, 5),
        15: (5, 5),
        16: (5, 6),
        17: (6, 6),
        18: (7, 7),
    }


@pytest.mark.parametrize("circuit", [STEANE, *(random_circuit(seed) for seed in range(20))])
def test_schedule_invariants(circuit: stim.Circuit):
    dag = build_scheduled(circuit)

    for node in dag.dag.values():
        asap, alap = node.schedule_interval
        assert 1 <= asap <= alap <= dag.depth
        for p in node.predecessors:
            assert dag.dag[p].schedule_interval[0] < asap
            assert dag.dag[p].schedule_interval[1] < alap
    assert dag.depth == max((node.schedule_interval[0] for node in dag.dag.values()), default=0)
    # the critical path is tight: some gates in the first and last layer cannot move
    assert any(node.schedule_interval == (dag.depth, dag.depth) for node in dag.dag.values())
    assert any(node.schedule_interval == (1, 1) for node in dag.dag.values())


@pytest.mark.parametrize("seed", range(20))
def test_schedule_matches_brute_force_over_all_gates_sharing_a_qubit(seed: int):
    dag = build_scheduled(random_circuit(seed))
    nodes = list(dag.dag.values())
    n = len(nodes)

    def shares_qubit(i: int, j: int) -> bool:
        return bool(set(nodes[i].qubits) & set(nodes[j].qubits))

    asap = [0] * n
    for i in range(n):
        asap[i] = max((asap[j] for j in range(i) if shares_qubit(i, j)), default=0) + 1
    depth = max(asap, default=0)
    alap = [0] * n
    for i in reversed(range(n)):
        alap[i] = min((alap[j] for j in range(i + 1, n) if shares_qubit(i, j)), default=depth + 1) - 1

    assert dag.depth == depth
    assert [node.schedule_interval for node in nodes] == list(zip(asap, alap))
