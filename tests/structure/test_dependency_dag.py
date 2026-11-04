import pathlib
import random

import pytest
import stim

from src.analysis.analyzer import CircuitAnalyzer
from src.structure.dependency_dag import DependencyDAG

STEANE = stim.Circuit((pathlib.Path(__file__).parents[2] / "data/circuits/steane_example.stim").read_text())


def build(circuit: str | stim.Circuit) -> DependencyDAG:
    dag = DependencyDAG()
    CircuitAnalyzer(dag).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit)
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
