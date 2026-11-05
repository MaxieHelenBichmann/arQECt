import pathlib
import random

import pytest
import stim

from src.analysis.analyzer import CircuitAnalyzer
from src.models.common import GateKind
from src.models.hardware.superconducting_qubits import SuperconductingQubitsModel
from src.models.noise.noise import NoiseModel

STEANE = stim.Circuit((pathlib.Path(__file__).parents[2] / "data/circuits/steane_example.stim").read_text())
STEANE_MEASURED = STEANE + stim.Circuit("M 0 1 2 3 4 5 6")

DURATIONS: dict[GateKind, float] = {"SN": 1.0, "EN": 2.0, "MS": 5.0}
T1_T2 = (1000.0, 800.0)  # in units of the gate durations


def make_noise_model(
    hardware_event: tuple = ("Depolarize", (0.01,)),
    single_qubit_gate: tuple = ("None", ()),
    two_qubit_gate: tuple = ("None", ()),
    measurement: tuple = ("None", ()),
    reset: tuple = ("None", ()),
) -> NoiseModel:
    return NoiseModel.from_specs(
        single_qubit_gate=single_qubit_gate,
        two_qubit_gate=two_qubit_gate,
        measurement=measurement,
        reset=reset,
        idle=("ThermalRelaxation", T1_T2),
        hardware_event=hardware_event,
    )


def make_model(nr_qubits: int, topology: dict[int, int], **noise_kwargs) -> SuperconductingQubitsModel:
    return SuperconductingQubitsModel(nr_qubits, DURATIONS, topology, make_noise_model(**noise_kwargs))


def run(
    circuit: str | stim.Circuit, nr_qubits: int, topology: dict[int, int], **noise_kwargs
) -> SuperconductingQubitsModel:
    model = make_model(nr_qubits, topology, **noise_kwargs)
    CircuitAnalyzer(model).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit)
    return model


def idle(model: SuperconductingQubitsModel, qubit: int, time: float) -> stim.Circuit:
    channel = model.noise_model.idle_noise(time)
    circuit = stim.Circuit()
    circuit.append(channel.gate_name, [qubit], channel.gate_arguments)
    return circuit


def without_noise(circuit: stim.Circuit) -> stim.Circuit:
    result = stim.Circuit()
    for op in circuit:
        data = stim.gate_data(op.name)
        if not (data.is_noisy_gate and not data.produces_measurements):
            result.append(op)
    return result


def with_edges(model: SuperconductingQubitsModel, edges: dict[tuple[int, int], int]) -> SuperconductingQubitsModel:
    for (a, b), time_step in edges.items():
        model._neighborhoods[a][b] = time_step
        model._neighborhoods[b][a] = time_step
    return model


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)


# ---------------------------------------------------------------- construction


def test_every_qubit_starts_with_an_empty_neighborhood_and_no_idle_time():
    model = make_model(3, {2: 3})

    assert model._neighborhoods == {0: {}, 1: {}, 2: {}}
    assert model._idle_time_steps == {0: 0, 1: 0, 2: 0}


# ---------------------------------------------------------------- single-qubit gates


def test_single_qubit_gates_measurements_and_resets_do_not_change_neighborhoods():
    model = run("R 0\nH 0\nM 0\nMR 1", 2, {1: 2})

    assert model._neighborhoods == {0: {}, 1: {}}
    assert model.hardware_events == {"virtual_swap": 0}


# ---------------------------------------------------------------- two-qubit gates and neighborhoods


def test_two_qubit_gate_with_room_connects_qubits_without_swap():
    model = run("CX 0 1", 3, {1: 3})

    assert model._neighborhoods == {0: {1: 0}, 1: {0: 0}, 2: {}}
    assert model.hardware_events == {"virtual_swap": 0}


def test_two_qubit_gate_cache_hit_refreshes_lru_time_independent_of_direction():
    model = run("CX 0 1\nCZ 1 0", 2, {1: 2})

    assert model._neighborhoods == {0: {1: 1}, 1: {0: 1}}
    assert model.hardware_events == {"virtual_swap": 0}


def test_full_neighborhood_moves_target_to_seat_of_least_recently_used_neighbor_of_source():
    # degree 2: qubit 0 is full with neighbors 1 (older) and 2, qubit 3 takes the seat of 1
    model = run("CX 0 1\nCX 0 2\nCX 0 3", 4, {2: 4})

    assert model._neighborhoods == {0: {2: 1, 3: 2}, 1: {}, 2: {0: 1}, 3: {0: 2}}
    assert model.hardware_events == {"virtual_swap": 1}


def test_source_without_neighbors_is_moved_next_to_target_instead():
    # degree 1: qubit 0 is full, qubit 2 has no neighbors to swap with, so it takes the seat of 1
    model = run("CX 0 1\nCX 2 0", 3, {1: 3})

    assert model._neighborhoods == {0: {2: 1}, 1: {}, 2: {0: 1}}
    assert model.hardware_events == {"virtual_swap": 1}


def test_ties_of_least_recently_used_neighbors_are_broken_randomly():
    displaced = set()
    for seed in range(30):
        random.seed(seed)
        model = with_edges(make_model(4, {2: 4}), {(0, 1): -1, (0, 2): -1})
        model.process(stim.CircuitInstruction("CX", [0, 3]))
        displaced |= {q for q in (1, 2) if q not in model._neighborhoods[0]}

    assert displaced == {1, 2}


# ---------------------------------------------------------------- fits


@pytest.mark.parametrize(
    "grown, expected",
    [((), True), ((4,), True), ((1,), False), ((0,), False), ((3, 4), False)],
)
def test_fits_checks_whether_neighborhood_sizes_can_be_assigned_to_the_topology(grown: tuple[int, ...], expected: bool):
    # topology seats [3, 2, 2, 2, 2], neighborhood sizes [3, 2, 2, 2, 1]
    model = make_model(5, {3: 1, 2: 4})
    for qubit, size in enumerate([3, 2, 2, 2, 1]):
        model._neighborhoods[qubit] = {100 + i: 0 for i in range(size)}

    assert model._fits(grown) is expected


# ---------------------------------------------------------------- swap


def test_swap_exchanges_seats_and_resets_times_of_pairs_that_never_interacted():
    model = with_edges(make_model(7, {3: 7}), {(0, 1): 0, (0, 2): 5, (1, 3): 7, (5, 6): 8})

    model._swap(1, 5)

    assert model._neighborhoods == {
        0: {2: 5, 5: -1},
        1: {6: -1},
        2: {0: 5},
        3: {5: -1},
        4: {},
        5: {0: -1, 3: -1},
        6: {1: -1},
    }


def test_swap_keeps_times_of_shared_neighbors_and_of_the_swapped_pair():
    model = with_edges(make_model(10, {3: 10}), {(1, 5): 4, (1, 9): 2, (5, 9): 3})

    model._swap(1, 5)

    assert model._neighborhoods[1] == {5: 4, 9: 2}
    assert model._neighborhoods[5] == {1: 4, 9: 3}
    assert model._neighborhoods[9] == {1: 2, 5: 3}


def test_swap_with_qubit_without_neighbors_moves_the_neighborhood():
    model = with_edges(make_model(4, {2: 4}), {(0, 1): 0, (1, 3): 1})

    model._swap(1, 2)

    assert model._neighborhoods == {0: {2: -1}, 1: {}, 2: {0: -1, 3: -1}, 3: {2: -1}}


@pytest.mark.parametrize("seed", range(20))
def test_swap_is_a_relabeling_that_preserves_symmetry_and_pair_history(seed: int):
    rng = random.Random(seed)
    nr_qubits = 7
    pairs = [(a, b) for a in range(nr_qubits) for b in range(a + 1, nr_qubits)]
    edges = {pair: rng.randint(0, 99) for pair in rng.sample(pairs, 10)}
    model = with_edges(make_model(nr_qubits, {6: nr_qubits}), edges)
    sizes_before = sorted(len(n) for n in model._neighborhoods.values())
    qubit_a, qubit_b = rng.sample(range(nr_qubits), 2)

    model._swap(qubit_a, qubit_b)

    neighborhoods = model._neighborhoods
    assert set(neighborhoods) == set(range(nr_qubits))
    assert sorted(len(n) for n in neighborhoods.values()) == sizes_before
    for p, neighborhood in neighborhoods.items():
        for q, time_step in neighborhood.items():
            assert neighborhoods[q][p] == time_step
            assert time_step == edges.get((min(p, q), max(p, q)), -1)


# ---------------------------------------------------------------- idle noise


def test_idle_noise_is_applied_once_for_the_accumulated_idle_time_before_the_next_gate():
    model = run("H 0\nH 1\nCX 0 1\nM 1\nH 0", 2, {1: 2})

    expected = stim.Circuit("H 0")
    expected += idle(model, 1, DURATIONS["SN"])  # qubit 1 waited for H 0
    expected += stim.Circuit("H 1")
    expected += idle(model, 0, DURATIONS["SN"])  # qubit 0 waited for H 1
    expected += stim.Circuit("CX 0 1\nM 1")
    expected += idle(model, 0, DURATIONS["MS"])  # qubit 0 waited for M 1
    expected += stim.Circuit("H 0")
    assert model.noisy_circuit == expected


def test_two_qubit_gates_advance_idle_time_by_their_duration():
    model = run("CX 0 1\nH 2", 3, {1: 3})

    assert model.noisy_circuit == stim.Circuit("CX 0 1") + idle(model, 2, DURATIONS["EN"]) + stim.Circuit("H 2")


def test_no_idle_noise_without_idle_time():
    model = run("H 0\nS 0\nM 0", 1, {1: 1})

    assert model.noisy_circuit == stim.Circuit("H 0\nS 0\nM 0")


def test_idle_time_is_reset_for_used_qubits_and_kept_for_waiting_ones():
    model = run("H 0\nCX 0 1", 3, {1: 3})

    assert model._idle_time_steps == {0: 0, 1: 0, 2: DURATIONS["SN"] + DURATIONS["EN"]}


# ---------------------------------------------------------------- gate and hardware noise


def test_swap_noise_is_applied_to_both_swapped_qubits_before_the_gate():
    model = run("CX 0 1\nCX 0 2", 3, {1: 3})

    expected = stim.Circuit("CX 0 1")
    expected += idle(model, 2, DURATIONS["EN"])
    expected += stim.Circuit("DEPOLARIZE1(0.01) 1 2\nCX 0 2")
    assert model.noisy_circuit == expected


def test_gate_reset_measurement_and_two_qubit_noise():
    model = run(
        "R 0\nH 0\nCX 0 1\nM 0",
        2,
        {1: 2},
        single_qubit_gate=("Depolarize", (0.001,)),
        two_qubit_gate=("Depolarize", (0.005,)),
        measurement=("Measurement", (0.03,)),
        reset=("BitFlip", (0.02,)),
    )

    assert without_noise(model.noisy_circuit) == stim.Circuit("R 0\nH 0\nCX 0 1\nM(0.03) 0")
    assert "X_ERROR(0.02) 0" in str(model.noisy_circuit)
    assert "DEPOLARIZE1(0.001) 0" in str(model.noisy_circuit)
    assert "DEPOLARIZE2(0.005) 0 1" in str(model.noisy_circuit)


def test_no_hardware_event_noise_adds_no_identity_instructions():
    model = run("CX 0 1\nCX 0 2", 3, {1: 3}, hardware_event=("None", ()))

    assert model.hardware_events == {"virtual_swap": 1}
    assert "I " not in str(model.noisy_circuit)


# ---------------------------------------------------------------- small circuits


@pytest.mark.parametrize("topology", [{1: 7}, {2: 7}, {3: 2, 2: 5}, {4: 7}, {6: 7}])
@pytest.mark.parametrize("seed", range(5))
def test_steane_neighborhood_invariants_after_every_gate(topology: dict[int, int], seed: int):
    random.seed(seed)
    model = make_model(7, topology)
    slots = sorted((d for d, count in topology.items() for _ in range(count)), reverse=True)
    violations = []

    class InvariantCheck:
        def process(self, gate: stim.CircuitInstruction) -> None:
            neighborhoods = model._neighborhoods
            if not all(neighborhoods[q].get(p) == t for p in neighborhoods for q, t in neighborhoods[p].items()):
                violations.append(("asymmetric neighborhoods", gate))
            sizes = sorted((len(n) for n in neighborhoods.values()), reverse=True)
            if any(s > d for s, d in zip(sizes, slots)):
                violations.append(("neighborhoods exceed topology", gate))
            if stim.gate_data(gate.name).is_two_qubit_gate:
                src, tgt = (t.value for t in gate.targets_copy())
                if tgt not in neighborhoods[src]:
                    violations.append(("two-qubit gate between non-neighbors", gate))

    CircuitAnalyzer(model, InvariantCheck()).run(STEANE_MEASURED)

    assert violations == []


def test_steane_needs_no_swaps_with_enough_connectivity():
    model = run(STEANE_MEASURED, 7, {6: 7})

    assert model.hardware_events == {"virtual_swap": 0}


def test_steane_noisy_circuit_without_noise_equals_input():
    model = run(STEANE_MEASURED, 7, {2: 7})
    expected = stim.Circuit()
    for op in STEANE_MEASURED:
        if op.name != "TICK":  # TICKs are not passed to the processors
            expected.append(op)

    assert without_noise(model.noisy_circuit) == expected
