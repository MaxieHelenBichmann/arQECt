import pathlib
import random

import pytest
import stim

from src.analysis.analyzer import CircuitAnalyzer
from src.models.hardware.trapped_ions import TrappedIonsModel
from src.models.noise.noise import NoiseModel

STEANE = stim.Circuit((pathlib.Path(__file__).parents[2] / "data/circuits/steane_example.stim").read_text())


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
        idle=("ThermalRelaxation", (100e-6, 80e-6)),
        hardware_event=hardware_event,
    )


def run(circuit: str | stim.Circuit, capacity: int, nr_zones: int, **noise_kwargs) -> TrappedIonsModel:
    model = TrappedIonsModel(capacity, nr_zones, make_noise_model(**noise_kwargs))
    CircuitAnalyzer(model).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit)
    return model


def zone_of(model: TrappedIonsModel, qubit: int) -> int | None:
    return next((i for i, zone in enumerate(model._zones) if qubit in zone), None)


def without_noise(circuit: stim.Circuit) -> stim.Circuit:
    result = stim.Circuit()
    for op in circuit:
        data = stim.gate_data(op.name)
        if not (data.is_noisy_gate and not data.produces_measurements):
            result.append(op)
    return result


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)


# ---------------------------------------------------------------- single-qubit gates


def test_single_qubit_gate_cache_miss_enters_zone():
    model = run("H 0", capacity=2, nr_zones=1)

    assert model._zones == ({0: 0},)
    assert model.hardware_events == {"interaction_zone_entry": 1, "interaction_zone_eviction": 0}
    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0\nH 0")


def test_single_qubit_gate_cache_hit_only_updates_lru_time():
    model = run("H 0\nS 0", capacity=2, nr_zones=1)

    assert model._zones == ({0: 1},)
    assert model._lru_time_step == 2
    assert model.hardware_events == {"interaction_zone_entry": 1, "interaction_zone_eviction": 0}


def test_measurements_and_resets_are_handled_as_single_qubit_gates():
    model = run("R 0\nM(0.02) 0\nMR 0\nMX 0", capacity=2, nr_zones=1)

    assert model.hardware_events == {"interaction_zone_entry": 1, "interaction_zone_eviction": 0}
    assert without_noise(model.noisy_circuit) == stim.Circuit("R 0\nM(0.02) 0\nMR 0\nMX 0")


def test_full_zone_evicts_least_recently_used_qubit():
    model = run("H 0\nH 1\nH 0\nH 2", capacity=2, nr_zones=1)

    assert set(model._zones[0]) == {0, 2}
    assert model.hardware_events == {"interaction_zone_entry": 3, "interaction_zone_eviction": 1}
    # stim merges consecutive instructions of the same kind: "DEPOLARIZE1(0.01) 1 2" is eviction of 1, entry of 2
    assert model.noisy_circuit == stim.Circuit(
        "DEPOLARIZE1(0.01) 0\nH 0\nDEPOLARIZE1(0.01) 1\nH 1 0\nDEPOLARIZE1(0.01) 1 2\nH 2"
    )


def test_gate_targets_are_split_into_individual_gates():
    model = run("H 0 1 2", capacity=3, nr_zones=1)

    assert set(model._zones[0]) == {0, 1, 2}
    assert without_noise(model.noisy_circuit) == stim.Circuit("H 0\nH 1\nH 2")


# ---------------------------------------------------------------- two-qubit gates


def test_two_qubit_gate_cache_miss_puts_both_qubits_in_one_zone():
    model = run("CX 0 1", capacity=2, nr_zones=2)

    assert zone_of(model, 0) is not None
    assert zone_of(model, 0) == zone_of(model, 1)
    assert model.hardware_events == {"interaction_zone_entry": 2, "interaction_zone_eviction": 0}


def test_two_qubit_gate_cache_hit_in_same_zone_has_no_hardware_events():
    model = run("CX 0 1\nCZ 0 1", capacity=2, nr_zones=1)

    assert model._zones == ({0: 2, 1: 3},)
    assert model.hardware_events == {"interaction_zone_entry": 2, "interaction_zone_eviction": 0}


def test_two_qubit_gate_partial_cache_hit_moves_uncached_qubit_to_partner():
    model = run("H 0\nCX 0 1", capacity=2, nr_zones=3)

    assert zone_of(model, 1) == zone_of(model, 0)
    assert model.hardware_events == {"interaction_zone_entry": 2, "interaction_zone_eviction": 0}


def test_two_qubit_gate_in_different_zones_moves_target_to_source():
    model = TrappedIonsModel(2, 2, make_noise_model())
    model._zones[0][0] = 0
    model._zones[1][1] = 1
    model._lru_time_step = 2

    model.process(stim.CircuitInstruction("CX", [0, 1]))

    assert model._zones == ({0: 2, 1: 3}, {})  # source is touched before the target enters
    assert model.hardware_events == {"interaction_zone_entry": 1, "interaction_zone_eviction": 1}


def test_two_qubit_gate_is_appended_to_noisy_circuit():
    model = run("CX 0 1", capacity=2, nr_zones=1)

    assert without_noise(model.noisy_circuit) == stim.Circuit("CX 0 1")


def test_two_qubit_gate_partial_cache_hit_does_not_evict_partner():
    # qubit 0 is the least recently used qubit in the full zone when qubit 2 has to enter
    model = run("H 0\nH 1\nCX 0 2", capacity=2, nr_zones=1)

    assert set(model._zones[0]) == {0, 2}


# ---------------------------------------------------------------- zone choice


def test_choose_zone_prefers_zones_with_free_capacity():
    model = TrappedIonsModel(2, 2, make_noise_model())
    model._zones[0].update({0: 0, 1: 1})

    assert model._choose_zone(1) == 1
    assert model._choose_zone(2) == 1


def test_choose_zone_minimizes_summed_lru_times_of_evicted_qubits():
    model = TrappedIonsModel(2, 2, make_noise_model())
    model._zones[0].update({0: 5, 1: 6})
    model._zones[1].update({2: 1, 3: 20})

    assert model._choose_zone(1) == 1  # costs: 5 vs. 1
    assert model._choose_zone(2) == 0  # costs: 11 vs. 21


def test_choose_zone_for_two_qubits_prefers_zones_with_one_spot_left():
    model = TrappedIonsModel(3, 3, make_noise_model())
    model._zones[0].update({0: 10, 1: 11})  # one spot left: one eviction
    model._zones[1].update({2: 0, 3: 1, 4: 2})  # full: two evictions, but oldest qubits
    model._zones[2].update({5: 12, 6: 13})  # one spot left: one eviction, but newer qubits

    assert model._choose_zone(2) == 0


# ---------------------------------------------------------------- noise


def test_hardware_event_noise_with_multiple_arguments():
    model = run("H 0", capacity=2, nr_zones=1, hardware_event=("Pauli", (0.01, 0.02, 0.03)))

    assert model.noisy_circuit == stim.Circuit("PAULI_CHANNEL_1(0.01, 0.02, 0.03) 0\nH 0")


def test_single_qubit_gate_noise_is_applied_after_the_gate():
    model = run("H 0", capacity=2, nr_zones=1, single_qubit_gate=("Depolarize", (0.001,)))

    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0\nH 0\nDEPOLARIZE1(0.001) 0")


def test_reset_noise_is_applied_after_the_reset():
    model = run("R 0", capacity=2, nr_zones=1, reset=("BitFlip", (0.02,)))

    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0\nR 0\nX_ERROR(0.02) 0")


def test_measurement_noise_is_the_flip_probability_of_the_measurement_in_its_basis():
    model = run("M 0\nMX 0\nMR 0", capacity=2, nr_zones=1, measurement=("Measurement", (0.03,)))

    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0\nM(0.03) 0\nMX(0.03) 0\nMR(0.03) 0")


def test_measurement_noise_flips_the_measurement_result():
    model = run("R 0\nM 0", capacity=2, nr_zones=1, hardware_event=("None", ()), measurement=("Measurement", (1.0,)))

    assert model.noisy_circuit.compile_sampler().sample(1)[0].tolist() == [True]


def test_measurement_without_measurement_noise_keeps_its_own_arguments():
    model = run("M(0.02) 0", capacity=2, nr_zones=1)

    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0\nM(0.02) 0")


def test_two_qubit_gate_noise_is_applied_to_both_qubits_after_the_gate():
    model = run("CX 0 1", capacity=2, nr_zones=1, two_qubit_gate=("Depolarize", (0.005,)))

    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0 1\nCX 0 1\nDEPOLARIZE2(0.005) 0 1")


def test_no_gate_noise_adds_no_identity_instructions():
    model = run("R 0\nH 0\nCX 0 1\nM 0", capacity=2, nr_zones=1)

    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0\nR 0\nH 0\nDEPOLARIZE1(0.01) 1\nCX 0 1\nM 0")


def test_no_hardware_event_noise_adds_no_identity_instructions():
    # capacity 2 with 3 qubits forces entries and evictions
    model = run("H 0\nH 1\nH 2\nCX 0 1", capacity=2, nr_zones=1, hardware_event=("None", ()))

    assert model.hardware_events["interaction_zone_eviction"] > 0
    assert model.noisy_circuit == stim.Circuit("H 0 1 2\nCX 0 1")


# ---------------------------------------------------------------- small circuits


def test_steane_without_capacity_pressure_enters_every_qubit_once():
    model = run(STEANE, capacity=7, nr_zones=1)

    assert set(model._zones[0]) == set(range(7))
    assert model.hardware_events == {"interaction_zone_entry": 7, "interaction_zone_eviction": 0}


@pytest.mark.parametrize("capacity, nr_zones", [(2, 1), (2, 3), (3, 2), (4, 2), (7, 3)])
@pytest.mark.parametrize("seed", range(5))
def test_steane_zone_invariants(capacity: int, nr_zones: int, seed: int):
    random.seed(seed)
    model = run(STEANE, capacity, nr_zones)

    qubits_in_zones = [qubit for zone in model._zones for qubit in zone]
    assert len(qubits_in_zones) == len(set(qubits_in_zones))  # every qubit in at most one zone
    assert all(len(zone) <= capacity for zone in model._zones)
    events = model.hardware_events
    assert events["interaction_zone_entry"] - events["interaction_zone_eviction"] == len(qubits_in_zones)
    assert events["interaction_zone_entry"] >= 7  # every qubit enters at least once


@pytest.mark.parametrize("capacity, nr_zones", [(2, 1), (3, 3), (4, 2), (7, 1), (7, 3)])
def test_steane_two_qubit_gates_are_executed_within_one_zone(capacity: int, nr_zones: int):
    model = TrappedIonsModel(capacity, nr_zones, make_noise_model())
    split_gates = []

    class CoLocationCheck:
        def process(self, gate: stim.CircuitInstruction) -> None:
            if stim.gate_data(gate.name).is_two_qubit_gate:
                src, tgt = (target.value for target in gate.targets_copy())
                if zone_of(model, src) is None or zone_of(model, src) != zone_of(model, tgt):
                    split_gates.append(gate)

        def finalize(self) -> None:
            pass

    CircuitAnalyzer(model, CoLocationCheck()).run(STEANE)

    assert split_gates == []


def test_steane_noisy_circuit_without_noise_equals_input():
    model = run(STEANE, capacity=7, nr_zones=1)
    expected = stim.Circuit()
    for op in STEANE:
        if op.name != "TICK":  # TICKs are not passed to the processors
            expected.append(op)

    assert without_noise(model.noisy_circuit) == expected
