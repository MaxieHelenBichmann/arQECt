import pathlib
import random

import pytest
import stim

from src.analysis.analyzer import CircuitAnalyzer
from src.models.hardware.neutral_atoms import NeutralAtomsModel
from src.models.noise.noise import NoiseModel

STEANE = stim.Circuit((pathlib.Path(__file__).parents[2] / "data/circuits/steane_example.stim").read_text())
STEANE_MEASURED = STEANE + stim.Circuit("M 0 1 2 3 4 5 6")

NO_EVENTS = {"storage_entanglement": 0, "storage_readout": 0, "entanglement_readout": 0, "inter_entanglement": 0}


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


def run(circuit: str | stim.Circuit, nr_pairs: int, readout_capacity: int, **noise_kwargs) -> NeutralAtomsModel:
    model = NeutralAtomsModel(nr_pairs, readout_capacity, make_noise_model(**noise_kwargs))
    CircuitAnalyzer(model).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit)
    return model


def events(**counts: int) -> dict[str, int]:
    return {**NO_EVENTS, **counts}


def pair_of(model: NeutralAtomsModel, qubit: int) -> int | None:
    return next((i for i, pair in enumerate(model._entanglement_zone) if qubit in pair), None)


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


# ---------------------------------------------------------------- construction


@pytest.mark.parametrize("nr_pairs, readout_capacity", [(0, 1), (1, 0), (-1, 1)])
def test_invalid_capacities_are_rejected(nr_pairs: int, readout_capacity: int):
    with pytest.raises(ValueError):
        NeutralAtomsModel(nr_pairs, readout_capacity, make_noise_model())


# ---------------------------------------------------------------- single-qubit gates and readout zone


def test_single_qubit_gates_and_resets_in_storage_cause_no_hardware_events():
    model = run("R 0\nH 0\nS 0", nr_pairs=1, readout_capacity=1)

    assert model.hardware_events == NO_EVENTS
    assert model.noisy_circuit == stim.Circuit("R 0\nH 0\nS 0")


def test_measurement_moves_qubit_from_storage_to_readout():
    model = run("M 0", nr_pairs=1, readout_capacity=1)

    assert model._readout_zone == {0: 0}
    assert model.hardware_events == events(storage_readout=1)
    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0\nM 0")


def test_repeated_measurement_is_a_readout_cache_hit():
    model = run("M 0\nMX 0", nr_pairs=1, readout_capacity=1)

    assert model._readout_zone == {0: 1}
    assert model.hardware_events == events(storage_readout=1)


def test_single_qubit_gate_after_measurement_moves_qubit_back_to_storage():
    model = run("M 0\nH 0", nr_pairs=1, readout_capacity=1)

    assert model._readout_zone == {}
    assert model.hardware_events == events(storage_readout=2)


def test_full_readout_zone_evicts_least_recently_used_qubit():
    model = run("M 0\nM 1\nM 0\nM 2", nr_pairs=1, readout_capacity=2)

    assert set(model._readout_zone) == {0, 2}
    assert model.hardware_events == events(storage_readout=4)  # 3 entries, 1 eviction of qubit 1


# ---------------------------------------------------------------- two-qubit gates and entanglement zone


def test_two_qubit_gate_cache_miss_moves_both_qubits_into_one_pair():
    model = run("CX 0 1", nr_pairs=2, readout_capacity=1)

    assert pair_of(model, 0) is not None
    assert pair_of(model, 0) == pair_of(model, 1)
    assert model.hardware_events == events(storage_entanglement=2)
    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0 1\nCX 0 1")


def test_two_qubit_gate_cache_hit_in_same_pair_has_no_hardware_events():
    model = run("CX 0 1\nCZ 0 1", nr_pairs=1, readout_capacity=1)

    assert model._entanglement_zone == ({0: 2, 1: 3},)
    assert model.hardware_events == events(storage_entanglement=2)


def test_two_qubit_gate_cache_miss_in_full_pair_evicts_both_qubits_to_storage():
    model = run("CX 0 1\nCX 2 3", nr_pairs=1, readout_capacity=1)

    assert set(model._entanglement_zone[0]) == {2, 3}
    assert model.hardware_events == events(storage_entanglement=6)  # 4 entries, 2 evictions


def test_two_qubit_gate_partial_cache_hit_evicts_the_other_qubit_not_the_partner():
    # qubit 0 is older than qubit 1, but is touched before qubit 2 enters
    model = run("CX 0 1\nCX 0 2", nr_pairs=1, readout_capacity=1)

    assert set(model._entanglement_zone[0]) == {0, 2}
    assert model.hardware_events == events(storage_entanglement=4)  # 3 entries, 1 eviction of qubit 1


def test_two_qubit_gate_in_different_pairs_moves_target_to_source_pair():
    model = NeutralAtomsModel(2, 1, make_noise_model())
    model._entanglement_zone[0][0] = 0
    model._entanglement_zone[1][1] = 1
    model._lru_time_step = 2

    model.process(stim.CircuitInstruction("CX", [0, 1]))

    assert model._entanglement_zone == ({0: 2, 1: 3}, {})
    assert model.hardware_events == events(inter_entanglement=1)


def test_two_qubit_gate_in_different_pairs_evicts_from_full_source_pair():
    model = NeutralAtomsModel(2, 1, make_noise_model())
    model._entanglement_zone[0].update({0: 0, 5: 1})
    model._entanglement_zone[1][1] = 2
    model._lru_time_step = 3

    model.process(stim.CircuitInstruction("CX", [0, 1]))

    assert model._entanglement_zone == ({0: 3, 1: 4}, {})
    assert model.hardware_events == events(inter_entanglement=1, storage_entanglement=1)  # eviction of qubit 5


def test_qubit_moves_from_readout_to_entanglement_zone():
    model = run("M 0\nCX 0 1", nr_pairs=1, readout_capacity=1)

    assert model._readout_zone == {}
    assert model.hardware_events == events(storage_readout=1, entanglement_readout=1, storage_entanglement=1)


def test_qubit_moves_from_entanglement_to_readout_zone():
    model = run("CX 0 1\nM 0", nr_pairs=1, readout_capacity=1)

    assert set(model._entanglement_zone[0]) == {1}
    assert set(model._readout_zone) == {0}
    assert model.hardware_events == events(storage_entanglement=2, entanglement_readout=1)


def test_single_qubit_gate_moves_qubit_from_entanglement_back_to_storage():
    model = run("CX 0 1\nH 0", nr_pairs=1, readout_capacity=1)

    assert set(model._entanglement_zone[0]) == {1}
    assert model.hardware_events == events(storage_entanglement=3)


# ---------------------------------------------------------------- pair choice


def test_choose_pair_prefers_empty_pairs():
    model = NeutralAtomsModel(2, 1, make_noise_model())
    model._entanglement_zone[0].update({0: 0, 1: 1})

    assert model._choose_pair() == 1


def test_choose_pair_prefers_pairs_with_fewer_qubits_over_older_qubits():
    model = NeutralAtomsModel(2, 1, make_noise_model())
    model._entanglement_zone[0].update({0: 0, 1: 1})
    model._entanglement_zone[1][2] = 5

    assert model._choose_pair() == 1


def test_choose_pair_minimizes_summed_lru_times_of_evicted_qubits():
    model = NeutralAtomsModel(2, 1, make_noise_model())
    model._entanglement_zone[0].update({0: 5, 1: 6})
    model._entanglement_zone[1].update({2: 1, 3: 20})

    assert model._choose_pair() == 0  # costs: 11 vs. 21


# ---------------------------------------------------------------- noise


def test_single_qubit_gate_and_reset_noise_are_applied_after_the_gate():
    model = run(
        "R 0\nH 0",
        nr_pairs=1,
        readout_capacity=1,
        single_qubit_gate=("Depolarize", (0.001,)),
        reset=("BitFlip", (0.02,)),
    )

    assert model.noisy_circuit == stim.Circuit("R 0\nX_ERROR(0.02) 0\nH 0\nDEPOLARIZE1(0.001) 0")


def test_measurement_noise_is_the_flip_probability_of_the_measurement_in_its_basis():
    model = run("M 0\nMX 0", nr_pairs=1, readout_capacity=1, measurement=("Measurement", (0.03,)))

    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0\nM(0.03) 0\nMX(0.03) 0")


def test_two_qubit_gate_noise_is_applied_to_both_qubits_after_the_gate():
    model = run("CX 0 1", nr_pairs=1, readout_capacity=1, two_qubit_gate=("Depolarize", (0.005,)))

    assert model.noisy_circuit == stim.Circuit("DEPOLARIZE1(0.01) 0 1\nCX 0 1\nDEPOLARIZE2(0.005) 0 1")


def test_hardware_event_noise_with_multiple_arguments():
    model = run("M 0", nr_pairs=1, readout_capacity=1, hardware_event=("Pauli", (0.01, 0.02, 0.03)))

    assert model.noisy_circuit == stim.Circuit("PAULI_CHANNEL_1(0.01, 0.02, 0.03) 0\nM 0")


def test_no_noise_adds_no_identity_instructions():
    # covers moves between all zones and evictions from pairs and the readout zone
    circuit = "CX 0 1\nCX 2 3\nM 0\nM 1\nH 0\nCX 0 2"
    model = run(circuit, nr_pairs=1, readout_capacity=1, hardware_event=("None", ()))

    assert model.hardware_events["storage_entanglement"] > 0
    assert model.noisy_circuit == stim.Circuit(circuit)


# ---------------------------------------------------------------- small circuits


@pytest.mark.parametrize("nr_pairs, readout_capacity", [(1, 1), (1, 3), (2, 1), (3, 3), (7, 7)])
@pytest.mark.parametrize("seed", range(5))
def test_steane_zone_invariants_after_every_gate(nr_pairs: int, readout_capacity: int, seed: int):
    random.seed(seed)
    model = NeutralAtomsModel(nr_pairs, readout_capacity, make_noise_model())
    violations = []

    class InvariantCheck:
        def process(self, gate: stim.CircuitInstruction) -> None:
            qubits_in_zones = [q for pair in model._entanglement_zone for q in pair] + list(model._readout_zone)
            if len(qubits_in_zones) != len(set(qubits_in_zones)):
                violations.append(("qubit in more than one zone", gate))
            if any(len(pair) > 2 for pair in model._entanglement_zone):
                violations.append(("pair with more than two qubits", gate))
            if len(model._readout_zone) > readout_capacity:
                violations.append(("readout zone over capacity", gate))
            data = stim.gate_data(gate.name)
            qubits = [t.value for t in gate.targets_copy()]
            if data.is_two_qubit_gate and (
                pair_of(model, qubits[0]) is None or pair_of(model, qubits[0]) != pair_of(model, qubits[1])
            ):
                violations.append(("two-qubit gate not within one pair", gate))
            if data.produces_measurements and qubits[0] not in model._readout_zone:
                violations.append(("measurement outside readout zone", gate))

        def finalize(self) -> None:
            pass

    CircuitAnalyzer(model, InvariantCheck()).run(STEANE_MEASURED)

    assert violations == []


def test_steane_measured_ends_with_last_measured_qubits_in_readout_zone():
    model = run(STEANE_MEASURED, nr_pairs=2, readout_capacity=3)

    assert set(model._readout_zone) == {4, 5, 6}
    assert all(len(pair) == 0 for pair in model._entanglement_zone)


def test_steane_noisy_circuit_without_noise_equals_input():
    model = run(STEANE_MEASURED, nr_pairs=2, readout_capacity=2)
    expected = stim.Circuit()
    for op in STEANE_MEASURED:
        if op.name != "TICK":  # TICKs are not passed to the processors
            expected.append(op)

    assert without_noise(model.noisy_circuit) == expected
