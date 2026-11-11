import pathlib

import stim

from src.analysis.analyzer import CircuitAnalyzer
from src.structure.qubit_timelines import QubitTimelines, TimelineGate

STEANE = stim.Circuit((pathlib.Path(__file__).parents[2] / "data/circuits/steane_example.stim").read_text())

SN = TimelineGate("SN")
MS = TimelineGate("MS")


def EN(partner: int) -> TimelineGate:
    return TimelineGate("EN", partner)


def build(circuit: str | stim.Circuit) -> QubitTimelines:
    timelines = QubitTimelines()
    CircuitAnalyzer(timelines).run(stim.Circuit(circuit) if isinstance(circuit, str) else circuit, "StatePrep")
    return timelines


def test_single_qubit_gates_and_resets_are_single_entries():
    assert build("H 0\nS 0\nR 0\nRX 0").timelines == {0: [SN, SN, SN, SN]}


def test_measurements_are_measurement_entries():
    assert build("M 0\nMX 0\nM(0.01) 0").timelines == {0: [MS, MS, MS]}


def test_measure_reset_counts_as_measurement():
    assert build("MR 0").timelines == {0: [MS]}


def test_two_qubit_gate_adds_entangling_entry_with_partner_to_both_qubits():
    assert build("CX 0 1").timelines == {0: [EN(1)], 1: [EN(0)]}


def test_entries_are_in_circuit_order():
    assert build("R 0 1\nH 0\nCX 0 1\nM 0 1").timelines == {
        0: [SN, SN, EN(1), MS],
        1: [SN, EN(0), MS],
    }


def test_only_qubits_with_gates_get_timelines():
    assert set(build("H 2\nCX 2 5").timelines) == {2, 5}


def test_steane():
    assert build(STEANE).timelines == {
        0: [SN, SN, EN(2), EN(5)],
        1: [SN, EN(3)],
        2: [SN, EN(0), EN(3)],
        3: [SN, SN, EN(1), EN(4), EN(2), EN(5)],
        4: [SN, SN, EN(3), EN(6)],
        5: [SN, SN, EN(0), EN(6), EN(3)],
        6: [SN, EN(4), EN(5)],
    }


def test_steane_entangling_entries_are_mirrored_between_partners():
    timelines = build(STEANE).timelines

    for qubit, timeline in timelines.items():
        for entry in timeline:
            if entry.kind == "EN":
                assert entry.partner is not None
                partner_entries = [e for e in timelines[entry.partner] if e == EN(qubit)]
                own_entries = [e for e in timeline if e == entry]
                assert len(partner_entries) == len(own_entries)
            else:
                assert entry.partner is None
