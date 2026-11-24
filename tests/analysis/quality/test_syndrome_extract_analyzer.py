import pathlib

import pytest
import stim

from src.analysis.quality._decoder import BeliefPropOrdStat
from src.analysis.quality.syndrome_extract import SyndromeExtractionAnalyzer
from src.models.common import GateKind
from src.models.hardware import SuperconductingQubitsModel

CODE_DIR = pathlib.Path(__file__).parents[3] / "data/circuits/steane-code"
STABILIZERS_TXT = [stim.PauliString(s) for s in (CODE_DIR / "stabilizers.txt").read_text().split()]
LOGICALS = [stim.PauliString(s) for s in (CODE_DIR / "logicals.txt").read_text().split()]
LOGICAL_OPERATORS = [(LOGICALS[0], LOGICALS[1])]
EXTRACTORS = sorted((CODE_DIR / "syndrome_extraction").glob("*.stim"))
ALPHA = stim.Circuit.from_file(CODE_DIR / "syndrome_extraction/steane-code--alphasyndrome-schedule-bp-osd-on-the-4-8-8-lattice.stim")
N = 7  # data qubits 0..6, ancillas 7..12
DURATIONS: dict[GateKind, float] = {"SN": 1.0, "EN": 2.0, "MS": 5.0}
SHOTS = 500


def measured_stabilizers(circuit: stim.Circuit) -> list[stim.PauliString]:
    # same extraction as in sandbox.py: propagate each ancilla measurement back to the data qubits
    inverse = stim.Tableau.from_circuit(circuit, ignore_reset=True, ignore_measurement=True).inverse()
    measured = [inverse.x_output(t.value) if gate.name == "MX" else inverse.z_output(t.value)
                for gate in circuit if stim.gate_data(gate.name).produces_measurements for t in gate.targets_copy()]
    return [p.sign * p[:N] for p in measured]


def analyze(circuit: stim.Circuit, noisy_circuit: stim.Circuit | None = None, stabilizers=None) -> dict[str, object]:
    # the analyzer only reads the noisy circuit, so it is set directly to control exactly which errors happen
    stabilizers = measured_stabilizers(circuit) if stabilizers is None else stabilizers
    hw = SuperconductingQubitsModel(circuit.num_qubits, DURATIONS, {4: circuit.num_qubits}, {"H", "CX", "M", "R"})
    hw.noisy_circuit = circuit if noisy_circuit is None else noisy_circuit
    decoder = BeliefPropOrdStat(hw, stabilizers)
    return SyndromeExtractionAnalyzer(hw.noisy_circuit, stabilizers, LOGICAL_OPERATORS, SHOTS, decoder, list(range(len(stabilizers)))).analyze()


def test_missing_or_mismatching_syndrome_measurements_raise():
    hw = SuperconductingQubitsModel(13, DURATIONS, {4: 13}, {"H", "CX", "M", "R"})
    stabilizers = measured_stabilizers(ALPHA)
    decoder = BeliefPropOrdStat(hw, stabilizers)

    with pytest.raises(ValueError):
        SyndromeExtractionAnalyzer(hw.noisy_circuit, stabilizers, LOGICAL_OPERATORS, SHOTS, decoder, None)
    with pytest.raises(ValueError):
        SyndromeExtractionAnalyzer(hw.noisy_circuit, stabilizers, LOGICAL_OPERATORS, SHOTS, decoder, [0, 1, 2])


def test_extracted_stabilizers_generate_the_code():
    # every circuit measures a generating set of the same stabilizer group as stabilizers.txt
    for path in EXTRACTORS:
        stabilizers = measured_stabilizers(stim.Circuit.from_file(path))
        assert len(stabilizers) == len(STABILIZERS_TXT)
        assert all(s.commutes(t) for s in stabilizers for t in STABILIZERS_TXT + LOGICALS)


@pytest.mark.parametrize("path", EXTRACTORS, ids=lambda p: p.stem)
def test_noiseless_extraction_has_no_failures(path: pathlib.Path):
    assert analyze(stim.Circuit.from_file(path)).items() >= {"syndrome_error_rate": 0.0, "n_failures": 0, "logical_error_rate": 0.0}.items()


def test_mismatching_stabilizer_order_shows_up_as_syndrome_errors():
    # the AlphaSyndrome circuit measures X0 X1 X3 X6 on its third ancilla instead of X3 X4 X5 X6 from stabilizers.txt
    result = analyze(ALPHA, stabilizers=STABILIZERS_TXT)

    assert result["syndrome_error_rate"] > 0
    assert result["logical_error_rate"] == 0.0


@pytest.mark.parametrize("error", ["X 3", "Z 5", "Y 0"])
def test_data_error_before_extraction_is_a_syndrome_error_but_corrected(error: str):
    # the error happens after the first perfect round (it is part of the noisy circuit), so the circuit extracts its syndrome
    # instead of the one of the input state; the perfect rounds see it and the decoder corrects it
    result = analyze(ALPHA, stim.Circuit(error) + ALPHA)

    assert result["syndrome_error_rate"] == 1.0
    assert result["logical_error_rate"] == 0.0


@pytest.mark.parametrize("error", ["X 3", "Z 5", "Y 0"])
def test_data_error_after_extraction_is_no_syndrome_error_but_corrected(error: str):
    # the circuit extracts the trivial syndrome, but the final perfect round sees the error
    result = analyze(ALPHA, ALPHA + stim.Circuit(error))

    assert result["syndrome_error_rate"] == 0.0
    assert result["logical_error_rate"] == 0.0


def test_data_error_in_the_middle_of_extraction_is_a_syndrome_error():
    # Z 6 after "CX 8 2 9 6": ancilla 9 already checked qubit 6 and misses it, ancillas 7 and 8 check it later and see it
    first, rest = ALPHA[:3], ALPHA[3:]
    result = analyze(ALPHA, first + stim.Circuit("Z 6") + rest)

    assert result["syndrome_error_rate"] == 1.0
    assert result["logical_error_rate"] == 0.0


def test_measurement_error_is_a_syndrome_error_without_logical_damage():
    # flip the ancilla right before it is measured
    without_measurement = ALPHA[:-1]
    result = analyze(ALPHA, without_measurement + stim.Circuit("X 10") + ALPHA[-1:])

    assert result["syndrome_error_rate"] == 1.0
    assert result["logical_error_rate"] == 0.0


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_every_logical_error_is_detected(pauli: str):
    # the Bell pair with the reference checks X_L and Z_L at once, so X_L, Y_L and Z_L all fail
    result = analyze(ALPHA, ALPHA + stim.Circuit(f"{pauli} 0 1 2 3 4 5 6"))

    assert result["syndrome_error_rate"] == 0.0
    assert result["logical_error_rate"] == 1.0
