import pytest
import stim

from src.analysis.efficiency.gd import GateDistributionAnalyzer
from src.analysis.processor import CircuitProcessor
from src.models.hardware import TrappedIonsModel
from src.structure.gate_distribution import GateDistribution

NATIVE_GATES: set[str] = {"CX", "H", "M", "R"}

# R, R, H, CX, SWAP, MX, MR -> 7 gates
CIRCUIT = "R 0 1\nH 0\nCX 0 1\nSWAP 1 2\nMX 0\nMR 1"


def analyze(circuit: str, native_gates: set[str] = NATIVE_GATES) -> dict[str, object]:
    gd = GateDistribution()
    CircuitProcessor(gd).run(stim.Circuit(circuit))
    result = GateDistributionAnalyzer(gd, TrappedIonsModel(2, 1, native_gates)).analyze()
    return {**result["runtime"], **result["compilation"]}


def test_metrics_are_split_into_runtime_and_compilation():
    gd = GateDistribution()
    CircuitProcessor(gd).run(stim.Circuit(CIRCUIT))
    result = GateDistributionAnalyzer(gd, TrappedIonsModel(2, 1, NATIVE_GATES)).analyze()

    assert set(result) == {"runtime", "compilation"}
    assert set(result["compilation"]) == {"share_native_gates", "share_native_2q_gates"}


def test_composition():
    result = analyze(CIRCUIT)

    assert result["nr_gates"] == 7
    assert result["share_2q_gates"] == pytest.approx(2 / 7)  # CX, SWAP
    assert result["share_measurements"] == pytest.approx(2 / 7)  # MX, MR
    assert result["share_resets"] == pytest.approx(3 / 7)  # R, R, MR


def test_native_gate_shares():
    result = analyze(CIRCUIT)

    assert result["share_native_gates"] == pytest.approx(4 / 7)  # R, R, H, CX
    assert result["share_native_2q_gates"] == pytest.approx(1 / 2)  # CX but not SWAP


def test_estimated_post_transpilation_counts():
    result = analyze(CIRCUIT)

    # 4 native + SWAP (9) + MX (3) + MR (3)
    assert result["estimated_nr_gates_post_transpilation"] == 19
    # CX (1) + SWAP (3)
    assert result["estimated_nr_2q_gates_post_transpilation"] == 4


def test_non_native_1q_and_2q_gates_cost():
    # S -> 2 native 1q gates, CZ -> 1 native 2q gate + 2 1q gates
    result = analyze("S 0\nCZ 0 1")

    assert result["estimated_nr_gates_post_transpilation"] == 2 + 3
    assert result["estimated_nr_2q_gates_post_transpilation"] == 1


def test_fully_native_circuit_needs_no_decomposition():
    result = analyze(CIRCUIT, NATIVE_GATES | {"SWAP", "MX", "MR"})

    assert result["share_native_gates"] == 1.0
    assert result["share_native_2q_gates"] == 1.0
    assert result["estimated_nr_gates_post_transpilation"] == result["nr_gates"]
    assert result["estimated_nr_2q_gates_post_transpilation"] == 2
