"""Static structural analysis of quantum circuits targeting specific hardware (efficiency metrics of the four circuit representations)."""

import time

import stim

from src.models.hardware import HardwareModel, PlatformTag, TrappedIonsModel, NeutralAtomsModel, SuperconductingQubitsModel
from collections.abc import Callable

from src.analysis.processor import CircuitProcessor
from src.structure import GateDistribution, InteractionGraph, DependencyDAG, QubitTimelines
from src.analysis.efficiency.dag import DependencyDAGAnalyzer
from src.analysis.efficiency.wig import InteractionGraphAnalyzer
from src.analysis.efficiency.qtl import QubitTimelinesAnalyzer
from src.analysis.efficiency.gd import GateDistributionAnalyzer


def run_ti_structural_analysis(hardware: TrappedIonsModel, circuit: stim.Circuit) -> tuple[dict, dict]:
    """Run the static efficiency metric extraction for the trapped ion hardware parameters on the given circuit."""
    start = time.perf_counter()
    dag, gd, wig, qtl = DependencyDAG(), GateDistribution(), InteractionGraph(), QubitTimelines()
    CircuitProcessor(dag, gd, wig, qtl).run(circuit)
    metrics = {}
    for analyzer in [DependencyDAGAnalyzer(dag, hardware), GateDistributionAnalyzer(gd, hardware), InteractionGraphAnalyzer(wig, hardware), QubitTimelinesAnalyzer(qtl, hardware)]:
        metrics.update(analyzer.analyze())
    wall_clock_s = time.perf_counter() - start

    return metrics, {"wall_clock_s": wall_clock_s}


def run_na_structural_analysis(hardware: NeutralAtomsModel, circuit: stim.Circuit) -> tuple[dict, dict]:
    """Run the static efficiency metric extraction for the neutral atom hardware parameters on the given circuit."""
    start = time.perf_counter()
    dag, gd, wig, qtl = DependencyDAG(), GateDistribution(), InteractionGraph(), QubitTimelines()
    CircuitProcessor(dag, gd, wig, qtl).run(circuit)
    metrics = {}
    for analyzer in [DependencyDAGAnalyzer(dag, hardware), GateDistributionAnalyzer(gd, hardware), InteractionGraphAnalyzer(wig, hardware), QubitTimelinesAnalyzer(qtl, hardware)]:
        metrics.update(analyzer.analyze())
    wall_clock_s = time.perf_counter() - start

    return metrics, {"wall_clock_s": wall_clock_s}


def run_sc_structural_analysis(hardware: SuperconductingQubitsModel, circuit: stim.Circuit) -> tuple[dict, dict]:
    """Run the static efficiency metric extraction for the superconducting qubit hardware parameters on the given circuit."""
    start = time.perf_counter()
    dag, gd, wig, qtl = DependencyDAG(), GateDistribution(), InteractionGraph(), QubitTimelines()
    CircuitProcessor(dag, gd, wig, qtl).run(circuit)
    metrics = {}
    for analyzer in [DependencyDAGAnalyzer(dag, hardware), GateDistributionAnalyzer(gd, hardware), InteractionGraphAnalyzer(wig, hardware), QubitTimelinesAnalyzer(qtl, hardware)]:
        metrics.update(analyzer.analyze())
    wall_clock_s = time.perf_counter() - start

    return metrics, {"wall_clock_s": wall_clock_s}

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

_RUN_STRUCTURAL_ANALYSIS : dict[PlatformTag, Callable[[HardwareModel, stim.Circuit], tuple[dict, dict]]] = {
    "TI": run_ti_structural_analysis,
    "NA": run_na_structural_analysis,
    "SC": run_sc_structural_analysis
}
