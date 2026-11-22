"""Analyze a given circuit on a specific hardware platform and configuration."""

from __future__ import annotations

import stim

from typing import TYPE_CHECKING
from src.circuits.circuit import CircuitFunction
from src.analysis import MetricExtractor
from src.structure import GateDistribution, InteractionGraph, DependencyDAG, QubitTimelines
from src.analysis.processor import CircuitProcessor
from src.analysis.efficiency.dag import DependencyDAGAnalyzer
from src.analysis.efficiency.wig import InteractionGraphAnalyzer
from src.analysis.efficiency.qtl import QubitTimelinesAnalyzer
from src.analysis.efficiency.gd import GateDistributionAnalyzer
from src.analysis.quality.encoder import EncoderCircuitAnalyzer
from src.analysis.quality.state_prep import StatePreparationCircuitAnalyzer
from src.analysis.quality.syndrome_extract import SyndromeExtractionAnalyzer
from dataclasses import dataclass
from src.circuits.circuit import QECCircuit
from src.analysis.quality._decoder import BeliefPropOrdStat

if TYPE_CHECKING:
    from src.models.hardware import HardwareModel

MAX_SHOTS = 10_000

@dataclass
class CircuitAnalysisResult:
    """Interface between Circuit Analysis and Circuit Evaluation."""
    compilation_cost_metrics : dict
    runtime_cost_metrics : dict
    noise_cost_metrics : dict


class CircuitAnalyzer:
        def __init__(self, hardware: HardwareModel):
            self.hardware = hardware
            

        def analyze(self, circuit: QECCircuit) -> CircuitAnalysisResult:
            """Analyze static and dynamic circuit properties dependent on the hardware."""
            dag = DependencyDAG()
            gd = GateDistribution()
            wig = InteractionGraph()
            qtl = QubitTimelines()
            
            # 1.) Process the circuit (execute it abstractly or build a relevant representation)
            processor = CircuitProcessor(dag, gd, wig, qtl, self.hardware)
            processor.run(circuit=circuit.circuit)

            # 2.) Extract the metrics
            efficiency_analyzer : list[MetricExtractor] = [
                DependencyDAGAnalyzer(dag, self.hardware),
                GateDistributionAnalyzer(gd, self.hardware),
                InteractionGraphAnalyzer(wig, self.hardware),
                QubitTimelinesAnalyzer(qtl, self.hardware) 
            ]

            compilation_cost = {}
            runtime_cost = {}
            for analyzer in efficiency_analyzer:
                result = analyzer.analyze()
                compilation_cost.update(result["compilation"])
                runtime_cost.update(result["runtime"])


            decoder = BeliefPropOrdStat(self.hardware, circuit.stabilizers)

            match circuit.function:
                case "StatePrep": 
                    quality_analyzer = [ StatePreparationCircuitAnalyzer(self.hardware, circuit.stabilizers, circuit.logical_operators, max_shots=MAX_SHOTS, decoder=decoder, accept_if=circuit.verification, state=circuit.quantum_state) ]
                case "Encoding": 
                    quality_analyzer = [ EncoderCircuitAnalyzer(self.hardware, circuit.stabilizers, circuit.logical_operators, max_shots=MAX_SHOTS, decoder=decoder, logical_qubits=circuit.logical_qubits) ]
                case "Syndrome": 
                    quality_analyzer = [ SyndromeExtractionAnalyzer(self.hardware, circuit.stabilizers, circuit.logical_operators, max_shots=MAX_SHOTS, decoder=decoder, syndrome_measurements=circuit.syndrome_measurements) ]
                case _:
                    quality_analyzer = []

            noise_cost = {}
            for analyzer in quality_analyzer:
                noise_cost.update(analyzer.analyze())

            return CircuitAnalysisResult(compilation_cost_metrics=compilation_cost, runtime_cost_metrics=runtime_cost, noise_cost_metrics=noise_cost)
