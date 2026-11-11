"""Analyze a given circuit on a specific hardware platform and configuration."""

from __future__ import annotations

import stim

from typing import TYPE_CHECKING
from src.analysis import CircuitFunction, MetricExtractor
from src.structure import GateDistribution, InteractionGraph, DependencyDAG, QubitTimelines
from src.analysis.processor import CircuitProcessor
from src.analysis.efficiency.dag import DependencyDAGAnalyzer
from src.analysis.efficiency.wip import InteractionGraphAnalyzer
from src.analysis.efficiency.qtl import QubitTimelinesAnalyzer
from src.analysis.efficiency.gd import GateDistributionAnalyzer
from src.analysis.quality.encoder import EncoderCircuitAnalyzer
from src.analysis.quality.state_prep import StatePreparationCircuitAnalyzer
from src.analysis.quality.syndrome_extract import SyndromeExtractionAnalyzer


if TYPE_CHECKING:
    from src.models.hardware import HardwareModel


class CircuitAnalyzer:
        def __init__(self, hardware: HardwareModel):
            self.hardware = hardware
            self.metrics = {}
            

        def analyze(self, circuit: stim.Circuit, function: CircuitFunction):
            dag = DependencyDAG()
            gd = GateDistribution()
            wig = InteractionGraph()
            qtl = QubitTimelines()
            
            # 1.) Process the circuit (execute it abstractly or build a relevant representation)
            processor = CircuitProcessor(dag, gd, wig, qtl, self.hardware)
            processor.run(circuit=circuit)

            # 2.) Extract the metrics
            efficiency_analyzer : list[MetricExtractor] = [
                DependencyDAGAnalyzer(dag, self.hardware),
                GateDistributionAnalyzer(gd, self.hardware),
                InteractionGraphAnalyzer(wig, self.hardware),
                QubitTimelinesAnalyzer(qtl, self.hardware) 
            ]

            match function:
                case "StatePrep": 
                    quality_analyzer = [ StatePreparationCircuitAnalyzer(self.hardware) ]
                case "Encoding": 
                    quality_analyzer = [ EncoderCircuitAnalyzer(self.hardware) ]
                case "Syndrome": 
                    quality_analyzer = [ SyndromeExtractionAnalyzer(self.hardware) ]
                  
            for analyzer in efficiency_analyzer + quality_analyzer:
                self.metrics.update(analyzer.analyze())
