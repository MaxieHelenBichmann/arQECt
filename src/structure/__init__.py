"""Structural representations of quantum circuits."""

from src.structure.dependency_dag import DependencyDAG
from src.structure.interaction_graph import InteractionGraph
from src.structure.qubit_timelines import QubitTimelines

CircuitRepresentations = DependencyDAG | InteractionGraph | QubitTimelines
