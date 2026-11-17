"""Class grouping together relevant aspects of a quantum error-correction circuit."""

import stim 
from typing import Literal

CircuitFunction = Literal["StatePrep", "Encoding", "Syndrome"]
QuantumState = Literal["0", "+", "1", "-"]

from dataclasses import dataclass

@dataclass
class QECCircuit:
    circuit: stim.Circuit
    function: CircuitFunction 
    stabilizers: list[stim.PauliString]
    logical_operators: list[stim.PauliString]
    quantum_state: QuantumState | None = None