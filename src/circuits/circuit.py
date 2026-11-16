"""Class grouping together relevant aspects of a quantum error-correction circuit."""

import stim 
from typing import Literal

CircuitFunction = Literal["StatePrep", "Encoding", "Syndrome"]

from dataclasses import dataclass

@dataclass
class QECCircuit:
    circuit: stim.Circuit
    function: CircuitFunction 
    stabilizers: list[str]
    logical_operators: list[str]