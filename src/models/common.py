from typing import Literal

GateKind = Literal["SN", "MS", "EN"]
NoiseTag = Literal["1q", "2q", "idle", "hw", "reset", "meas"]
HardwareEventLocation = tuple[int, int] # (qubit,  number of completed anchor operations), "anchor" meaning compiler keeps their order somewhat fixed, aka 2q-gates/resets/measurements