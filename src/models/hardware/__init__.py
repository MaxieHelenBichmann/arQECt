"""Hardware and noise models for the abstract execution."""

from src.models.hardware.trapped_ions import TrappedIonsModel, TrappedIonsHardwareEvents
from src.models.hardware.neutral_atoms import NeutralAtomsModel, NeutralAtomsHardwareEvents
from src.models.hardware.superconducting_qubits import SuperconductingQubitsModel, SuperconductingQubitsHardwareEvents
from src.models.noise.noise import NoiseModel

from typing import Literal

PlatformTag = Literal["TI", "SC", "NA"]
HardwareModel = TrappedIonsModel | NeutralAtomsModel | SuperconductingQubitsModel
Model = HardwareModel | NoiseModel

HardwareEvents = TrappedIonsHardwareEvents | NeutralAtomsHardwareEvents | SuperconductingQubitsHardwareEvents