"""Hardware models and the collectors that observe their abstract execution."""

from src.models.hardware.trapped_ions import TrappedIonsModel
from src.models.hardware.neutral_atoms import NeutralAtomsModel
from src.models.hardware.superconducting_qubits import SuperconductingQubitsModel



HardwareModel = TrappedIonsModel | NeutralAtomsModel | SuperconductingQubitsModel


