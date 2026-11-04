"""Hardware and noise models for the abstract execution."""

from src.models.hardware.trapped_ions import TrappedIonsModel
from src.models.hardware.neutral_atoms import NeutralAtomsModel
from src.models.hardware.superconducting_qubits import SuperconductingQubitsModel
from src.models.noise.noise import NoiseModel


HardwareModel = TrappedIonsModel | NeutralAtomsModel | SuperconductingQubitsModel
Model = HardwareModel | NoiseModel