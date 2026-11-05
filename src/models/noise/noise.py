"""Class for storing information about the customizable circuit-level noise model.

With its information, an abstract execution on a hardware model can calculate the noise that would be applied to the circuit.
There are two types of noise modelled here:

A) Direct noise, which has known circuit locations, depending on the fidelities of existing gates in the circuit (e.g. single-qubit gates, two-qubit gates, measurements, etc.).

B) Estimated hardware noise, which is inferred from the hardware events, by running an abstract circuit execution on the constructed hardware model, approximating aspects of the compilation process (e.g. shuttling- or movement-induced noise, idle noise, etc.).


For simplicity, every noise channel is a stochastic Pauli-type channel. Every other type of noise (leakage, amplitude damping, etc.) is approximated using techniques like Pauli-twirling.

The NoiseModel class is parameterized to store the noise information and can be used for any underlying hardware model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Literal

# TODO: have more sophisticated implementation of error channels, especially considering 2-qubit error channels
# TODO: check for invalid inputs 

ErrorType = Literal[
    "BitFlip", "BitPhaseFlip", "PhaseFlip", "None", "Depolarize", "Pauli", "AmplitudeDamping", "Leakage", "Loss", "Measurement"
]
IdleErrorType = Literal["ThermalRelaxation"]


@dataclass
class ErrorChannel:
    """Class for storing information about a single Pauli-error channel with no target information."""

    gate_name: str | None
    gate_arguments: list[float]


def make_x_error_channel(input: tuple[float, ...], _: int) -> ErrorChannel:
    """input[0] is the probability p of an X error occurring"""
    p = input[0]
    return ErrorChannel("X_ERROR", [p])


def make_y_error_channel(input: tuple[float, ...], _: int) -> ErrorChannel:
    """input[0] is the probability p of an Y error occurring"""
    p = input[0]
    return ErrorChannel("Y_ERROR", [p])


def make_z_error_channel(input: tuple[float, ...], _: int) -> ErrorChannel:
    """input[0] is the probability p of an Z error occurring"""
    p = input[0]
    return ErrorChannel("Z_ERROR", [p])


def make_identity_error_channel(input: tuple[float, ...], _: int) -> ErrorChannel:
    return ErrorChannel("I", [])


def make_depolarizing_error_channel(input: tuple[float, ...], qubit: int) -> ErrorChannel:
    """input[0] is the probability p of an depolarizing error occurring"""
    p = input[0]
    if qubit == 1:
        return ErrorChannel("DEPOLARIZE1", [p])
    return ErrorChannel("DEPOLARIZE2", [p])


def make_pauli_error_channel(input: tuple[float, ...], qubit: int) -> ErrorChannel:
    if qubit == 1:
        return ErrorChannel("PAULI_CHANNEL_1", list(input[:3]))
    return ErrorChannel("PAULI_CHANNEL_2", list(input[:15]))


def make_amplitude_damping_channel(input: tuple[float, ...], _: int) -> ErrorChannel:
    gamma = input[0]
    return ErrorChannel(
        "PAULI_CHANNEL_1", [gamma / 4.0, gamma / 4.0, (1 - math.sqrt(1 - gamma)) / 2.0 - gamma / 4.0]
    )  # TODO: verify


def make_loss_channel(input: tuple[float, ...], _: int) -> ErrorChannel:
    raise NotImplementedError  # TODO: find out how to Pauli twirl loss channel


def make_leakage_channel(input: tuple[float, ...], _: int) -> ErrorChannel:
    raise NotImplementedError  # TODO: find out how to Pauli twirl leakage channel


def make_classical_error_channel(input: tuple[float, ...], _: int) -> ErrorChannel:
    p = input[0]
    return ErrorChannel(None, [p])


def make_thermal_relaxation_idle_channel(input: tuple[float, ...]) -> Callable[[float], ErrorChannel]:
    t1, t2 = input[0], input[1]

    def idle_error_channel(t: float) -> ErrorChannel:
        p_relax = 1 - math.exp(-t / t1)
        p_dephase = 1 - math.exp(-t / t2)
        return ErrorChannel("PAULI_CHANNEL_1", [p_relax / 4.0, p_relax / 4.0, p_dephase / 2.0 - p_relax / 4.0]) # TODO: verify

    return idle_error_channel

def make_identity_idle_channel(_: tuple[float, ...]) -> Callable[[float], ErrorChannel]:
    return lambda t: ErrorChannel("I", [])


PauliMapping: Mapping[ErrorType, Callable[[tuple[float, ...], int], ErrorChannel]] = {
    "BitFlip": make_x_error_channel,
    "BitPhaseFlip": make_y_error_channel,
    "PhaseFlip": make_z_error_channel,
    "None": make_identity_error_channel,
    "Depolarize": make_depolarizing_error_channel,
    "Pauli": make_pauli_error_channel,
    "AmplitudeDamping": make_amplitude_damping_channel,
    "Leakage": make_leakage_channel,
    "Loss": make_loss_channel,
    "Measurement": make_classical_error_channel,
}

IdleMapping: Mapping[IdleErrorType, Callable[[tuple[float, ...]], Callable[[float], ErrorChannel]]] = {
    "ThermalRelaxation": make_thermal_relaxation_idle_channel,
    "None": make_identity_idle_channel,
}


@dataclass
class NoiseModel:
    """Class for storing information about the customizable circuit-level noise model."""

    single_qubit_gate_noise: ErrorChannel
    two_qubit_gate_noise: ErrorChannel
    measurement_noise: ErrorChannel
    reset_noise: ErrorChannel
    idle_noise: Callable[[float], ErrorChannel]
    hardware_event_noise: ErrorChannel

    @classmethod
    def from_specs(
        cls,
        single_qubit_gate: tuple[ErrorType, tuple[float, ...]] = ("None", ()),
        two_qubit_gate: tuple[ErrorType, tuple[float, ...]] = ("None", ()),
        measurement: tuple[ErrorType, tuple[float, ...]] = ("None", ()),
        reset: tuple[ErrorType, tuple[float, ...]] = ("None", ()),
        idle: tuple[IdleErrorType, tuple[float, ...]] = ("None", ()),
        hardware_event: tuple[ErrorType, tuple[float, ...]] = ("None", ()),
    ) -> NoiseModel:
        """Build the noise model from (error type, parameters) pairs."""
        return cls(
            single_qubit_gate_noise=PauliMapping[single_qubit_gate[0]](single_qubit_gate[1], 1),
            two_qubit_gate_noise=PauliMapping[two_qubit_gate[0]](two_qubit_gate[1], 2),
            measurement_noise=PauliMapping[measurement[0]](measurement[1], 1),
            reset_noise=PauliMapping[reset[0]](reset[1], 1),
            idle_noise=IdleMapping[idle[0]](idle[1]),
            hardware_event_noise=PauliMapping[hardware_event[0]](hardware_event[1], 1)
        )
