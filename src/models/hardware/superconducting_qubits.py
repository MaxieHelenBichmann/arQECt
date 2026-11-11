"""Class for modeling superconducting qubit quantum computers.

The focus of this model lies on the fixed connectivity of the qubits and their interaction neighborhoods, instead of the fine-grained details of the concrete qubit layout and virtual routing. This is to allow for a more abstract representation of the hardware, avoiding compiler-specific details to qubit routing and mapping.
We abstract away the details of the actual qubit layout, and instead focus on their interaction neighborhoods, in which are the (only) qubits that they can interact with in entangling gates, as a shared and limited resource. As a consequence, during the abstract execution, each qubit has a neighborhood that becomes a cache which follows a Last-Recently-Used (LRU) policy. If a new neighbor is needed, the least recently used qubit in the neighborhood is swapped out, preferring to move the target qubit in an asymmetric entangling gate.

It uses neighborhood entry and eviction of the qubits as a approximation for general virtual swap operations, thus establishing our execution proxy. Thus, these are the relevant hardware events for the noisy circuit.

An initial optimal layout of the qubits is assumed, so that populating the neighborhoods for the first time is cost free. When there are multiple available degrees, they are distributed dynamically to the qubits, and the maximal degree is preferred.

Different hardware-architectures can be modeled by varying the available qubit degrees and their distribution. Concrete topology constrains, thus connections between the neighborhoods, are however not modelled.
"""

# TODO: this only models immediate neighborhood, and no steps beyond that; as this doesn't model the actual topology we dont have a concrete layout, the focus og SC qubit evaluation lies on the weighted interaction graph and its metrics, not this abstract execution here
# TODO: T1 and T2 need to be in the same unit ("gate duration")
# TODO: potentially replace this implementation with a interaction graph based one and a greedy static layout
    
import random
from collections import defaultdict
from src.analysis.processor import GateProcessor
from src.models.noise.noise import ErrorChannel, NoiseModel
from src.models.common import GateKind, HardwareEventLocation, NoiseTag
import stim

from dataclasses import dataclass, field

@dataclass
class SuperconductingQubitsHardwareEvents:
    virtual_swap: list[tuple[HardwareEventLocation, HardwareEventLocation]] = field(default_factory=list)

class SuperconductingQubitsModel(GateProcessor):
    def __init__(self, number_of_qubits: int, gate_durations: dict[GateKind, float], topology: dict[int, int], noise_model: NoiseModel | None = None, seed: int = 42):
        if number_of_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")
        if not all(degree > 0 and count >= 0 for degree, count in topology.items()):
            raise ValueError("Degrees and counts in the topology must be positive.")
        if sum(topology.values()) < number_of_qubits:
            raise ValueError("The sum of the topology counts must be at least the number of qubits.")
        if not all(duration >= 0 for duration in gate_durations.values()):
            raise ValueError("All gate durations must be positive.")
        
        self.number_of_qubits: int = number_of_qubits # needed for idle noise
        self.gate_durations: dict[GateKind, float] = gate_durations # needed for idle noise
        self.topology: dict[int, int] = topology  # degree -> max number of qubits with that degree
        self.noise_model: NoiseModel = noise_model if noise_model is not None else NoiseModel.empty()
        self._rng: random.Random = random.Random(seed)

        self.hardware_events: SuperconductingQubitsHardwareEvents = SuperconductingQubitsHardwareEvents()
        self.noisy_circuit: stim.Circuit = stim.Circuit()

        self._qubit_steps: defaultdict[int, int] = defaultdict(int) # qubit -> number of completed anchor operations

        self._neighborhoods: dict[int, dict[int, int]] = {q: {} for q in range(number_of_qubits)}  # qubit -> neighborhood (qubit -> lru_time_step)
        self._lru_time_step: int = 0
        self._idle_time_steps: dict[int, float] = {q: 0 for q in range(number_of_qubits)}  # qubit -> idle time steps since last gate

    def process(self, gate: stim.CircuitInstruction) -> None:
        if stim.gate_data(gate.name).is_single_qubit_gate:
            qubit = gate.targets_copy()[0].value

            data = stim.gate_data(gate.name)
            if data.produces_measurements:
                self._update_idle_noise(self.gate_durations["MS"], qubit)
                if self.noise_model.measurement_noise.gate_arguments:
                    self.noisy_circuit.append(stim.CircuitInstruction(gate.name, gate.targets_copy(), self.noise_model.measurement_noise.gate_arguments, tag="meas"))
                else:
                    self.noisy_circuit.append(gate)
            else:
                self._update_idle_noise(self.gate_durations["SN"], qubit)
                self.noisy_circuit.append(gate)
                if data.is_reset:
                    self._add_noise(self.noise_model.reset_noise, [qubit], "reset")
                else:
                    self._add_noise(self.noise_model.single_qubit_gate_noise, [qubit], "1q")

            if data.produces_measurements or data.is_reset:
                self._qubit_steps[qubit] += 1

        if stim.gate_data(gate.name).is_two_qubit_gate:
            src_qubit = gate.targets_copy()[0].value
            tgt_qubit = gate.targets_copy()[1].value

            self._update_idle_noise(self.gate_durations["EN"], src_qubit, tgt_qubit)

            if src_qubit in self._neighborhoods[tgt_qubit] or self._fits((tgt_qubit, src_qubit)):  # Cache Hit or both have room in their neighborhoods
                pass
            elif self._neighborhoods[src_qubit]: # prefer moving the target qubit to the place of the lru neighbor of src
                oldest = min(self._neighborhoods[src_qubit].values())
                lru_qubit = self._rng.choice([q for q, t in self._neighborhoods[src_qubit].items() if t == oldest])
                self._swap(lru_qubit, tgt_qubit)
            else: # src has no neighbors to swap with, move src instead
                oldest = min(self._neighborhoods[tgt_qubit].values())
                lru_qubit = self._rng.choice([q for q, t in self._neighborhoods[tgt_qubit].items() if t == oldest])
                self._swap(lru_qubit, src_qubit)

            self._neighborhoods[src_qubit][tgt_qubit] = self._lru_time_step
            self._neighborhoods[tgt_qubit][src_qubit] = self._lru_time_step
            self._lru_time_step += 1

            self.noisy_circuit.append(gate)
            self._add_noise(self.noise_model.two_qubit_gate_noise, [src_qubit, tgt_qubit], "2q")

            self._qubit_steps[src_qubit] += 1
            self._qubit_steps[tgt_qubit] += 1

    def finalize(self) -> None:
        pass
    
    def _fits(self, grown: tuple[int, ...]) -> bool:
        sizes = sorted((len(n) + grown.count(q) for q, n in self._neighborhoods.items()), reverse=True)
        slots = sorted((d for d, count in self.topology.items() for _ in range(count)), reverse=True)
        return all(s <= d for s, d in zip(sizes, slots))

    def _swap(self, qubit_a: int, qubit_b: int) -> None:
        a_neighborhood = self._neighborhoods[qubit_a]
        b_neighborhood = self._neighborhoods[qubit_b]

        a_back = {n: self._neighborhoods[n].pop(qubit_a) for n in a_neighborhood if n != qubit_b}
        b_back = {n: self._neighborhoods[n].pop(qubit_b) for n in b_neighborhood if n != qubit_a}

        # inherited edges keep the time step of the qubit pair if it already interacted (shared neighbors), otw -1
        for n in a_back:
            self._neighborhoods[n][qubit_b] = b_neighborhood.get(n, -1)
        for n in b_back:
            self._neighborhoods[n][qubit_a] = a_neighborhood.get(n, -1)

        def other(q: int) -> int:
            return qubit_b if q == qubit_a else qubit_a if q == qubit_b else q

        self._neighborhoods[qubit_a] = {
            other(n): t if n == qubit_a else a_neighborhood.get(n, -1) for n, t in b_neighborhood.items()
        }
        self._neighborhoods[qubit_b] = {
            other(n): t if n == qubit_b else b_neighborhood.get(n, -1) for n, t in a_neighborhood.items()
        }

        self.hardware_events.virtual_swap.append(((qubit_a, self._qubit_steps[qubit_a]), (qubit_b, self._qubit_steps[qubit_b])))
        self._add_noise(self.noise_model.hardware_event_noise, [qubit_a], "hw")
        self._add_noise(self.noise_model.hardware_event_noise, [qubit_b], "hw")

    def _update_idle_noise(self, duration: float, *qubits: int) -> None:
        for q in qubits:
            idle_time = self._idle_time_steps[q]
            if idle_time > 0:
                self._add_noise(self.noise_model.idle_noise(idle_time), [q], "idle")

        for q in self._idle_time_steps.keys():
            if q in qubits:
                self._idle_time_steps[q] = 0
            else:
                self._idle_time_steps[q] += duration

    def _add_noise(self, channel: ErrorChannel, qubits: list[int], tag: NoiseTag) -> None:
        if channel.gate_name != "I":
            self.noisy_circuit.append(stim.CircuitInstruction(channel.gate_name, qubits, channel.gate_arguments, tag=tag))

    def __repr__(self) -> str:
        return (
            f"SuperconductingQubitsModel(number_of_qubits={self.number_of_qubits}, "
            f"gate_durations={self.gate_durations}, topology={self.topology})"
        )

    def __str__(self) -> str:
        lines = [
            f"SuperconductingQubitsModel with {self.number_of_qubits} qubits on topology (degree: count) {self.topology}:",
            f"  gate durations: {self.gate_durations}",
            f"  hardware events: {self.hardware_events}",
            "  neighborhoods (qubit: lru time):",
            *(f"    {q:>3}: {dict(sorted(n.items()))}" for q, n in sorted(self._neighborhoods.items())),
            f"  idle times: {self._idle_time_steps}",
            "  noisy circuit:",
            "  ------- start -------",
            f"{self.noisy_circuit}",
            "  -------- end --------",
        ]
        return "\n".join(lines)

