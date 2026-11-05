"""Class for modeling superconducting qubit quantum computers.

The focus of this model lies on the fixed connectivity of the qubits and their interaction neighborhoods, instead of the fine-grained details of the concrete qubit layout and virtual routing. This is to allow for a more abstract representation of the hardware, avoiding compiler-specific details to qubit routing and mapping.
We abstract away the details of the actual qubit layout, and instead focus on their interaction neighborhoods, in which are the (only) qubits that they can interact with in entangling gates, as a shared and limited resource. As a consequence, during the abstract execution, each qubit has a neighborhood that becomes a cache which follows a Last-Recently-Used (LRU) policy. It a new neighbor is needed, the least recently used qubit in the neighborhood is swapped out, preferring to move the target qubit in an asymmetric entangling gate.

It uses neighborhood entry and eviction of the qubits as a approximation for general virtual swap operations, thus establishing our execution proxy. Thus, these are the relevant hardware events for the noisy circuit.

A initial optimal layout of the qubits is assumed, so that populating the neighborhoods for the first time is cost free. When there are multiple available degrees, they are distributed dynamically to the qubits, and the maximal degree is preferred.

Different hardware-architectures can be modeled by varying the available qubit degrees and their distribution. Concrete topology constrains, thus connections between the neighborhoods, are however not modelled.
"""

# TODO: this only models immediate neighborhood, and no steps beyond that; as this doesn't model the actual topology we dont have a concrete layout, the focus og SC qubit evaluation lies on the weighted interaction graph and its metrics, not this abstract execution here
    
import random
from src.analysis.analyzer import GateProcessor
from src.models.noise.noise import NoiseModel
import stim

class SuperconductingQubitsModel(GateProcessor):
    def __init__(self, topology: dict[int, int], noise_model: NoiseModel):
        # assume that sum(topology.values()) >= qubits in the circuit
        self.topology: dict[int, int] = topology  # degree -> max number of qubits with that degree
        self.noise_model: NoiseModel = noise_model

        self.hardware_events={"virtual_swap": 0}
        self.noisy_circuit: stim.Circuit = stim.Circuit()

        # sufficient space is implicit: every qubit that is not in the neighborhood of another qubit is considered to be somewhere else in the topology
        self._neighborhoods: dict[int, dict[int, int]] = {}  # qubit -> neighborhood (qubit -> lru_time_step)
        self._lru_time_step: int = 0

    def process(self, gate: stim.CircuitInstruction) -> None:
        if stim.gate_data(gate.name).is_single_qubit_gate:
            qubit = gate.targets_copy()[0].value
            data = stim.gate_data(gate.name)
            if data.produces_measurements:
                self.noisy_circuit.append(gate.name, gate.targets_copy(), self.noise_model.measurement_noise.gate_arguments or gate.gate_args_copy())
            else:
                self.noisy_circuit.append(gate)
                if data.is_reset:
                    if self.noise_model.reset_noise.gate_name != "I":
                        self.noisy_circuit.append(self.noise_model.reset_noise.gate_name, [qubit], self.noise_model.reset_noise.gate_arguments)
                elif self.noise_model.single_qubit_gate_noise.gate_name != "I":
                    self.noisy_circuit.append(self.noise_model.single_qubit_gate_noise.gate_name, [qubit], self.noise_model.single_qubit_gate_noise.gate_arguments)

        if stim.gate_data(gate.name).is_two_qubit_gate:
            src_qubit = gate.targets_copy()[0].value
            tgt_qubit = gate.targets_copy()[1].value

            if src_qubit in self._neighborhoods.get(tgt_qubit, {}) or self._fits((tgt_qubit, src_qubit)):  # Cache Hit or both have room in their neighborhoods
                pass
            elif self._neighborhoods.get(src_qubit): # prefer moving the target qubit to the place of the lru neighbor of src
                oldest = min(self._neighborhoods[src_qubit].values())
                lru_qubit = random.choice([q for q, t in self._neighborhoods[src_qubit].items() if t == oldest])
                self._swap(lru_qubit, tgt_qubit)
            else: # src has no neighbors to swap with, move src instead
                oldest = min(self._neighborhoods[tgt_qubit].values())
                lru_qubit = random.choice([q for q, t in self._neighborhoods[tgt_qubit].items() if t == oldest])
                self._swap(lru_qubit, src_qubit)

            self._neighborhoods.setdefault(src_qubit, {})[tgt_qubit] = self._lru_time_step
            self._neighborhoods.setdefault(tgt_qubit, {})[src_qubit] = self._lru_time_step
            self._lru_time_step += 1

            self.noisy_circuit.append(gate)
            if self.noise_model.two_qubit_gate_noise.gate_name != "I":
                self.noisy_circuit.append(self.noise_model.two_qubit_gate_noise.gate_name, [src_qubit, tgt_qubit], self.noise_model.two_qubit_gate_noise.gate_arguments)

    def _fits(self, grown: tuple[int, ...]) -> bool:
        sizes = sorted(
            [len(n) + grown.count(q) for q, n in self._neighborhoods.items()]
            + [grown.count(q) for q in set(grown) if q not in self._neighborhoods],  # qubits without a neighborhood yet
            reverse=True,
        )
        slots = sorted((d for d, count in self.topology.items() for _ in range(count)), reverse=True)
        return all(s <= d for s, d in zip(sizes, slots))

    def _swap(self, qubit_a: int, qubit_b: int) -> None:
        a_neighborhood = self._neighborhoods.pop(qubit_a, {})
        b_neighborhood = self._neighborhoods.pop(qubit_b, {})

        a_back = {n: self._neighborhoods[n].pop(qubit_a) for n in a_neighborhood if n != qubit_b}
        b_back = {n: self._neighborhoods[n].pop(qubit_b) for n in b_neighborhood if n != qubit_a}

        # inherited edges keep the time step of the qubit pair if it already interacted (shared neighbors), otw -1
        for n in a_back:
            self._neighborhoods[n][qubit_b] = b_neighborhood.get(n, -1)
        for n in b_back:
            self._neighborhoods[n][qubit_a] = a_neighborhood.get(n, -1)

        def other(q: int) -> int:
            return qubit_b if q == qubit_a else qubit_a if q == qubit_b else q

        if b_neighborhood:
            self._neighborhoods[qubit_a] = {
                other(n): t if n == qubit_a else a_neighborhood.get(n, -1) for n, t in b_neighborhood.items()
            }
        if a_neighborhood:
            self._neighborhoods[qubit_b] = {
                other(n): t if n == qubit_b else b_neighborhood.get(n, -1) for n, t in a_neighborhood.items()
            }

        self.hardware_events["virtual_swap"] += 1
        if self.noise_model.hardware_event_noise.gate_name != "I":
            self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [qubit_a], self.noise_model.hardware_event_noise.gate_arguments)
            self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [qubit_b], self.noise_model.hardware_event_noise.gate_arguments)

