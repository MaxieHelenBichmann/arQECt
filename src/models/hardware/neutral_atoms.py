"""Class for modeling neutral atom quantum computers, namely a zoned architecture.

The focus of this model lies on the zones - namely the storage, entanglement, and readout zone - instead of the fine-grained details and spacial rearrangement restrictions of the individual atoms moved with optical tweezers. This is to allow for a more abstract representation of the hardware, avoiding compiler-specific details to qubit routing and scheduling.
We abstract away the details of the concrete movements, and instead focus on the zones, each with a specific function, as a shared and limited resource. As a consequence, during the abstract execution, the zones become a cache which follows a Last-Recently-Used (LRU) policy, preferring to move the target qubit in an asymmetric entangling gate.
The storage zones is used to store qubits and perform single-qubit gates, the entanglement zone is used to perform two-qubit gates, and the readout zone is used to perform measurements.

It uses zone entry, eviction and certain movements within a zone of the atoms as a approximation for general movement operations, thus establishing our execution proxy. Thus, these are the relevant hardware events for the noisy circuit.

All atoms are assumed to start in the storage zone.

Different hardware-architectures can be modeled by varying the capacity of the zones, and it is planned to support disallowing single-qubit gates in the storage zone. Concrete topology constrains are however not modelled, as an average movement cost is used.
"""
import random
from src.analysis.analyzer import GateProcessor
from src.models.noise.noise import ErrorChannel, NoiseModel
import stim
from src.models.common import NoiseTag

from dataclasses import dataclass

@dataclass
class NeutralAtomsHardwareEvents: # no order
    storage_entanglement: int
    storage_readout: int
    entanglement_readout: int
    inter_entanglement: int


# TODO: measurement leaves qubit in readout, how realistic?
# TODO: does this 1q/2q gate distinction make sense here on this level, bc probably the 2q gates are will be decomposed into 1q and 2q gates during compilation, aka they aren't reused in the ent zone anyway?

class NeutralAtomsModel(GateProcessor):
    def __init__(self, entanglement_zone_capacity: int, readout_zone_capacity: int, noise_model: NoiseModel):
        # entanglement_zone_capacity in qubit PAIRS
        # assume that storage_zone_capacity >= qubits in the circuit
        if entanglement_zone_capacity < 1:
            raise ValueError("Entanglement zone capacity must be at least 1.")
        if readout_zone_capacity < 1:
            raise ValueError("Readout zone capacity must be at least 1.")

        self.entanglement_zone_capacity: int = entanglement_zone_capacity
        self.readout_zone_capacity: int = readout_zone_capacity
        self.noise_model: NoiseModel = noise_model
        self.hardware_events=NeutralAtomsHardwareEvents(storage_entanglement=0, storage_readout=0, entanglement_readout=0, inter_entanglement=0)

        self.noisy_circuit: stim.Circuit = stim.Circuit()

        # storage zone is implicit: every qubit that is neither in the readout nor in the entanglement zone
        self._readout_zone: dict[int, int] = {} # (qubit -> lru_time_step)
        self._entanglement_zone: tuple[dict[int, int], ...] = tuple(dict() for _ in range(self.entanglement_zone_capacity)) # tuple of pairs (qubit1 -> lru_time_step), (qubit2 -> lru_time_step)
        self._lru_time_step: int = 0

    def process(self, gate: stim.CircuitInstruction) -> None:
        """Executes a gate on the neutral atoms model, extracting the execution proxy (zone entries and evictions) and constructing a noisy circuit on its basis."""

        if stim.gate_data(gate.name).is_single_qubit_gate:
            data = stim.gate_data(gate.name)
            qubit = gate.targets_copy()[0].value
            
            if data.produces_measurements:
                self._add_to_readout_zone(qubit)

                if self.noise_model.measurement_noise.gate_arguments:
                    self.noisy_circuit.append(stim.CircuitInstruction(gate.name, gate.targets_copy(), self.noise_model.measurement_noise.gate_arguments, tag="meas"))
                else:
                    self.noisy_circuit.append(gate)
            else:
                self._add_to_storage_zone(qubit)

                self.noisy_circuit.append(gate)
                if data.is_reset:
                    self._add_noise(self.noise_model.reset_noise, [qubit], "reset")
                else:
                    self._add_noise(self.noise_model.single_qubit_gate_noise, [qubit], "1q")

        elif stim.gate_data(gate.name).is_two_qubit_gate:
            src_qubit = gate.targets_copy()[0].value
            tgt_qubit = gate.targets_copy()[1].value

            pair_id_src = next((i for i, p in enumerate(self._entanglement_zone) if src_qubit in p), None)
            pair_id_tgt = next((i for i, p in enumerate(self._entanglement_zone) if tgt_qubit in p), None)

            if pair_id_src is not None and pair_id_tgt is not None: # Cache Hit
                if pair_id_src == pair_id_tgt: # both in same pair
                    self._entanglement_zone[pair_id_src][src_qubit] = self._lru_time_step
                    self._lru_time_step += 1
                    self._entanglement_zone[pair_id_tgt][tgt_qubit] = self._lru_time_step
                    self._lru_time_step += 1
                else: # both in different pairs: Target -> Source
                    self._entanglement_zone[pair_id_src][src_qubit] = self._lru_time_step
                    self._lru_time_step += 1
                    self._add_qubit_to_entanglement_pair(tgt_qubit, pair_id_src)

            elif (pair_id_src is None) != (pair_id_tgt is None): # Partial Cache Hit
                if pair_id_src is not None:
                    cached_zone, not_cached_qubit, cached_qubit = pair_id_src, tgt_qubit, src_qubit
                else:
                    assert pair_id_tgt is not None  # guaranteed and necessary for mypy
                    cached_zone, not_cached_qubit, cached_qubit = pair_id_tgt, src_qubit, tgt_qubit

                self._entanglement_zone[cached_zone][cached_qubit] = self._lru_time_step
                self._lru_time_step += 1
                self._add_qubit_to_entanglement_pair(not_cached_qubit, cached_zone)

            else: # Cache Miss
                pair_id = self._choose_pair()

                self._add_qubit_to_entanglement_pair(src_qubit, pair_id)
                self._add_qubit_to_entanglement_pair(tgt_qubit, pair_id)

            self.noisy_circuit.append(gate)
            self._add_noise(self.noise_model.two_qubit_gate_noise, [src_qubit, tgt_qubit], "2q")

    def finalize(self) -> None:
        pass
    
    def _add_qubit_to_entanglement_pair(self, qubit: int, pair_id: int) -> None:
        if len(self._entanglement_zone[pair_id]) >= 2: 
            lru_qubit = min(self._entanglement_zone[pair_id], key=self._entanglement_zone[pair_id].__getitem__)
            del self._entanglement_zone[pair_id][lru_qubit]
            self._add_noise(self.noise_model.hardware_event_noise, [lru_qubit], "hw")
            self.hardware_events.storage_entanglement += 1

        other_pair = next((p for p in self._entanglement_zone if qubit in p), None)
        if other_pair is not None: # from another pair in the entanglement zone
            del other_pair[qubit]
            self.hardware_events.inter_entanglement += 1
        elif qubit in self._readout_zone:
            del self._readout_zone[qubit]
            self.hardware_events.entanglement_readout += 1
        else:
            self.hardware_events.storage_entanglement += 1

        self._entanglement_zone[pair_id][qubit] = self._lru_time_step
        self._lru_time_step += 1

        self._add_noise(self.noise_model.hardware_event_noise, [qubit], "hw")

    def _add_to_readout_zone(self, qubit: int) -> None:
        if qubit in self._readout_zone:
            self._readout_zone[qubit] = self._lru_time_step
            self._lru_time_step += 1
            return

        if len(self._readout_zone) >= self.readout_zone_capacity:
            lru_qubit = min(self._readout_zone, key=self._readout_zone.__getitem__)
            del self._readout_zone[lru_qubit]
            self._add_noise(self.noise_model.hardware_event_noise, [lru_qubit], "hw")
            self.hardware_events.storage_readout += 1
        
        pair = next((p for p in self._entanglement_zone if qubit in p), None)
        if pair is not None:
            del pair[qubit]
            self.hardware_events.entanglement_readout += 1
        else:
            self.hardware_events.storage_readout += 1

        self._readout_zone[qubit] = self._lru_time_step
        self._lru_time_step += 1
        self._add_noise(self.noise_model.hardware_event_noise, [qubit], "hw")

    def _add_to_storage_zone(self, qubit: int) -> None:
        pair = next((p for p in self._entanglement_zone if qubit in p), None)
        if pair is not None:
            del pair[qubit]
            self.hardware_events.storage_entanglement += 1
        elif qubit in self._readout_zone:
            del self._readout_zone[qubit]
            self.hardware_events.storage_readout += 1
        else:
            return

        self._add_noise(self.noise_model.hardware_event_noise, [qubit], "hw")

    def _choose_pair(self) -> int:
        def eviction_cost(pair: dict[int, int]) -> tuple[int, int]:
            nr_evictions = len(pair)
            return nr_evictions, sum(pair.values())

        costs = [eviction_cost(pair) for pair in self._entanglement_zone]
        minimal_cost = min(costs)
        minimal_pairs = [i for i, cost in enumerate(costs) if cost == minimal_cost]

        return random.choice(minimal_pairs)

    def _add_noise(self, channel: ErrorChannel, qubits: list[int], tag: NoiseTag) -> None:
        if channel.gate_name != "I":
            self.noisy_circuit.append(stim.CircuitInstruction(channel.gate_name, qubits, channel.gate_arguments, tag=tag))

    def __repr__(self) -> str:
        return (
            f"NeutralAtomsModel(entanglement_zone_capacity={self.entanglement_zone_capacity}, "
            f"readout_zone_capacity={self.readout_zone_capacity})"
        )

    def __str__(self) -> str:
        lines = [
            f"NeutralAtomsModel with {self.entanglement_zone_capacity} entanglement pairs and a readout zone of capacity {self.readout_zone_capacity}:",
            f"  hardware events: {self.hardware_events}",
            "  entanglement zone (qubit: lru time):",
            *(f"    pair {i:>3}: {dict(sorted(pair.items()))}" for i, pair in enumerate(self._entanglement_zone)),
            f"  readout zone (qubit: lru time): {dict(sorted(self._readout_zone.items()))}",
            "  storage zone: all other qubits",
            "  noisy circuit:",
            "  ------- start -------",
            f"{self.noisy_circuit}",
            "  -------- end --------",
        ]
        return "\n".join(lines)