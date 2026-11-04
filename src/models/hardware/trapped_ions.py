"""Class for modeling trapped ion quantum computers, namely quantum charge-coupled devices.

The focus of this model lies on the interaction zones, instead of the fine-grained details of the individual ions shuttled between different traps. This is to allow for a more abstract representation of the hardware, avoiding compiler-specific details to qubit routing and scheduling.
We abstract away the details of the individual traps, and instead focus on the interaction zones, which are the (only) regions where ions can be manipulated and entangled, as a shared and limited resource. As a consequence, during the abstract execution, the zones become a cache which follows a Last-Recently-Used (LRU) policy, preferring to move the target qubit in an asymmetric entangling gate.

It uses interaction-zone entry and eviction of the ions as a approximation for general shuttling operations, thus establishing our execution proxy. Thus, these are the relevant hardware events for the noisy circuit.

Different hardware-architectures can be modeled by varying the number of interaction zones, and the capacity of the zones. Concrete topology constrains are however not modelled, as an average shuttling cost is used.
"""

import numpy as np
import stim
import random

from src.models.noise.noise import NoiseModel
from src.analysis.analyzer import GateProcessor


class TrappedIonsModel(GateProcessor):
    def __init__(self, interaction_zone_capacity: int, nr_interaction_sites: int, noise_model: NoiseModel):
        if interaction_zone_capacity < 2:
            raise ValueError("Interaction zone capacity must be at least 2.")
        if nr_interaction_sites < 1:
            raise ValueError("Number of interaction sites must be at least 1.")
        
        self.interaction_zone_capacity: int = interaction_zone_capacity 
        self.nr_interaction_sites: int = nr_interaction_sites
        self.noise_model: NoiseModel = noise_model

        self.hardware_events={"interaction_zone_entry": 0, "interaction_zone_eviction": 0}
        self.noisy_circuit: stim.Circuit = stim.Circuit()

        self._zones: tuple[dict[int, int], ...] = tuple(dict() for _ in range(self.nr_interaction_sites))
        self._lru_time_step: int = 0

    def process(self, gate: stim.CircuitInstruction) -> None:
        """Executes a gate on the trapped ion model, extracting the execution proxy (interaction zone entries and evictions) and constructing a noisy circuit on its basis."""

        if stim.gate_data(gate.name).is_single_qubit_gate:
            qubit = gate.targets_copy()[0].value
            zone_id = next((i for i, zone in enumerate(self._zones) if qubit in zone), None)

            if zone_id is not None: # Cache Hit
                self._zones[zone_id][qubit] = self._lru_time_step
                self._lru_time_step += 1
            else: # Cache Miss
                zone_id = self._choose_zone(1)
                self._add_qubit_to_zone(qubit, zone_id)

            self.noisy_circuit.append(gate)

        elif stim.gate_data(gate.name).is_two_qubit_gate:
            src_qubit = gate.targets_copy()[0].value
            tgt_qubit = gate.targets_copy()[1].value

            zone_id_src = next((i for i, zone in enumerate(self._zones) if src_qubit in zone), None)
            zone_id_tgt = next((i for i, zone in enumerate(self._zones) if tgt_qubit in zone), None)

            if zone_id_src is not None and zone_id_tgt is not None: # Cache Hit
                if zone_id_src == zone_id_tgt: # both in same zone
                    self._zones[zone_id_src][src_qubit] = self._lru_time_step
                    self._lru_time_step += 1
                    self._zones[zone_id_tgt][tgt_qubit] = self._lru_time_step
                    self._lru_time_step += 1
                else: # both in different zones: Target -> Source
                    del self._zones[zone_id_tgt][tgt_qubit]
                    self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [tgt_qubit], self.noise_model.hardware_event_noise.gate_arguments)
                    self.hardware_events["interaction_zone_eviction"] += 1

                    self._zones[zone_id_src][src_qubit] = self._lru_time_step
                    self._lru_time_step += 1
                    self._add_qubit_to_zone(tgt_qubit, zone_id_src)

            elif (zone_id_src is None) != (zone_id_tgt is None): # Partial Cache Hit
                if zone_id_src is not None:
                    cached_zone, not_cached_qubit, cached_qubit = zone_id_src, tgt_qubit, src_qubit
                else:
                    assert zone_id_tgt is not None  # guaranteed and necessary for mypy
                    cached_zone, not_cached_qubit, cached_qubit = zone_id_tgt, src_qubit, tgt_qubit

                self._zones[cached_zone][cached_qubit] = self._lru_time_step
                self._lru_time_step += 1
                self._add_qubit_to_zone(not_cached_qubit, cached_zone)

            else: # Cache Miss
                zone_id = self._choose_zone(2)


                self._add_qubit_to_zone(src_qubit, zone_id)
                self._add_qubit_to_zone(tgt_qubit, zone_id)

            self.noisy_circuit.append(gate)

    def _choose_zone(self, qubit_count: int) -> int:
        free_zones = [i for i, zone in enumerate(self._zones) if len(zone) + qubit_count <= self.interaction_zone_capacity]
        if free_zones:
            return random.choice(free_zones)

        costs = [sum(sorted(zone.values())[:qubit_count]) for zone in self._zones]
        minimal_cost = min(costs)
        minimal_zones = [i for i, cost in enumerate(costs) if cost == minimal_cost]

        return random.choice(minimal_zones) # choose zone to only evict 1 or 2 "average-oldest" qubits within one zone

    def _add_qubit_to_zone(self, qubit: int, zone_id: int) -> None:
        if len(self._zones[zone_id]) < self.interaction_zone_capacity:
            self._zones[zone_id][qubit] = self._lru_time_step
            self._lru_time_step += 1
            self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [qubit], self.noise_model.hardware_event_noise.gate_arguments)
            self.hardware_events["interaction_zone_entry"] += 1
        else: # evict (will never evict the cached partner qubit of a two-qubit gate, as it is most recently used and there exists at least one more stale qubit in the zone as it is full and at least of size 2)
            lru_qubit = min(self._zones[zone_id], key=self._zones[zone_id].__getitem__)
            del self._zones[zone_id][lru_qubit]
            self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [lru_qubit], self.noise_model.hardware_event_noise.gate_arguments)
            self.hardware_events["interaction_zone_eviction"] += 1

            self._zones[zone_id][qubit] = self._lru_time_step
            self._lru_time_step += 1
            self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [qubit], self.noise_model.hardware_event_noise.gate_arguments)
            self.hardware_events["interaction_zone_entry"] += 1

    def __repr__(self) -> str:
        return (
            f"TrappedIonsModel(interaction_zone_capacity={self.interaction_zone_capacity}, "
            f"nr_interaction_sites={self.nr_interaction_sites})"
        )

    def __str__(self) -> str:
        lines = [
            f"TrappedIonsModel with {self.nr_interaction_sites} interaction zones of capacity {self.interaction_zone_capacity}:",
            f"  hardware events: {self.hardware_events}",
            "  noisy circuit:",
            "  ------- start -------",
            f"{self.noisy_circuit}",
            "  -------- end --------",
        ]
        return "\n".join(lines)
