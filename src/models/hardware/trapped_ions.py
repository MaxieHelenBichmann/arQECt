"""Class for modeling trapped ion quantum computers, namely quantum charge-coupled devices.

The focus of this model lies on the interaction zones, instead of the fine-grained details of the individual ions shuttled between different traps. This is to allow for a more abstract representation of the hardware, avoiding compiler-specific details to qubit routing and scheduling.
We abstract away the details of the individual traps, and instead focus on the interaction zones, which are the (only) regions where ions can be manipulated and entangled, as a shared and limited resource. As a consequence, during the abstract execution, the zones become a cache which follows a Last-Recently-Used (LRU) policy.

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
        self.interaction_zone_capacity: int = interaction_zone_capacity
        self.nr_interaction_sites: int = nr_interaction_sites
        self.noise_model: NoiseModel = noise_model

        self.hardware_events={"interaction_zone_entry": 0, "interaction_zone_eviction": 0}
        self.noisy_circuit: stim.Circuit = stim.Circuit()

        self.zones: tuple[dict[int, int], ...] = tuple(dict() for _ in range(self.nr_interaction_sites))
        self.lru_time_step: int = 0

    def process(self, gate: stim.CircuitInstruction, _ :int) -> None:
        """Executes a gate on the trapped ion model, extracting the execution proxy (interaction zone entries and evictions) and constructing a noisy circuit on its basis."""

        def choose_zone(qubit_count: int) -> int:
            free_zones = [i for i, zone in enumerate(self.zones) if len(zone) + qubit_count <= self.interaction_zone_capacity]
            if free_zones:
                return random.choice(free_zones)

            costs = [sum(sorted(zone.values())[:qubit_count]) for zone in self.zones]
            minimal_cost = min(costs)
            minimal_zones = [i for i, cost in enumerate(costs) if cost == minimal_cost]

            return random.choice(minimal_zones) # choose to only evict 1 or 2 "average-oldest" qubits within one zone

        def add_qubit_to_zone(qubit: int, zone_id: int):
            if len(self.zones[zone_id]) < self.interaction_zone_capacity:
                self.zones[zone_id][qubit] = self.lru_time_step
                self.lru_time_step += 1
                self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [qubit], *self.noise_model.hardware_event_noise.gate_arguments)
                self.hardware_events["interaction_zone_entry"] += 1
            else: # evict
                lru_qubit = min(self.zones[zone_id], key=self.zones[zone_id].get)
                del self.zones[zone_id][lru_qubit]
                self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [lru_qubit], *self.noise_model.hardware_event_noise.gate_arguments)
                self.hardware_events["interaction_zone_eviction"] += 1

                self.zones[zone_id][qubit] = self.lru_time_step
                self.lru_time_step += 1
                self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [qubit], *self.noise_model.hardware_event_noise.gate_arguments)
                self.hardware_events["interaction_zone_entry"] += 1

        if stim.gate_data(gate.name).is_single_qubit_gate:
            qubit = gate.targets_copy()[0].value
            zone_id = next((i for i, zone in enumerate(self.zones) if qubit in zone), None)

            if zone_id is not None: # Cache Hit
                self.zones[zone_id][qubit] = self.lru_time_step
                self.lru_time_step += 1
            else: # Cache Miss
                zone_id = choose_zone(1)
                add_qubit_to_zone(qubit, zone_id)

            self.noisy_circuit.append(gate)

        elif stim.gate_data(gate.name).is_two_qubit_gate:
            src_qubit = gate.targets_copy()[0].value
            tgt_qubit = gate.targets_copy()[1].value

            zone_id_src = next((i for i, zone in enumerate(self.zones) if src_qubit in zone), None)
            zone_id_tgt = next((i for i, zone in enumerate(self.zones) if tgt_qubit in zone), None)

            if zone_id_src is not None and zone_id_tgt is not None: # Cache Hit
                if zone_id_src == zone_id_tgt: # both in same zone
                    self.zones[zone_id_src][src_qubit] = self.lru_time_step
                    self.lru_time_step += 1
                    self.zones[zone_id_tgt][tgt_qubit] = self.lru_time_step
                    self.lru_time_step += 1
                else: # both in different zones: Target -> Source
                    del self.zones[zone_id_tgt][tgt_qubit]
                    self.noisy_circuit.append(self.noise_model.hardware_event_noise.gate_name, [tgt_qubit], *self.noise_model.hardware_event_noise.gate_arguments)
                    self.hardware_events["interaction_zone_eviction"] += 1

                    add_qubit_to_zone(tgt_qubit, zone_id_src)

            elif (zone_id_src is None) != (zone_id_tgt is None): # Partial Cache Hit
                if zone_id_src is not None:
                    cached_zone, not_cached_qubit = zone_id_src, tgt_qubit
                else:
                    cached_zone, not_cached_qubit = zone_id_tgt, src_qubit

                add_qubit_to_zone(not_cached_qubit, cached_zone)

            else: # Cache Miss
                zone_id = choose_zone(2)

                add_qubit_to_zone(src_qubit, zone_id)
                add_qubit_to_zone(tgt_qubit, zone_id)
