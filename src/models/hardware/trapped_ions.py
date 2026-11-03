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
from src.models.hardware.common import HardwareModel, ExecutionProxyMetrics


class TrappedIons(HardwareModel):
    def __init__(self, interaction_zone_capacity: int, nr_interaction_sites: int):
        self.interaction_zone_capacity = interaction_zone_capacity
        self.nr_interaction_sites = nr_interaction_sites

    def abstract_execution(self, circuit: stim.Circuit, noise_model: NoiseModel) -> tuple[stim.Circuit, ExecutionProxyMetrics]:
        """Executes a circuit on the trapped ion model, extracting the execution proxy (interaction zone entries and evictions) and constructing a noisy circuit on its basis."""
        hardware_events = {"interaction_zone_entry": 0, "interaction_zone_eviction": 0}
        noise_circuit = stim.Circuit()

        zones: tuple[dict[int, int], ...] = tuple(dict() for _ in range(self.nr_interaction_sites))
        lru_time_step = 0
   
        def choose_zone(qubit_count: int) -> int:
            free_zones = [i for i, zone in enumerate(zones) if len(zone) + qubit_count <= self.interaction_zone_capacity]
            if free_zones:
                return random.choice(free_zones)

            costs = min([sum(sorted(zone.values())[:qubit_count]) for zone in zones])
            minimal_cost = min(costs)
            minimal_zones = [i for i, cost in enumerate(costs) if cost == minimal_cost]

            return random.choice(minimal_zones) # need to evict oldest 1 or 2 qubits from same zone 

        for op in circuit:
            for gate in op.target_groups(): # assume there are no stim.CircuitRepeatBlock, DETECTOR, OBSERVABLE_INCLUDE, QUBIT_COORDS, noisy gates, assume only 1q and 2q gates

                if stim.gate_data(op.name).is_single_qubit_gate:
                    qubit = gate[0].qubit_value

                    target_zone = [i for i, zone in enumerate(zones) if qubit in zone.keys()]
                    if target_zone: # Cache Hit
                        zone_id = target_zone[0]
                        zones[zone_id][qubit] = lru_time_step
                        lru_time_step += 1
                    else: # Cache Miss
                        zone_id = choose_zone(1)

                        if len(zones[zone_id]) < self.interaction_zone_capacity:
                            zones[zone_id][qubit] = lru_time_step
                            lru_time_step += 1
                            noise_circuit.append(noise_model.hardware_event_noise.gate_name, [qubit], *noise_model.hardware_event_noise.gate_arguments)
                            hardware_events["interaction_zone_entry"] += 1
                        else: # evict
                            lru_qubit = min(zones[zone_id], key=zones[zone_id].get)
                            del zones[zone_id][lru_qubit]
                            noise_circuit.append(noise_model.hardware_event_noise.gate_name, [lru_qubit], *noise_model.hardware_event_noise.gate_arguments)
                            hardware_events["interaction_zone_eviction"] += 1
            
                            zones[zone_id][qubit] = lru_time_step
                            lru_time_step += 1
                            noise_circuit.append(noise_model.hardware_event_noise.gate_name, [qubit], *noise_model.hardware_event_noise.gate_arguments)
                            hardware_events["interaction_zone_entry"] += 1
                    

                else:
                    pass

               


        return noise_circuit, ExecutionProxyMetrics(hardware_events=hardware_events)
