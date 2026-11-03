"""Class for modeling trapped ion quantum computers, namely quantum charge-coupled devices.

The focus of this model lies on the interaction zones, instead of the fine-grained details of the individual ions shuttled between different traps. This is to allow for a more abstract representation of the hardware, avoiding compiler-specific details to qubit routing and scheduling.
We abstract away the details of the individual traps, and instead focus on the interaction zones, which are the (only) regions where ions can be manipulated and entangled, as a shared and limited resource. As a consequence, during the abstract execution, the zones become a cache which follows a Last-Recently-Used (LRU) policy.

It uses interaction-zone entry and eviction of the ions as a approximation for general shuttling operations, thus establishing our execution proxy. Thus, these are the relevant hardware events for the noisy circuit.

Different hardware-architectures can be modeled by varying the number of interaction zones, and the capacity of the zones. Concrete topology constrains are however not modelled, as an average shuttling cost is used.
"""

import numpy as np
import stim

from src.models.noise.noise import NoiseModel
from src.models.hardware.common import HardwareModel, ExecutionProxyMetrics


class TrappedIons(HardwareModel):
    def __init__(self, interaction_zone_capacity: int, nr_interaction_sites: int):
        self.interaction_zone_capacity = interaction_zone_capacity
        self.nr_interaction_sites = nr_interaction_sites

    def abstract_execution(self, circuit: stim.Circuit, noise_model: NoiseModel) -> tuple[stim.Circuit, ExecutionProxyMetrics]:
        """Executes a circuit on the trapped ion model, extracting the execution proxy (interaction zone entries and evictions) and constructing a noisy circuit on its basis."""
        pass
