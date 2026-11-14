"""Metric extractor for the weighted interaction graph of a circuit. It exists to keep the InteractionGraph as a structure separate from the concrete metric extraction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.analysis import MetricExtractor
from src.models.hardware import HardwareModel, TrappedIonsModel, SuperconductingQubitsModel, NeutralAtomsModel

if TYPE_CHECKING:
    from src.structure.interaction_graph import InteractionGraph

class InteractionGraphAnalyzer(MetricExtractor):
    def __init__(self, wig: InteractionGraph, hardware: HardwareModel):
        self.wig = wig
        self.hw = hardware

    def analyze(self) -> dict[str, object]: # assume valid hardware parameters (especially capacities >0, SC topology has at least as many qubits as the wig), symmetric wig
        result: dict[str, object] = {}

        # partner capacity
        result["fraction_of_partner_capacity_exceeding_qubits"] = self._fraction_of_partner_capacity_exceeding_qubits()
        result["weighted_fraction_of_partner_capacity_exceeding_qubits"] = self._weighted_fraction_of_partner_capacity_exceeding_qubits()
        result["fraction_of_effective_partner_capacity_exceeding_qubits"] = self._fraction_of_effective_partner_capacity_exceeding_qubits()
        result["max_partner_capacity_ratio"] = self._max_partner_capacity_ratio()

        # interaction spread
        result["average_interaction_partner_diversity"] = self._average_interaction_partner_diversity()
        result["average_interaction_entropy"], result["average_effective_partner_number"] = self._average_interaction_entropy()
        result["interaction_concentration"] = self._interaction_concentration()

        # global structure vs. hardware
        result["statically_servable_interaction_fraction"] = self._statically_servable_interaction_fraction()
        result["capacity_constrained_qubit_partition"] = self._capacity_constrained_qubit_partition()
        result["non_bipartite_interaction_fraction"] = self._non_bipartite_interaction_fraction()
        result["nr_connected_components"], result["largest_component_fraction"] = self._connected_components()
        return result

    def _fraction_of_partner_capacity_exceeding_qubits(self) -> float:
        """Fraction of interacting qubits with more distinct interaction partners than they can have simultaneously local on the hardware.

        Range [0, 1]. Lower is better (0 = every qubit's partners can stay local).
        """
        if isinstance(self.hw, TrappedIonsModel):
            capacity = {q: self.hw.interaction_zone_capacity - 1 for q in self.wig.graph}
        elif isinstance(self.hw, NeutralAtomsModel):
            capacity = {q: 1 for q in self.wig.graph}
        else:
            hw_degrees = sorted((deg for deg, n in self.hw.topology.items() for _ in range(n)), reverse=True)
            capacity = dict(zip(sorted(self.wig.graph, key=lambda q: len(self.wig.graph[q]), reverse=True), hw_degrees))

        return sum(len(neighbors) > capacity[q] for q, neighbors in self.wig.graph.items()) / len(self.wig.graph)

    def _weighted_fraction_of_partner_capacity_exceeding_qubits(self) -> float:
        """Mean (over all interacting qubits) fraction of a qubit's interactions that go to partners beyond its local capacity, keeping heaviest partners local.

        Range [0, 1). Lower is better (fewer interactions require a partner to be swapped/moved in).
        """
        if isinstance(self.hw, TrappedIonsModel):
            capacity = {q: self.hw.interaction_zone_capacity - 1 for q in self.wig.graph}
        elif isinstance(self.hw, NeutralAtomsModel):
            capacity = {q: 1 for q in self.wig.graph}
        else:
            hw_degrees = sorted((deg for deg, n in self.hw.topology.items() for _ in range(n)), reverse=True)
            capacity = dict(zip(sorted(self.wig.graph, key=lambda q: len(self.wig.graph[q]), reverse=True), hw_degrees))

        non_local = [1 - sum(sorted(neighbors.values(), reverse=True)[:capacity[q]]) / sum(neighbors.values()) for q, neighbors in self.wig.graph.items()]
        return sum(non_local) / len(non_local)

    def _fraction_of_effective_partner_capacity_exceeding_qubits(self) -> float:
        """Fraction of interacting qubits whose effective number of partners exceeds their local capacity. Softer than normal unweighted version.

        Range [0, 1]. Lower is better.
        """
        if isinstance(self.hw, TrappedIonsModel):
            capacity = {q: self.hw.interaction_zone_capacity - 1 for q in self.wig.graph}
        elif isinstance(self.hw, NeutralAtomsModel):
            capacity = {q: 1 for q in self.wig.graph}
        else:
            hw_degrees = sorted((deg for deg, n in self.hw.topology.items() for _ in range(n)), reverse=True)
            capacity = dict(zip(sorted(self.wig.graph, key=lambda q: len(self.wig.graph[q]), reverse=True), hw_degrees))

        exceeding = 0
        for q, neighbors in self.wig.graph.items():
            total = sum(neighbors.values())
            H = -sum(w / total * math.log2(w / total) for w in neighbors.values())
            exceeding += 2**H > capacity[q] + 1e-9
        return exceeding / len(self.wig.graph)

    def _max_partner_capacity_ratio(self) -> float:
        """Maximum over all qubits of (number of distinct partners / local capacity).

        Range (0, inf). <= 1 is optimal (no qubit exceeds its capacity); higher = busier hub qubits that dominate interactions.
        """
        if isinstance(self.hw, TrappedIonsModel):
            capacity = {q: self.hw.interaction_zone_capacity - 1 for q in self.wig.graph}
        elif isinstance(self.hw, NeutralAtomsModel):
            capacity = {q: 1 for q in self.wig.graph}
        else:
            hw_degrees = sorted((deg for deg, n in self.hw.topology.items() for _ in range(n)), reverse=True)
            capacity = dict(zip(sorted(self.wig.graph, key=lambda q: len(self.wig.graph[q]), reverse=True), hw_degrees))

        return max(len(neighbors) / capacity[q] for q, neighbors in self.wig.graph.items())

    def _average_interaction_partner_diversity(self) -> float:
        """Mean fraction of all other qubits a qubit interacts with (edge density of the graph).

        Range [0, 1]. Lower is better (1 = all-to-all interaction).
        """
        return sum(len(neighbors) / (len(self.wig.graph) - 1) for neighbors in self.wig.graph.values()) / len(self.wig.graph)

    def _average_interaction_entropy(self) -> tuple[float, float]:
        """Mean Shannon entropy (bits) of each qubit's interaction weight distribution over its partners, and the mean effective number of partners.

        Range entropy [0, log2(nr_qubits - 1)], effective partners [1, nr_qubits - 1]. Lower is better (interactions concentrated on few partners);
        effective number of partners is comparable to the local capacity.
        """
        entropy = []
        for neighbors in self.wig.graph.values():
            total = sum(neighbors.values())
            entropy.append(-sum(w / total * math.log2(w / total) for w in neighbors.values()))
        return sum(entropy) / len(entropy), sum(2**H for H in entropy) / len(entropy)

    def _interaction_concentration(self) -> float:
        """Mean number of 2q gates per interacting qubit pair (interaction reuse).

        Range [1, inf). Higher is usually better.
        """
        weights = [w for a, neighbors in self.wig.graph.items() for b, w in neighbors.items() if a < b]
        return sum(weights) / len(weights)

    def _statically_servable_interaction_fraction(self) -> float:
        """Fraction of the interaction weight that can be served by a static assignment of local partners, respecting the capacity of both qubits.

        Greedy maximum-weight b-matching: edges are taken heaviest first if both qubits still have free capacity. For SC this is optimistic.
        Range (0, 1]. Higher is better (1 = no swaps/movements needed at all).
        """
        if isinstance(self.hw, TrappedIonsModel):
            capacity = {q: self.hw.interaction_zone_capacity - 1 for q in self.wig.graph}
        elif isinstance(self.hw, NeutralAtomsModel):
            capacity = {q: 1 for q in self.wig.graph}
        else:
            hw_degrees = sorted((deg for deg, n in self.hw.topology.items() for _ in range(n)), reverse=True)
            capacity = dict(zip(sorted(self.wig.graph, key=lambda q: len(self.wig.graph[q]), reverse=True), hw_degrees))

        edges = sorted(((w, a, b) for a, neighbors in self.wig.graph.items() for b, w in neighbors.items() if a < b), reverse=True)
        served, total = 0, 0
        for w, a, b in edges:
            if capacity[a] > 0 and capacity[b] > 0:
                served += w
                capacity[a] -= 1
                capacity[b] -= 1
            total += w
        return served / total

    def _capacity_constrained_qubit_partition(self) -> float:
        """Fraction of the interaction weight cut when the qubits are greedily partitioned into disjoint groups that are local on the hardware.

        Greedy: seed a group with the unassigned qubit of highest total interaction weight, then repeatedly add the unassigned qubit with the most weight into the group. For SC this is pessimistic (real neighborhoods overlap).
        Range [0, 1). Lower is better (0 = the circuit splits into hardware-local clusters).
        """
        star = isinstance(self.hw, SuperconductingQubitsModel)
        if isinstance(self.hw, TrappedIonsModel):
            capacity = {q: self.hw.interaction_zone_capacity - 1 for q in self.wig.graph}
        elif isinstance(self.hw, NeutralAtomsModel):
            capacity = {q: 1 for q in self.wig.graph}
        else:
            hw_degrees = sorted((deg for deg, n in self.hw.topology.items() for _ in range(n)), reverse=True)
            capacity = dict(zip(sorted(self.wig.graph, key=lambda q: len(self.wig.graph[q]), reverse=True), hw_degrees))

        group_of: dict[int, int] = {}
        center: dict[int, int] = {} # group -> seed qubit
        nr_groups = 0
        while len(group_of) < len(self.wig.graph):
            seed = max((q for q in self.wig.graph if q not in group_of), key=lambda q: sum(self.wig.graph[q].values()))
            group = [seed]
            group_of[seed] = nr_groups
            center[nr_groups] = seed
            while len(group) <= capacity[seed]:
                if star:
                    weight_into_group = {q: self.wig.graph[seed].get(q, 0) for q in self.wig.graph if q not in group_of}
                else:
                    weight_into_group = {q: sum(self.wig.graph[g].get(q, 0) for g in group) for q in self.wig.graph if q not in group_of}
                if not weight_into_group or max(weight_into_group.values()) == 0:
                    break
                best = max(weight_into_group, key=weight_into_group.__getitem__)
                group.append(best)
                group_of[best] = nr_groups
            nr_groups += 1

        cut = sum(w for a, neighbors in self.wig.graph.items() for b, w in neighbors.items() if a < b and (group_of[a] != group_of[b] or (star and center[group_of[a]] not in (a, b))))
        total = sum(w for a, neighbors in self.wig.graph.items() for b, w in neighbors.items() if a < b)
        return cut / total

    def _non_bipartite_interaction_fraction(self) -> float:
        """Fraction of the interaction weight that does not fit into a bipartite graph.

        Greedy 2-coloring: qubits (heaviest total weight first) are put on the side where they have less weight to the already colored qubits.
        Range [0, 0.5]. Lower is better for SC usually (0 = bipartite WIG).
        """
        side: dict[int, int] = {}
        for q in sorted(self.wig.graph, key=lambda q: sum(self.wig.graph[q].values()), reverse=True):
            weight_to_side = [sum(w for neighbor, w in self.wig.graph[q].items() if side.get(neighbor) == s) for s in (0, 1)]
            side[q] = 0 if weight_to_side[0] <= weight_to_side[1] else 1

        same_side = sum(w for a, neighbors in self.wig.graph.items() for b, w in neighbors.items() if a < b and side[a] == side[b])
        total = sum(w for a, neighbors in self.wig.graph.items() for b, w in neighbors.items() if a < b)
        return same_side / total

    def _connected_components(self) -> tuple[int, float]:
        """Number of connected components of the WIG and the fraction of interacting qubits in the largest one.

        Range [1, nr_qubits / 2] and (0, 1]. More components and smaller largest component better = independent blocks.
        """
        seen: set[int] = set()
        sizes = []
        for start in self.wig.graph:
            if start in seen:
                continue
            seen.add(start)
            stack = [start]
            size = 0
            while stack:
                q = stack.pop()
                size += 1
                for neighbor in self.wig.graph[q]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            sizes.append(size)

        return len(sizes), max(sizes) / len(self.wig.graph)
