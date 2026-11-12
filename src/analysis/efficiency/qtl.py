"""Metric extractor for the qubit timelines of a circuit. It exists to keep the QubitTimelines as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from src.analysis import MetricExtractor
from src.models.hardware import HardwareModel, TrappedIonsModel, SuperconductingQubitsModel, NeutralAtomsModel

if TYPE_CHECKING:
    from src.structure.qubit_timelines import QubitTimelines


class QubitTimelinesAnalyzer(MetricExtractor):
    def __init__(self, qtl: QubitTimelines, hardware: HardwareModel):
        self.qtl = qtl
        self.hw = hardware

    def analyze(self) -> dict[str, object]:
        result: dict[str, object] = {}

        # reuse metrics
        result["average_interaction_reuse_distance"] = self._average_interaction_reuse_distance()
        result["share_partner_reusing_qubits"] = self._share_partner_reusing_qubits()
        result["capacity_constrained_reuse"] = self._capacity_constrained_reuse()
        result["partner_switching_frequency"] = self._partner_switching_frequency()
        result["zone_switching_frequency"] = self._zone_switching_frequency()

        # composition
        result["share_mid_circuit_measurements"] = self._share_mid_circuit_measurements()
        result["_average_1q_run_length"] = self._average_1q_run_length()
        result["_activity_imbalance"] = self._activity_imbalance()
        return result

    def _average_interaction_reuse_distance(self) -> float:
        """Average distance (in operations on the qubit) between consecutive interactions with the same partner, averaged per qubit and then over qubits.

        Range [1, inf), 0 if no qubit reuses a partner. Lower is better.
        """
        averages = []
        for timeline in self.qtl.timelines.values():
            interactions = defaultdict(list)
            for i, gate in enumerate(timeline):
                if gate.partner is not None:
                    interactions[gate.partner].append(i)
            gaps = [b - a for ia in interactions.values() for a, b in zip(ia, ia[1:])]
            if gaps:
                averages.append(sum(gaps) / len(gaps))
        return sum(averages) / len(averages) if averages else 0.0

    def _share_partner_reusing_qubits(self) -> float:
        """Fraction of interacting qubits that interact with some partner more than once.

        Range [0, 1], 0 if no qubit interacts. Higher = more interaction locality to exploit.
        """
        partners = [[gate.partner for gate in timeline if gate.partner is not None] for timeline in self.qtl.timelines.values()]
        interacting = [p for p in partners if p]
        return sum(len(set(p)) < len(p) for p in interacting) / len(interacting) if interacting else 0.0

    def _capacity_constrained_reuse(self) -> float:
        """Fraction of partner reuses that fit the qubit's local capacity, i.e. that have the potential to be reused on the hardware. ( 1 - capacity_constrained_reuse ) are the theoretical reuses that cannot be exploited because of hardware constraints.

        Range [0, 1], 0 if no qubit reuses a partner. Higher is better.
        """
        if isinstance(self.hw, TrappedIonsModel):
            local_capacity = self.hw.interaction_zone_capacity - 1
        elif isinstance(self.hw, NeutralAtomsModel):
            local_capacity = 1
        else:
            local_capacity = sum(deg * n for deg, n in self.hw.topology.items()) / sum(self.hw.topology.values())

        fits, reuses = 0, 0
        for timeline in self.qtl.timelines.values():
            last_seen: dict[int, int] = {} 
            for i, gate in enumerate(timeline):
                if gate.partner is None:
                    continue
                if gate.partner in last_seen:
                    between = timeline[last_seen[gate.partner] + 1 : i]
                    if isinstance(self.hw, NeutralAtomsModel):
                        fits += not between
                    else:
                        fits += len({g.partner for g in between if g.partner is not None}) < local_capacity
                    reuses += 1
                last_seen[gate.partner] = i
        return fits / reuses if reuses else 0.0

    def _partner_switching_frequency(self) -> float:
        """Fraction of consecutive 2q gates on a qubit (ignoring 1q gates in between) whose partner differs from the previous one.

        Range [0, 1], 0 if no qubit has two 2q gates. Lower is better.
        """
        switches, transitions = 0, 0
        for timeline in self.qtl.timelines.values():
            partners = [gate.partner for gate in timeline if gate.partner is not None]
            switches += sum(a != b for a, b in zip(partners, partners[1:]))
            transitions += max(len(partners) - 1, 0)
        return switches / transitions if transitions else 0.0

    def _zone_switching_frequency(self) -> float | None:
        """Fraction of operations that require the qubit to necessarily change zones.

        Range [0, 1]. Lower is better.
        """
        switches, operations = 0, 0
        if isinstance(self.hw, NeutralAtomsModel):
            for timeline in self.qtl.timelines.values():
                kind = "SN"
                for gate in timeline:
                    switches += gate.kind != kind
                    kind = gate.kind
                operations += len(timeline)
        elif isinstance(self.hw, TrappedIonsModel):
            for timeline in self.qtl.timelines.values():
                switches += 1
                operations += len(timeline)
        else:
            return None
        return switches / operations if operations else 0.0

    def _share_mid_circuit_measurements(self) -> float:
        """Fraction of measurements that are followed by further operations on the same qubit.

        Range [0, 1], 0 if there are no measurements. Lower is better.
        """
        mid, measurements = 0, 0
        for timeline in self.qtl.timelines.values():
            for i, gate in enumerate(timeline):
                if gate.kind == "MS":
                    measurements += 1
                    mid += i < len(timeline) - 1
        return mid / measurements if measurements else 0.0

    def _activity_imbalance(self) -> float:
        """Number of operations on the busiest qubit divided by the average number of operations per qubit.

        Range [1, nr_qubits]. 1 = all qubits equally busy (better); higher = hotspot qubits.
        """
        counts = [len(timeline) for timeline in self.qtl.timelines.values()]
        return max(counts) / (sum(counts) / len(counts))

    def _average_1q_run_length(self) -> float:
        """Average length of maximal runs of consecutive 1q gates / resets on a qubit.

        Range [1, inf), 0 if there are no 1q gates. Higher = better (more potential to fuse 1q gates into one native gate).
        """
        runs = []
        for timeline in self.qtl.timelines.values():
            length = 0
            for gate in timeline:
                if gate.kind == "SN":
                    length += 1
                elif length:
                    runs.append(length)
                    length = 0
            if length:
                runs.append(length)
        return sum(runs) / len(runs) if runs else 0.0