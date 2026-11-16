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
        runtime_result : dict[str, object] = {}
        compilation_result : dict[str, object] = {}

        # reuse metrics
        runtime_result["average_interaction_reuse_distance"] = self._average_interaction_reuse_distance()
        runtime_result["capacity_constrained_reuse"] = self._capacity_constrained_reuse()
        runtime_result["partner_switches_per_qubit"], runtime_result["capacity_constrained_partner_switches_per_qubit"] = self._partner_switches_per_qubit()

        # composition
        runtime_result["share_mid_circuit_measurements"] = self._share_mid_circuit_measurements()
        compilation_result["average_1q_run_length"] = self._average_1q_run_length()
        runtime_result["activity_imbalance"] = self._activity_imbalance()
        return {"runtime": runtime_result, "compilation": compilation_result}

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

    def _partner_switches_per_qubit(self) -> tuple[float, float]:
        """Number of two-sided partner switches per interacting qubit.

        Range [0, 2 * nr_2q_gates / nr_interacting_qubits) each, 0 if no qubit has two 2q gates. Lower is better.
        """
        if isinstance(self.hw, TrappedIonsModel):
            local_capacity = self.hw.interaction_zone_capacity - 1
        elif isinstance(self.hw, NeutralAtomsModel):
            local_capacity = 1
        else:
            local_capacity = sum(deg * n for deg, n in self.hw.topology.items()) / sum(self.hw.topology.values())

        partners = {q: [gate.partner for gate in timeline if gate.partner is not None] for q, timeline in self.qtl.timelines.items()}
        meetings = {q: defaultdict(list) for q in partners}  # qubit -> partner -> positions of their meetings in the qubit's partner list
        for q, p in partners.items():
            for i, partner in enumerate(p):
                meetings[q][partner].append(i)

        switches, capacity_switches = 0, 0
        for a, p in partners.items():
            for k in range(1, len(p)):
                b = p[k]
                m = meetings[a][b].index(k)
                if m == 0:  # first meeting of the pair
                    between = float("inf")
                else:  # distinct other partners in between, worse of both sides
                    ib = meetings[b][a]
                    between = max(len(set(p[meetings[a][b][m - 1] + 1 : k])), len(set(partners[b][ib[m - 1] + 1 : ib[m]])))
                switches += between > 0
                capacity_switches += between >= local_capacity
        interacting = sum(1 for p in partners.values() if p)
        return (switches / interacting, capacity_switches / interacting) if interacting else (0.0, 0.0)

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