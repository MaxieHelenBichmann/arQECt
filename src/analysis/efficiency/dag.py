"""Metric extractor for the dependency graph of a circuit. It exists to keep the DependencyDAG as a structure separate from the concrete metric extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING
import math

from src.analysis import MetricExtractor
from itertools import combinations_with_replacement
from collections import defaultdict


from src.models.hardware import HardwareModel, TrappedIonsModel, SuperconductingQubitsModel, NeutralAtomsModel

if TYPE_CHECKING:
    from src.structure.dependency_dag import DependencyDAG

class DependencyDAGAnalyzer(MetricExtractor):
    def __init__(self, dag: DependencyDAG, hardware: HardwareModel):
        self.dag = dag
        self.hw = hardware

    def analyze(self) -> dict[str, object]: # assume valid hardware parameters (especially capacities/durations >0), dag completed with .finalize()
        result : dict[str, object] = {}

        # parallelizability
        result["critical_path_depth"] = self.dag.depth
        result["2q_depth"] = self._2q_depth()
        result["critical_path_duration"] = self._critical_path_duration()
        result["parallelism_saturation"] = self._parallelism_saturation()
        result["capacity_constrained_depth_bound"] = self._2q_depth()
        result["min_idle_steps"] = self._min_idle_steps()

        # scheduling choices
        result["average_scheduling_slack"] = self._average_scheduling_slack()
        result["average_weighted_scheduling_slack"] = self._average_weighted_scheduling_slack()
        result["no_slack_fraction"] = self._no_slack_fraction()
        result["min_scheduling_window_congestion"] = self._min_scheduling_window_congestion()
        result["max_scheduling_window_congestion"] = self._max_scheduling_window_congestion()

        # layered resource usage
        result["average_asap_qubit_reuse"] = self._average_asap_qubit_reuse()
        result["average_alap_qubit_reuse"] = self._average_alap_qubit_reuse()
        result["average_asap_qubit_interaction"] = self._average_asap_qubit_interaction()
        return result

    def _2q_depth(self) -> int:
        """Longest dependency chain counting only 2q gates.

        Range [0, critical_path_depth]. Lower is better (theoretically more parallelism possible).
        """
        depth_2q: dict[int, int] = {}
        for i in sorted(self.dag.dag):           
            g = self.dag.dag[i]
            d = max((depth_2q[p] for p in g.predecessors), default=0)
            depth_2q[i] = d + (len(g.qubits) == 2)
        return max(depth_2q.values(), default=0)
    
    def _critical_path_duration(self) -> float | None:
        """Longest dependency chain weighted by the hardware's gate durations.

        Range [0, inf), in the same time unit as the gate durations; 0 for an empty circuit.
        Lower is better. This is a runtime lower bound for the given DAG, ignoring routing and shared resources.
        Returns None for hardware without gate durations (currently trapped ions and neutral atoms).
        """
        if not isinstance(self.hw, SuperconductingQubitsModel):
            return None
        finish: dict[int, float] = {}
        for i in sorted(self.dag.dag):
            g = self.dag.dag[i]
            if g.readout:
                duration = self.hw.gate_durations["MS"]
            elif len(g.qubits) == 2:
                duration = self.hw.gate_durations["EN"]
            else:
                duration = self.hw.gate_durations["SN"]
            finish[i] = max((finish[p] for p in g.predecessors), default=0.0) + duration
        return max(finish.values(), default=0.0)

    def _nr_gates(self) -> int:
        """Total number of gates. Lower is better."""
        return len(self.dag.dag)

    def _average_scheduling_slack(self) -> float:
        """Mean slack (ALAP - ASAP time) per gate, i.e. by how many layers a gate can be shifted without increasing depth.

        Range [0, depth - 1]. 0 = fully rigid, higher = more freedom for the scheduler.
        """
        return sum([g.schedule_interval[1] - g.schedule_interval[0] for g in self.dag.dag.values()]) * 1.0 / len(self.dag.dag)

    def _average_weighted_scheduling_slack(self) -> float:
        """Like the average slack, but each gate's slack is weighted by its number of qubits, so flexible 2q gates count double.

        Range [0, 2 * (depth - 1)]. Higher = more freedom, especially for 2q gates.
        """
        return sum([(g.schedule_interval[1] - g.schedule_interval[0]) * len(g.qubits) for g in self.dag.dag.values()]) * 1.0 / len(self.dag.dag)

    def _no_slack_fraction(self) -> float:
        """Fraction of gates with zero scheduling slack (ALAP = ASAP), using unit-duration layers.

        Range [0, 1], 0 for an empty circuit. High = many gates are fixed in any minimum-depth schedule.
        Lower = more freedom to accommodate hardware constraints without increasing depth.
        """
        return sum(g.schedule_interval[0] == g.schedule_interval[1] for g in self.dag.dag.values()) / len(self.dag.dag)
    
    def _parallelism_saturation(self) -> float:
        """Minimum number of layers needed if only zone capacity mattered (ignoring dependencies).

        Range [0, inf), 0 for hardware without zone limits.
        """
        if isinstance(self.hw, TrappedIonsModel):
            nr_2q_gates = sum([1 for g in  self.dag.dag.values() if len(g.qubits) == 2])
            parallel_capacity = self.hw.interaction_zone_capacity * self.hw.nr_interaction_sites

            return math.ceil(nr_2q_gates * 1.0 / parallel_capacity)
        
        elif isinstance(self.hw, NeutralAtomsModel):
            nr_2q_gates = sum([1 for g in  self.dag.dag.values() if len(g.qubits) == 2])
            nr_rd_gates = sum([1 for g in  self.dag.dag.values() if g.readout])

            ent_pressure = math.ceil(nr_2q_gates * 1.0 / self.hw.entanglement_zone_capacity)
            rd_pressure = math.ceil(nr_rd_gates * 1.0 / self.hw.readout_zone_capacity)

            return max(ent_pressure, rd_pressure)
        
        else:
            return 0

    def _capacity_constrained_depth_bound(self) -> float: # parallelism pressure
        """Lower bound on the executable depth: max(critical path depth, layers needed due to zone capacity).

        Range [depth, inf). Optimum = depth (hardware parallelism is not the bottleneck); larger = capacity-limited.
        """
        return max(self.dag.depth, self._parallelism_saturation())

    def _min_scheduling_window_congestion(self) -> float:
        """Fraction of layer windows [s, e] whose must-execute gates (scheduling interval fully inside the window) exceed the window's maximum hardware capacity.

        Range [0, 1]. 0 = optimal (critical path depth is achievable); > 0 = schedule is forced to stretch.
        """
        n = 0
        congestion = 0
        for s, e in combinations_with_replacement(range(1, self.dag.depth + 1), 2):
            if isinstance(self.hw, TrappedIonsModel):
                parallel_capacity = self.hw.interaction_zone_capacity * self.hw.nr_interaction_sites
                must_execute_candidates = sum([1 for g in self.dag.dag.values() if g.schedule_interval[0] >= s and g.schedule_interval[1] <= e])
                if must_execute_candidates > (e - s + 1) * parallel_capacity:
                    congestion += 1
            elif isinstance(self.hw, NeutralAtomsModel):
                parallel_capacity_ent = self.hw.entanglement_zone_capacity
                must_execute_candidates_ent = sum([1 for g in self.dag.dag.values() if g.schedule_interval[0] >= s and g.schedule_interval[1] <= e and len(g.qubits) == 2])

                parallel_capacity_rd = self.hw.readout_zone_capacity
                must_execute_candidates_rd = sum([1 for g in self.dag.dag.values() if g.schedule_interval[0] >= s and g.schedule_interval[1] <= e and g.readout])

                if must_execute_candidates_ent > (e - s + 1) * parallel_capacity_ent or must_execute_candidates_rd > (e - s + 1) * parallel_capacity_rd:
                    congestion += 1
            else:
                return 0.0
            n += 1

        return congestion  * 1.0 / n

    def _max_scheduling_window_congestion(self) -> float:
        """Fraction of layers whose may-execute gates (scheduling interval contains the layer) exceed the capacity.

        Range [0, 1]. 0 = no contention even in the worst case; high = the scheduler must make (hardware-aware) choices (not necessarily a depth increase).
        """
        congestion = 0
        for i in range(1, self.dag.depth + 1):
            if isinstance(self.hw, TrappedIonsModel):
                parallel_capacity = self.hw.interaction_zone_capacity * self.hw.nr_interaction_sites
                may_execute_candidates = sum([1 for g in self.dag.dag.values() if g.schedule_interval[0] <= i and g.schedule_interval[1] >= i])
                if may_execute_candidates > parallel_capacity:
                    congestion += 1
            elif isinstance(self.hw, NeutralAtomsModel):
                parallel_capacity_ent = self.hw.entanglement_zone_capacity
                may_execute_candidates_ent = sum([1 for g in self.dag.dag.values() if g.schedule_interval[0] <= i and g.schedule_interval[1] >= i and len(g.qubits) == 2])

                parallel_capacity_rd = self.hw.readout_zone_capacity
                may_execute_candidates_rd = sum([1 for g in self.dag.dag.values() if g.schedule_interval[0] <= i and g.schedule_interval[1] >= i and g.readout])

                if may_execute_candidates_ent > parallel_capacity_ent or may_execute_candidates_rd > parallel_capacity_rd:
                    congestion += 1
            else:
                return 0.0

        return congestion * 1.0 / self.dag.depth

    def _average_asap_qubit_reuse(self) -> float:
        """Mean fraction of qubits active in ASAP layer i that are also active in layer i+1.

        Range [0, 1]. High = qubits are busy back-to-back (possibly can be reused); low = qubits idle between layers.
        """
        gate_asap_layer: defaultdict[int, list[int]] = defaultdict(list)
        for idx, g in self.dag.dag.items():
            gate_asap_layer[g.schedule_interval[0]].append(idx)

        reuse = []
        for i in range(1, self.dag.depth):
            qubits_i = {q for g in gate_asap_layer[i] for q in self.dag.dag[g].qubits}
            qubits_next = {q for g in gate_asap_layer[i+1] for q in self.dag.dag[g].qubits}
            reuse.append(len(qubits_i & qubits_next) * 1.0 / len(qubits_i))

        return sum(reuse) / len(reuse)

    def _average_alap_qubit_reuse(self) -> float:
        """Same as the ASAP qubit reuse, but on the ALAP schedule. Range [0, 1], interpretation as for ASAP."""
        gate_alap_layer: defaultdict[int, list[int]] = defaultdict(list)
        for idx, g in self.dag.dag.items():
            gate_alap_layer[g.schedule_interval[1]].append(idx)

        reuse = []
        for i in range(1, self.dag.depth):
            qubits_i = {q for g in gate_alap_layer[i] for q in self.dag.dag[g].qubits}
            qubits_next = {q for g in gate_alap_layer[i+1] for q in self.dag.dag[g].qubits}
            reuse.append(len(qubits_i & qubits_next) * 1.0 / len(qubits_i))

        return sum(reuse) / len(reuse)

    def _average_asap_qubit_interaction(self, window: int = 4) -> float:
        """Interaction graph density within sliding constant-sized windows of consecutive ASAP layers.

        Per window: unique interacting qubit pairs / possibly interacting qubit pairs.
        Range [0, 1]. Low (better) = the same few pairs interact repeatedly (locality);
        1 = all-to-all interaction within the window (worst for routing).
        """
        gate_asap_layer: defaultdict[int, list[int]] = defaultdict(list)
        for idx, g in self.dag.dag.items():
            gate_asap_layer[g.schedule_interval[0]].append(idx)

        interaction = []
        for s in range(1, max(self.dag.depth - window + 1, 1) + 1):
            pairs = {frozenset(self.dag.dag[g].qubits) for i in range(s, s + window) for g in gate_asap_layer[i] if len(self.dag.dag[g].qubits) == 2}
            active = {q for p in pairs for q in p}
            if len(active) >= 3:
                interaction.append(len(pairs) * 1.0 / math.comb(len(active), 2))

        return sum(interaction) / len(interaction) if interaction else 0.0

    def _min_idle_steps(self) -> int:
        """Lower bound on unavoidable idle layers, summed over all qubits between consecutive gates.

        Range [0, inf). Lower is better (less forced waiting while a qubit stores its state).
        """
        last_gate: dict[int, int] = {}
        idle = 0
        for i in sorted(self.dag.dag):
            g = self.dag.dag[i]
            for q in g.qubits:
                if q in last_gate:
                    previous = last_gate[q]
                    finish = {previous: 0}
                    for j in range(previous + 1, i):
                        reachable = [finish[p] for p in self.dag.dag[j].predecessors if p in finish]
                        if reachable:
                            finish[j] = max(reachable) + 1
                    idle += max((finish[p] for p in g.predecessors if p in finish), default=0)
                last_gate[q] = i
        return idle
