"""Compiler for quantum circuits targeting specific hardware."""

import contextlib
import io
import json
import math
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict

import stim
from mqt.core.ir import QuantumComputation
from qiskit import QuantumCircuit, qasm2

from src.models.noise.noise import NoiseModel

from mqt.ionshuttler.linear import (
    Architecture as TIArchitecture, LinearCompiler, CompilationStatus, LinearCompilerConfig, SearchConfig
)
from mqt.ionshuttler.linear.actions import Shuttle, PhysicalSwap, Rx, Rzz
from mqt.ionshuttler.multi_shuttler.main import main as ms_main
from mqt.ionshuttler.multi_shuttler.outside import shuttle as ms_shuttle

from mqt.qmap.na.zoned import (
    ZonedNeutralAtomArchitecture,
    RoutingAgnosticCompiler,
    RoutingAwareCompiler,
)

from mqt.qmap.plugins.qiskit.sc import compile_
import networkx as nx
from mqt.core.ir.operations import OpType
from mqt.qmap.sc import (
    Architecture as SCArchitecture, Configuration as SCConfiguration, Heuristic, InitialLayout, Layering, LookaheadHeuristic,
    Method, map_ as sc_map,
)

from src.models.hardware import ( TrappedIonsModel, TrappedIonsHardwareEvents, NeutralAtomsModel, NeutralAtomsHardwareEvents, SuperconductingQubitsModel, SuperconductingQubitsHardwareEvents)



def run_ti_compiler(hardware: TIArchitecture | dict, circuit: stim.Circuit, max_compile_time: float = 600.0) -> tuple[TrappedIonsHardwareEvents, dict]: # (proxy information, details)
    """Run the trapped ion compiler (MQT IonShuttler) on the given circuit and given hardware architecture."""
    # 1.) 1-to-1 transpilation (one stim target group -> one native op) and format change (stim -> qiskit)
    qc = QuantumCircuit(circuit.num_qubits)
    for op in circuit:
        data = stim.gate_data(op.name)
        for group in op.target_groups():
            ions = [t.value for t in group]
            if data.is_two_qubit_gate:
                qc.rzz(1.0, *ions)
            elif data.produces_measurements or data.is_reset:  # MR counts as one anchor, as in TrappedIonsModel
                qc.rx(1.0, *ions)
            else:
                qc.ry(1.0, *ions)

    hardware_events = TrappedIonsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # ion -> number of completed anchor operations
    num_qubits = circuit.num_qubits

    if isinstance(hardware, TIArchitecture): # linear architecture
        # 2.) compile (initial layout: ions spread evenly over the storage sites)
        arch = hardware
        config = LinearCompilerConfig(search=SearchConfig(max_compile_time=max_compile_time))
        storage_sites = [site for site in range(arch.num_sites) if arch.get_processing_zone(site) is None]
        initial_positions = [storage_sites[i * len(storage_sites) // num_qubits] for i in range(num_qubits)]
        result = LinearCompiler(arch, config=config).compile(qc, initial_positions=initial_positions)
        if result.status is not CompilationStatus.SUCCESS:
            raise RuntimeError(f"Trapped ion compilation did not succeed: {result.status}")

        # 3.) extract events by replaying schedule
        shuttles: Counter[tuple[int, int]] = Counter()  # (qubit, completed anchors) -> number of shuttles of that qubit there
        swaps: Counter[tuple[int, int]] = Counter()  # (qubit, completed anchors) -> number of swaps that qubit took part in there
        for action in result.path:
            if isinstance(action, Shuttle):
                moves = [(action.ion, action.src, action.dst)]
                shuttles[(action.ion, qubit_steps[action.ion])] += 1
            elif isinstance(action, PhysicalSwap):
                moves = [(action.ion_a, action.pos_a, action.pos_b), (action.ion_b, action.pos_b, action.pos_a)]
                swaps[(action.ion_a, qubit_steps[action.ion_a])] += 1
                swaps[(action.ion_b, qubit_steps[action.ion_b])] += 1
            else:
                moves = []
                if isinstance(action, Rzz):
                    qubit_steps[action.ion_a] += 1
                    qubit_steps[action.ion_b] += 1
                elif isinstance(action, Rx):
                    qubit_steps[action.ion] += 1
            for ion, src, dst in moves:
                zone_src, zone_dst = arch.get_processing_zone(src), arch.get_processing_zone(dst)
                if zone_src != zone_dst and zone_src is not None:
                    hardware_events.interaction_zone_eviction.append((ion, qubit_steps[ion]))
                if zone_src != zone_dst and zone_dst is not None:
                    hardware_events.interaction_zone_entry.append((ion, qubit_steps[ion]))

        details = {
            "shuttles": dict(sorted(shuttles.items())),  # (qubit, completed anchors) -> nr of single-site moves of that qubit there
            "swaps": dict(sorted(swaps.items())),  # (qubit, completed anchors) -> nr of swaps that qubit took part in there (each swap appears for both ions)
            "wall_clock_s": result.wall_clock_s,
        }

    else:  # ring or grid architecture
        # 2.) compile  (multi_shuttler is a bit more annoying to run)
        config = {**hardware, "algorithm_name": "circuit", "abs_num_ions": num_qubits}

        timeline: list = []  # ("positions", graph, {ion: edge}) | ("gates", graph, [gate ids])
        get_ions, remove_processed_gates = ms_shuttle.get_ions, ms_shuttle.remove_processed_gates
        def record_positions(graph):  # called every timestep, so it also enforces max_compile_time (multi_shuttler has no time limit)
            if time.perf_counter() > start + max_compile_time:
                raise TimeoutError(f"Trapped ion compilation did not succeed: TIMEOUT after {max_compile_time}s")
            positions = get_ions(graph)
            timeline.append(("positions", graph, positions))
            return positions

        def record_gates(graph, dag, nodes) -> None:
            timeline.append(("gates", graph, [graph.dag_gate_id_lookup[node.node_id] for node in nodes.values()]))
            remove_processed_gates(graph, dag, nodes)

        ms_shuttle.get_ions, ms_shuttle.remove_processed_gates = record_positions, record_gates
        cwd = os.getcwd()
        rng_state = random.getstate()
        try:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                os.makedirs(os.path.join(tmp, "circuit"))
                with open(os.path.join(tmp, "circuit", f"circuit_{num_qubits}.qasm"), "w") as f:
                    f.write(qasm2.dumps(qc))
                os.chdir(tmp)
                start = time.perf_counter()
                ms_main({**config, "qasm_base_dir": tmp})
                wall_clock_s = time.perf_counter() - start
        finally:
            os.chdir(cwd)
            random.setstate(rng_state)
            ms_shuttle.get_ions, ms_shuttle.remove_processed_gates = get_ions, remove_processed_gates

        # 3.) extract events by replaying schedule
        graph = timeline[0][1]
        zone_edges = {tuple(sorted(pz.parking_edge, key=sum)) for pz in graph.pzs}
        edge_graph = nx.line_graph(nx.Graph(graph))  # nodes = trap edges, adjacent if they share a node
        def edge_distance(a, b) -> int:  # an ion occasionally moves more than one edge per timestep (e.g. out of a zone's exit leg)
            return nx.shortest_path_length(edge_graph, a if a in edge_graph else a[::-1], b if b in edge_graph else b[::-1])
        positions = dict(timeline[0][2])
        shuttles = Counter()  # (qubit, completed anchors) -> number of single-edge moves of that qubit there
        for kind, _, data in timeline:
            if kind == "gates":
                for gate_id in data:
                    qubits = graph.gate_info[gate_id].qubits
                    if len(qubits) == 2 or graph.gate_info[gate_id].qasm.startswith("rx"):
                        for qubit in qubits:
                            qubit_steps[qubit] += 1
                continue
            for ion, edge in data.items():
                if edge != positions[ion]:
                    shuttles[(ion, qubit_steps[ion])] += edge_distance(positions[ion], edge)
                    if positions[ion] in zone_edges:
                        hardware_events.interaction_zone_eviction.append((ion, qubit_steps[ion]))
                    if edge in zone_edges:
                        hardware_events.interaction_zone_entry.append((ion, qubit_steps[ion]))
            positions = dict(data)

        details = {
            "shuttles": dict(sorted(shuttles.items())),  # (qubit, completed anchors) -> nr of single-edge moves of that qubit there (incl. being pushed along by a rotation)
            "swaps": {},  # multi_shuttler has no explicit swap operation (blocking ions are pushed along by rotating cycles, counted as shuttles)
            "wall_clock_s": wall_clock_s,
        }

    expected_steps = Counter(t.value for op in circuit for group in op.target_groups() for t in group
                             if stim.gate_data(op.name).is_two_qubit_gate or stim.gate_data(op.name).produces_measurements
                             or stim.gate_data(op.name).is_reset)
    if {qubit: steps for qubit, steps in qubit_steps.items() if steps} != dict(expected_steps):  # idle ions may appear with 0
        raise RuntimeError("Trapped ion schedule replay does not reproduce the anchor operations of the circuit.")
    
    return hardware_events, details


def run_na_compiler(parameters: dict, circuit: stim.Circuit) -> tuple[NeutralAtomsHardwareEvents, dict]:
    """Run the neutral atom compiler (MQT QMAP zoned, storage + entanglement zone) on the given circuit.

    The MQT zoned compiler only supports CZ + 1q gates and has no readout zone / measurements, so the translation is 1:1 on the
    interaction structure: 2q gates -> CZ, measurements/resets (anchors) -> RZ (used as a marker, 1q gates are neither fused nor
    reordered per qubit), other 1q gates -> H. Consequently `storage_readout` / `entanglement_readout` stay empty and
    `readout_zone_capacity` is ignored. Note that the compiler executes 1q gates in place (also inside the entanglement zone).
    Zone crossings are reconstructed by replaying the .naviz schedule, anchored like in `NeutralAtomsModel`.
    """
    # translate stim -> mqt, compacting unused qubit indices (every index would become a physical atom)
    used_qubits = sorted({t.value for op in circuit for t in op.targets_copy() if t.is_qubit_target})
    to_atom = {qubit: atom for atom, qubit in enumerate(used_qubits)}
    qc = QuantumComputation(len(used_qubits))
    expected_steps: Counter[int] = Counter()  # atom -> number of anchor operations in the circuit (sanity check of the replay)
    for op in circuit:
        data = stim.gate_data(op.name)
        if not (data.is_unitary or data.produces_measurements or data.is_reset):
            continue  # TICK, DETECTOR, noise channels, ...
        for group in op.target_groups():
            atoms = [to_atom[t.value] for t in group]
            if data.is_two_qubit_gate:
                qc.cz(*atoms)
                expected_steps.update(atoms)
            elif data.produces_measurements or data.is_reset:  # MR counts as one anchor, as in NeutralAtomsModel
                qc.rz(1.0, *atoms)
                expected_steps.update(atoms)
            else:
                qc.h(*atoms)

    # architecture: storage SLM grid (sep 3), below it one row of `entanglement_zone_capacity` CZ sites (two SLMs, 2 apart, sep 12)
    capacity, num_atoms = parameters["entanglement_zone_capacity"], len(used_qubits)
    storage_cols = max(4 * capacity, math.ceil(math.sqrt(parameters.get("storage_factor", 2) * num_atoms)))  # ~ as wide as the EZ
    storage_rows = math.ceil(parameters.get("storage_factor", 2) * num_atoms / storage_cols)
    ez_x, ez_y, ez_sep = 0, 3 * storage_rows + 20, 12
    arch_spec = {
        "name": f"na_ez{capacity}_q{num_atoms}",
        "operation_duration": {"rydberg_gate": 0.36, "single_qubit_gate": 52, "atom_transfer": 15},
        "operation_fidelity": {"rydberg_gate": 0.995, "single_qubit_gate": 0.9997, "atom_transfer": 0.999},
        "qubit_spec": {"T": 1.5e6},
        "storage_zones": [{"zone_id": 0, "offset": [0, 0], "dimension": [3 * storage_cols, 3 * storage_rows],
                           "slms": [{"id": 0, "site_separation": [3, 3], "r": storage_rows, "c": storage_cols, "location": [0, 0]}]}],
        "entanglement_zones": [{"zone_id": 0, "offset": [ez_x, ez_y], "dimension": [ez_sep * capacity, 10],
                                "slms": [{"id": 1, "site_separation": [ez_sep, 10], "r": 1, "c": capacity, "location": [ez_x, ez_y]},
                                         {"id": 2, "site_separation": [ez_sep, 10], "r": 1, "c": capacity, "location": [ez_x + 2, ez_y]}]}],
        "aods": [{"id": 0, "site_separation": 2, "r": 100, "c": 100}],
        "rydberg_range": [[[ez_x - 2, ez_y - 2], [ez_x + ez_sep * capacity, ez_y + 2]]],
    }
    arch = ZonedNeutralAtomArchitecture.from_json_string(json.dumps(arch_spec))

    # compile (all atoms start in the storage zone, as in NeutralAtomsModel); max_filling_factor=1 to allow using every CZ site
    compiler_cls = RoutingAgnosticCompiler if parameters.get("na_compiler") == "routing_agnostic" else RoutingAwareCompiler
    compiler = compiler_cls(arch, log_level="W", max_filling_factor=1.0)
    start = time.perf_counter()
    naviz = compiler.compile(qc)
    wall_clock_s = time.perf_counter() - start

    # parse .naviz into (op, targets): statements are "atom (x, y) atomN" or "@+ <op> [args] <target | [targets]>",
    # where a target is "atomN" or "(x, y) atomN" (move), and multi-line "[ ... ]" lists are joined into one statement
    instructions, statement = [], ""
    for line in naviz.splitlines():
        statement += " " + line.strip()
        if statement.count("[") == statement.count("]") and statement.strip():
            words = statement.split()
            targets = [(int(a), (float(x), float(y)) if x else None)
                       for x, y, a in re.findall(r"(?:\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*)?\batom(\d+)\b", statement)]
            instructions.append((words[0] if words[0] == "atom" else words[1], targets))
            statement = ""

    # replay schedule: zone crossings per atom transport (load ... store), anchored by (qubit, number of completed anchor operations)
    def location(pos: tuple[float, float]) -> tuple[str, int | None]:  # (zone, CZ site index or None)
        return ("entanglement", int((pos[0] - ez_x) // ez_sep)) if pos[1] >= ez_y - 1 else ("storage", None)

    hardware_events = NeutralAtomsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)
    positions: dict[int, tuple[float, float]] = {}
    in_aod: dict[int, tuple[float, float]] = {}  # atom -> position it was picked up from
    transitions: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    num_jobs = num_cz_gates = 0
    for op, targets in instructions:
        op_counts[op] += 1
        if op == "atom":
            positions.update(targets)
        elif op == "load":
            num_jobs += not in_aod  # a rearrangement job = one AOD pick-up until the AOD is empty again
            in_aod.update((atom, positions[atom]) for atom, _ in targets)
        elif op == "move":
            positions.update(targets)
        elif op == "store":
            for atom, _ in targets:
                (zone_src, site_src), (zone_dst, site_dst) = location(in_aod.pop(atom)), location(positions[atom])
                transitions[f"{zone_src}->{zone_dst}" + ("(same site)" if zone_src == zone_dst == "entanglement" and site_src == site_dst else "")] += 1
                anchor = (used_qubits[atom], qubit_steps[atom])
                if zone_src != zone_dst:
                    hardware_events.storage_entanglement.append(anchor)
                elif zone_src == "entanglement" and site_src != site_dst:
                    hardware_events.inter_entanglement.append(anchor)
        elif op == "cz":
            sites = Counter(location(pos) for atom, pos in positions.items() if atom not in in_aod)
            for atom, pos in positions.items():
                if atom not in in_aod and location(pos)[0] == "entanglement" and sites[location(pos)] == 2:
                    qubit_steps[atom] += 1
                    num_cz_gates += 0.5
        elif op == "rz":  # measurement / reset marker
            for atom, _ in targets:
                qubit_steps[atom] += 1
    if dict(qubit_steps) != {atom: steps for atom, steps in expected_steps.items() if steps}:
        raise RuntimeError("Neutral atom schedule replay does not reproduce the anchor operations of the circuit.")

    num_atom_transports = sum(transitions.values())
    details = {
        "op_counts": dict(op_counts),  # raw .naviz instructions: load, move, store, cz (stages), rz (anchor markers), u (other 1q)
        "transitions": dict(transitions),  # per-atom transports by (source zone -> destination zone)
        "num_atom_transports": num_atom_transports,  # one atom picked up, moved, and dropped off
        "num_trap_transfers": 2 * num_atom_transports,  # SLM <-> AOD transfers (one load + one store per transport)
        "num_rearrangement_jobs": num_jobs,  # parallel AOD jobs (pick-up until the AOD is empty again)
        "num_move_instructions": op_counts["move"],  # raw AOD move steps (each job uses several)
        "num_cz_stages": op_counts["cz"],  # Rydberg pulses
        "num_cz_gates": int(num_cz_gates),
        "compiler_stats": compiler.stats(),  # internal timings in µs
        "wall_clock_s": wall_clock_s,
        "architecture": arch_spec,
        "naviz": naviz,  # full compiled schedule (atomN = compacted qubit index, used_qubits[N] = stim qubit)
    }
    return hardware_events, details



def run_sc_compiler(hardware: SCArchitecture, circuit: stim.Circuit, max_compile_time: float = 600.0) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
    """Run the superconducting qubit compiler (MQT QMAP heuristic mapper) on the given circuit and given hardware architecture."""
    # 1.) 1-to-1 transpilation (one stim target group -> one native op) and format change (stim -> mqt);
    qc = QuantumComputation(circuit.num_qubits)
    for op in circuit:
        data = stim.gate_data(op.name)
        for group in op.target_groups():
            qubits = [t.value for t in group]
            if data.is_two_qubit_gate:
                qc.cx(*qubits)
            elif data.produces_measurements or data.is_reset:  # MR counts as one anchor, as in SuperconductingQubitsModel
                qc.rz(1.0, *qubits)
            else:
                qc.h(*qubits)

    # 2.) compile
    config = SCConfiguration()
    config.method = Method.heuristic
    config.heuristic = Heuristic.gate_count_max_distance
    config.initial_layout = InitialLayout.dynamic
    config.iterative_bidirectional_routing = True
    config.iterative_bidirectional_routing_passes = 10
    config.layering = Layering.individual_gates
    config.lookahead_heuristic = LookaheadHeuristic.gate_count_max_distance
    config.lookaheads = 15
    config.lookahead_factor = 0.5
    config.automatic_layer_splits = False
    config.pre_mapping_optimizations = False
    config.post_mapping_optimizations = False
    config.add_measurements_to_mapped_circuit = False
    config.timeout = int(max_compile_time * 1000)  # ms
    start = time.perf_counter()
    mapped, results = sc_map(qc, hardware, config)
    wall_clock_s = time.perf_counter() - start
    if results.timeout:
        raise TimeoutError(f"Superconducting qubit compilation did not succeed: TIMEOUT after {max_compile_time}s")

    # 3.) extract events by replaying schedule
    active = sorted({q for op in qc for q in [ctl.qubit for ctl in op.controls] + list(op.targets)})
    layout = {physical: active[qubit] for physical, qubit in mapped.output_permutation.items()}  # physical -> circuit qubit
    layout.update(zip(sorted(set(range(hardware.num_qubits)) - set(layout)), sorted(set(range(qc.num_qubits)) - set(active))))
    for op in reversed(list(mapped)):
        if op.type_ == OpType.swap:
            layout[op.targets[0]], layout[op.targets[1]] = layout[op.targets[1]], layout[op.targets[0]]
    hardware_events = SuperconductingQubitsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # qubit -> number of completed anchor operations
    swaps: Counter[tuple[int, int]] = Counter()  # (qubit, completed anchors) -> number of swaps that qubit took part in there
    for op in mapped:
        physicals = [ctl.qubit for ctl in op.controls] + list(op.targets)
        qubits = [layout[p] for p in physicals]
        if op.type_ == OpType.swap:
            hardware_events.virtual_swap.append(((qubits[0], qubit_steps[qubits[0]]), (qubits[1], qubit_steps[qubits[1]])))
            swaps.update((qubit, qubit_steps[qubit]) for qubit in qubits)
            layout[physicals[0]], layout[physicals[1]] = qubits[1], qubits[0]
        elif op.type_ == OpType.x and len(physicals) == 2:  # cx (control, target)
            assert tuple(physicals) in hardware.coupling_map, "mapped CX on uncoupled qubits"
            qubit_steps[qubits[0]] += 1
            qubit_steps[qubits[1]] += 1
        elif op.type_ == OpType.rz:  # measurement / reset marker
            qubit_steps[qubits[0]] += 1
    expected_steps = Counter(t.value for op in circuit for group in op.target_groups() for t in group
                             if stim.gate_data(op.name).is_two_qubit_gate or stim.gate_data(op.name).produces_measurements
                             or stim.gate_data(op.name).is_reset)
    if {qubit: steps for qubit, steps in qubit_steps.items() if steps} != dict(expected_steps):
        raise RuntimeError("Superconducting qubit schedule replay does not reproduce the anchor operations of the circuit.")

    details = {
        "swaps": dict(sorted(swaps.items())),  # (qubit, completed anchors) -> nr of swaps that qubit took part in there (each swap appears for both qubits)
        "wall_clock_s": wall_clock_s,
    }
    return hardware_events, details


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_ti_abstraction(hardware: TrappedIonsModel, circuit: stim.Circuit) -> tuple[TrappedIonsHardwareEvents, dict]:
    """Run the trapped ion abstract execution on the given circuit."""
    start = time.perf_counter()
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))
    wall_clock_s = time.perf_counter() - start

    return hardware.hardware_events, {"wall_clock_s": wall_clock_s}


def run_na_abstraction(hardware: NeutralAtomsModel, circuit: stim.Circuit) -> tuple[NeutralAtomsHardwareEvents, dict]:
    """Run the neutral atom abstract execution on the given circuit."""
    start = time.perf_counter()
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))
    wall_clock_s = time.perf_counter() - start

    return hardware.hardware_events, {"wall_clock_s": wall_clock_s}

def run_sc_abstraction(hardware: SuperconductingQubitsModel, circuit: stim.Circuit) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
    """Run the superconducting qubit abstract execution on the given circuit."""
    start = time.perf_counter()
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))
    wall_clock_s = time.perf_counter() - start

    return hardware.hardware_events, {"wall_clock_s": wall_clock_s}