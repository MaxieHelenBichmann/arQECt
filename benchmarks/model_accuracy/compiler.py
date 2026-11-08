"""Compiler for quantum circuits targeting specific hardware."""

import contextlib
import io
import json
import math
import os
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
from mqt.ionshuttler.linear.actions import Shuttle, PhysicalSwap, Rx, Rzz, AdvanceTime
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
    """Run the trapped ion compiler (MQT IonShuttler) on the given circuit, for hardware from `prepare_ti_hardware`.

    A `linear` Architecture is compiled with the `linear` compiler (A* search, bounded by `max_compile_time`), a
    `multi_shuttler` config (ring / grid) with `multi_shuttler` (heuristic). Both use their default initial placement.

    Translation is 1:1 (one stim target group -> one native op), only the interaction structure matters for shuttling:
    2q gates -> Rzz, measurements/resets (anchors) -> Rx, other 1q gates -> Ry (all physical, i.e. need the ion in a zone).
    IZ entries/evictions are reconstructed by replaying the schedule, anchored like in `TrappedIonsModel`.
    """
    # Adapt circuit: transpilation + stim -> qiskit
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

    if isinstance(hardware, TIArchitecture):
        arch = hardware
        config = LinearCompilerConfig(search=SearchConfig(max_compile_time=max_compile_time))
        result = LinearCompiler(arch, config=config).compile(qc)
        if result.status is not CompilationStatus.SUCCESS:
            raise RuntimeError(f"Trapped ion compilation did not succeed: {result.status}")

        # replay schedule: a Shuttle moves one ion to an adjacent empty site, a PhysicalSwap exchanges two adjacent ions
        # (reordering, not counted as transport, but it moves ions across a zone boundary if it happens there)
        for action in result.path:
            if isinstance(action, Shuttle):
                moves = [(action.ion, action.src, action.dst)]
            elif isinstance(action, PhysicalSwap):
                moves = [(action.ion_a, action.pos_a, action.pos_b), (action.ion_b, action.pos_b, action.pos_a)]
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

        action_counts = Counter(type(action).__name__ for action in result.path if not isinstance(action, AdvanceTime))
        details = {
            "action_counts": dict(action_counts),  # Shuttle, PhysicalSwap, Rzz, Rx, Ry
            "num_shuttles": action_counts["Shuttle"],  # ion moves by one site
            "num_timesteps": result.num_timesteps,  # makespan
            "wall_clock_s": result.wall_clock_s,
            "architecture": arch.to_dict(),
            "result": result,  # full CompilationResult (result.path holds the ordered actions; ion = stim qubit index)
        }

    else:
        # multi_shuttler reads the circuit from <qasm_base_dir>/<algorithm_name>/<algorithm_name>_<num_ions>.qasm
        config = {**hardware, "algorithm_name": "circuit", "abs_num_ions": num_qubits}

        # multi_shuttler only reports the number of timesteps, so record its per-timestep ion positions (ion -> trap edge)
        # and its finished gates by temporarily wrapping the two functions its scheduling loop calls for that
        timeline: list = []  # ("positions", graph, {ion: edge}) | ("gates", graph, [gate ids])
        get_ions, remove_processed_gates = ms_shuttle.get_ions, ms_shuttle.remove_processed_gates
        def record_positions(graph):
            positions = get_ions(graph)
            timeline.append(("positions", graph, positions))
            return positions

        def record_gates(graph, dag, nodes) -> None:
            timeline.append(("gates", graph, [graph.dag_gate_id_lookup[node.node_id] for node in nodes.values()]))
            remove_processed_gates(graph, dag, nodes)

        ms_shuttle.get_ions, ms_shuttle.remove_processed_gates = record_positions, record_gates
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                os.makedirs(os.path.join(tmp, "circuit"))
                with open(os.path.join(tmp, "circuit", f"circuit_{num_qubits}.qasm"), "w") as f:
                    f.write(qasm2.dumps(qc))
                os.chdir(tmp)  # multi_shuttler writes a runs/ folder into the working directory
                start = time.perf_counter()
                num_timesteps = ms_main({**config, "qasm_base_dir": tmp})
                wall_clock_s = time.perf_counter() - start
        finally:
            os.chdir(cwd)
            ms_shuttle.get_ions, ms_shuttle.remove_processed_gates = get_ions, remove_processed_gates

        # replay: an ion enters / leaves a gate zone when it moves onto / off a zone's parking edge
        graph = timeline[0][1]
        zone_edges = {tuple(sorted(pz.parking_edge, key=sum)) for pz in graph.pzs}
        positions = dict(timeline[0][2])
        moves_per_qubit: Counter[int] = Counter()
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
                    moves_per_qubit[ion] += 1
                    if positions[ion] in zone_edges:
                        hardware_events.interaction_zone_eviction.append((ion, qubit_steps[ion]))
                    if edge in zone_edges:
                        hardware_events.interaction_zone_entry.append((ion, qubit_steps[ion]))
            positions = dict(data)

        details = {
            "num_shuttles": sum(moves_per_qubit.values()),  # ion moves by one trap edge (incl. through junctions)
            "shuttles_per_qubit": dict(sorted(moves_per_qubit.items())),
            "num_timesteps": num_timesteps,  # makespan
            "wall_clock_s": wall_clock_s,
            "architecture": config,
            "nr_trap_edges": sum(1 for *_, t in graph.edges(data="edge_type") if t == "trap"),
            "initial_positions": dict(sorted(timeline[0][2].items())),  # ion -> trap edge
        }

    expected_steps = Counter(t.value for op in circuit for group in op.target_groups() for t in group
                             if stim.gate_data(op.name).is_two_qubit_gate or stim.gate_data(op.name).produces_measurements
                             or stim.gate_data(op.name).is_reset)
    if dict(qubit_steps) != dict(expected_steps):
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



def run_sc_compiler(parameters: dict, circuit: stim.Circuit) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
    """Run the superconducting qubit compiler (MQT QMAP heuristic mapper) on the given circuit.

    Translation is 1:1 on the interaction structure: 2q gates -> CX (control/target order kept), measurements/resets (anchors)
    -> RZ (used as a marker, as in `run_na_compiler`), other 1q gates -> H. Pre-/post-mapping optimizations are disabled, so
    the mapped circuit is the input plus inserted SWAPs (no gate cancellation, fusion or SWAP decomposition).
    The coupling graph is a compact patch of a device lattice (square, hex, heavy-hex, octagonal) with one physical qubit per
    circuit qubit; the lattice is `parameters["layout"]` or inferred from the degrees in the abstract `topology`. With layout
    "random", the largest degrees of the topology are assigned to the circuit qubits (like in `SuperconductingQubitsModel`),
    and among random connected graphs with that degree sequence the one with the smallest average distance is used.
    Virtual swaps are reconstructed by replaying the SWAPs on the layout, anchored like in `SuperconductingQubitsModel`.
    """
    # translate stim -> mqt, compacting unused qubit indices (every index would become a logical qubit to place)
    used_qubits = sorted({t.value for op in circuit for t in op.targets_copy() if t.is_qubit_target})
    to_logical = {qubit: logical for logical, qubit in enumerate(used_qubits)}
    num_qubits = len(used_qubits)
    qc = QuantumComputation(num_qubits)
    expected_ops: defaultdict[int, list] = defaultdict(list)  # logical -> anchor ops in order (sanity check of the replay)
    for op in circuit:
        data = stim.gate_data(op.name)
        if not (data.is_unitary or data.produces_measurements or data.is_reset):
            continue  # TICK, DETECTOR, noise channels, ...
        for group in op.target_groups():
            logicals = [to_logical[t.value] for t in group]
            if data.is_two_qubit_gate:
                qc.cx(*logicals)
                expected_ops[logicals[0]].append(("cx", logicals[1]))
                expected_ops[logicals[1]].append(("cx", logicals[0]))
            elif data.produces_measurements or data.is_reset:  # MR counts as one anchor, as in SuperconductingQubitsModel
                qc.rz(1.0, *logicals)
                expected_ops[logicals[0]].append(("rz",))
            else:
                qc.h(*logicals)

    # A) architecture: one physical qubit per circuit qubit (no empty sites, as in the abstract model)
    # layout: explicit `parameters["layout"]`, otherwise inferred from the degrees in the topology:
    # {4, ...} -> square grid, {3} -> hex (honeycomb), {3, 2, ...} -> heavy-hex, anything else -> random graph.
    # Octagonal (Rigetti Aspen) is 3-regular in the bulk like hex, so it can only be selected explicitly.
    topology_degrees = {d for d, count in parameters["topology"].items() if count > 0}
    lattice_name = parameters.get("layout") or (
        "square" if max(topology_degrees) == 4 else
        "hex" if topology_degrees == {3} else
        "heavy_hex" if max(topology_degrees) == 3 and 2 in topology_degrees else
        "random"
    )
    if lattice_name == "random":
        # degrees = the `num_qubits` largest slots of the topology (like in the abstract model); among random connected
        # graphs with that degree sequence, the one with the smallest average distance is used
        degrees = sorted((d for d, count in parameters["topology"].items() for _ in range(count)), reverse=True)[:num_qubits]
        if len(degrees) < num_qubits:
            raise ValueError(f"Topology {parameters['topology']} has fewer slots than the {num_qubits} circuit qubits.")
        degrees = [min(d, num_qubits - 1) for d in degrees]  # no multi-edges / self-loops
        if sum(degrees) % 2:  # a graph has an even degree sum: drop one stub from the smallest positive degree
            degrees[max(i for i, d in enumerate(degrees) if d > 0)] -= 1
        best_graph, best_distance = None, math.inf
        for seed in range(parameters.get("topology_samples", 200)):
            try:
                graph = nx.random_degree_sequence_graph(degrees, seed=parameters.get("seed", 0) * 100_003 + seed, tries=100)
            except nx.NetworkXError:  # non-graphical sequence / sampling failed
                continue
            if nx.is_connected(graph) and (distance := nx.average_shortest_path_length(graph)) < best_distance:
                best_graph, best_distance = graph, distance
        if best_graph is None:
            raise RuntimeError(f"Could not construct a connected coupling graph with degrees {degrees}.")
    else:
        # device lattice (large enough), nodes with a "pos" attribute; the device is the compact patch of `num_qubits` qubits
        # around the center (by hop distance, ties by euclidean distance), i.e. a small chip of that family
        side = math.ceil(math.sqrt(num_qubits)) + 3
        if lattice_name == "square":
            lattice = nx.grid_2d_graph(side, side)
            nx.set_node_attributes(lattice, {node: node for node in lattice}, "pos")
        elif lattice_name in ("hex", "heavy_hex"):
            hexagons = nx.hexagonal_lattice_graph(side, side)
            if lattice_name == "hex":
                lattice = hexagons
            else:  # heavy-hex: an extra (degree 2) qubit on every edge of the honeycomb
                lattice = nx.Graph()
                lattice.add_nodes_from(hexagons.nodes(data=True))
                for u, v in hexagons.edges:
                    (ux, uy), (vx, vy) = hexagons.nodes[u]["pos"], hexagons.nodes[v]["pos"]
                    lattice.add_node((u, v), pos=((ux + vx) / 2, (uy + vy) / 2))
                    lattice.add_edges_from([(u, (u, v)), ((u, v), v)])
        elif lattice_name == "octagonal":  # grid of 8-qubit rings, neighboring rings coupled by 2 edges (Rigetti Aspen)
            lattice = nx.Graph()
            for row in range(side):
                for col in range(side):
                    for i in range(8):  # node i at angle (i + 0.5) * 45°: 0/7 right, 1/2 top, 3/4 left, 5/6 bottom
                        angle = (i + 0.5) * math.pi / 4
                        lattice.add_node((row, col, i), pos=(3 * col + math.cos(angle), 3 * row + math.sin(angle)))
                        lattice.add_edge((row, col, i), (row, col, (i + 1) % 8))
                    if col > 0:
                        lattice.add_edges_from([((row, col - 1, 0), (row, col, 3)), ((row, col - 1, 7), (row, col, 4))])
                    if row > 0:
                        lattice.add_edges_from([((row - 1, col, 1), (row, col, 6)), ((row - 1, col, 2), (row, col, 5))])
        else:
            raise ValueError(f"Unknown layout {lattice_name!r} (square, hex, heavy_hex, octagonal, random).")
        pos = nx.get_node_attributes(lattice, "pos")
        centroid = [sum(p[k] for p in pos.values()) / len(pos) for k in (0, 1)]
        # grow a patch from a start qubit, always adding the neighboring qubit with the most couplings into the patch, ties by
        # euclidean distance to the patch centroid (keeps it round and closes rings); over several start qubits near the lattice
        # center, the patch with the most couplings (then smallest average distance) is the device
        best_graph, best_key = None, None
        for start in sorted(lattice, key=lambda node: (math.dist(pos[node], centroid), str(node)))[:24]:
            patch = [start]
            while len(patch) < num_qubits:
                frontier = {n for node in patch for n in lattice[node]} - set(patch)
                patch_centroid = [sum(pos[node][k] for node in patch) / len(patch) for k in (0, 1)]
                patch.append(min(frontier, key=lambda node: (-sum(n in patch for n in lattice[node]),
                                                             round(math.dist(pos[node], patch_centroid), 6), str(node))))
            graph = nx.relabel_nodes(lattice.subgraph(patch), {node: i for i, node in enumerate(patch)})  # connected
            key = (-graph.number_of_edges(), nx.average_shortest_path_length(graph) if num_qubits > 1 else 0.0)
            if best_key is None or key < best_key:
                best_graph, best_key = graph, key
        best_distance = best_key[1]
    coupling_map = {edge for u, v in best_graph.edges for edge in [(u, v), (v, u)]}  # bidirectional, no direction reversals
    arch = SCArchitecture(num_qubits, coupling_map)

    # B) compile with the heuristic (A*) mapper; bidirectional routing passes refine the initial layout
    config = SCConfiguration()
    config.method = Method.heuristic
    config.heuristic = Heuristic.gate_count_max_distance
    config.initial_layout = InitialLayout.dynamic
    config.iterative_bidirectional_routing = True
    config.iterative_bidirectional_routing_passes = parameters.get("bidirectional_passes", 10)
    config.layering = Layering.individual_gates
    config.lookahead_heuristic = LookaheadHeuristic.gate_count_max_distance
    config.lookaheads = 15
    config.lookahead_factor = 0.5
    config.automatic_layer_splits = False
    config.pre_mapping_optimizations = False
    config.post_mapping_optimizations = False
    config.add_measurements_to_mapped_circuit = False
    start = time.perf_counter()
    mapped, results = sc_map(qc, arch, config)
    wall_clock_s = time.perf_counter() - start
    if results.timeout:
        raise RuntimeError("Superconducting qubit mapping timed out.")

    # C) replay: SWAPs exchange the logical qubits of two physical qubits, anchored by (qubit, number of completed anchor operations)
    # `mapped.initial_layout` is not reliable with the dynamic layout, so undo the SWAPs on the (correct) output permutation
    layout = {physical: logical for physical, logical in mapped.output_permutation.items()}  # physical -> logical
    for op in reversed(list(mapped)):
        if op.type_ == OpType.swap:
            layout[op.targets[0]], layout[op.targets[1]] = layout[op.targets[1]], layout[op.targets[0]]
    initial_layout = dict(layout)
    hardware_events = SuperconductingQubitsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)
    replayed_ops: defaultdict[int, list] = defaultdict(list)
    swaps_per_qubit: Counter[int] = Counter()
    num_swaps = 0
    for op in mapped:
        physicals = [ctl.qubit for ctl in op.controls] + list(op.targets)
        logicals = [layout[p] for p in physicals]
        if op.type_ == OpType.swap:
            num_swaps += 1
            swaps_per_qubit.update(used_qubits[logical] for logical in logicals)
            hardware_events.virtual_swap.append(tuple((used_qubits[logical], qubit_steps[logical]) for logical in logicals))
            layout[physicals[0]], layout[physicals[1]] = logicals[1], logicals[0]
        elif op.type_ == OpType.x and len(physicals) == 2:  # cx: (control, target)
            assert best_graph.has_edge(*physicals), "mapped CX on uncoupled qubits"
            replayed_ops[logicals[0]].append(("cx", logicals[1]))
            replayed_ops[logicals[1]].append(("cx", logicals[0]))
            qubit_steps[logicals[0]] += 1
            qubit_steps[logicals[1]] += 1
        elif op.type_ == OpType.rz:  # measurement / reset marker
            replayed_ops[logicals[0]].append(("rz",))
            qubit_steps[logicals[0]] += 1
    if replayed_ops != expected_ops:
        raise RuntimeError("Superconducting qubit schedule replay does not reproduce the anchor operations of the circuit.")

    # D) details
    swapped_pairs = Counter(frozenset(q for q, _ in swap) for swap in hardware_events.virtual_swap)
    details = {
        "num_swaps": num_swaps,  # all inserted SWAPs (each = 3 CX on hardware); here every SWAP is between two circuit qubits
        "num_swaps_reported": results.output.swaps,  # QMAP's own count (should equal num_swaps)
        "num_cx_input": results.input_.cnots,
        "num_cx_output": results.output.cnots,  # incl. 3 per SWAP
        "swaps_per_qubit": dict(swaps_per_qubit),  # stim qubit -> number of SWAPs it took part in
        "swapped_pairs": {tuple(sorted(pair)): count for pair, count in swapped_pairs.items()},  # stim qubit pair -> count
        "depth_input": qc.depth(),
        "depth_output": mapped.depth(),
        "initial_layout": {used_qubits[logical]: physical for physical, logical in initial_layout.items()},  # stim qubit -> physical
        "final_layout": {used_qubits[logical]: physical for physical, logical in layout.items()},
        "layout": lattice_name,
        "degrees": dict(best_graph.degree),  # physical qubit -> degree in the used coupling graph
        "coupling_graph_avg_distance": best_distance,
        "coupling_graph_diameter": nx.diameter(best_graph),
        "coupling_map": sorted((u, v) for u, v in best_graph.edges),  # undirected edges
        "mapper_time_s": results.time,
        "wall_clock_s": wall_clock_s,
        "mapped_circuit": mapped.qasm3_str(),  # physical qubits; cx/swap/h/rz (rz = measurement/reset marker)
    }
    return hardware_events, details


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_ti_abstraction(hardware: TrappedIonsModel, circuit: stim.Circuit) -> TrappedIonsHardwareEvents:
    """Run the trapped ion abstract execution on the given circuit."""
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))

    return hardware.hardware_events


def run_na_abstraction(hardware: NeutralAtomsModel, circuit: stim.Circuit) -> NeutralAtomsHardwareEvents:
    """Run the neutral atom abstract execution on the given circuit."""
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))

    return hardware.hardware_events

def run_sc_abstraction(hardware: SuperconductingQubitsModel, circuit: stim.Circuit) -> SuperconductingQubitsHardwareEvents:
    """Run the superconducting qubit abstract execution on the given circuit."""
    for op in circuit:
        for group in op.target_groups():
            hardware.process(stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag))

    return hardware.hardware_events