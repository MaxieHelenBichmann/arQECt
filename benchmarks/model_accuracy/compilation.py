"""Compiler for quantum circuits targeting specific hardware."""

import contextlib
import io
import math
import multiprocessing
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict

import stim
from mqt.core.ir import QuantumComputation
from qiskit import QuantumCircuit, qasm2
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout
from bloqade import squin
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects import place, move
from bloqade.lanes.transform import PhysicalNativeToPlace, PlaceToMove
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.physical.movement import make_physical_placement_strategy
from bloqade.lanes.analysis.atom import AtomInterpreter

from mqt.ionshuttler.linear import (
    Architecture as TIArchitecture, LinearCompiler, CompilationStatus, LinearCompilerConfig, SearchConfig
)
from mqt.ionshuttler.linear.actions import Shuttle, PhysicalSwap, Rx, Rzz
from mqt.ionshuttler.multi_shuttler.main import main as ms_main
from mqt.ionshuttler.multi_shuttler.outside import shuttle as ms_shuttle

from mqt.qmap.na.zoned import (
    ZonedNeutralAtomArchitecture,
    RoutingAwareCompiler,
)

import networkx as nx
from mqt.core.ir.operations import OpType
from mqt.qmap.sc import (
    Architecture as SCArchitecture, Configuration as SCConfiguration, Heuristic, InitialLayout, Layering, LookaheadHeuristic,
    Method, map_ as sc_map,
)

from src.models.hardware import ( TrappedIonsModel, TrappedIonsHardwareEvents, NeutralAtomsModel, NeutralAtomsHardwareEvents, SuperconductingQubitsModel, SuperconductingQubitsHardwareEvents)



def run_ti_mqt_compiler(hardware: TIArchitecture | dict, circuit: stim.Circuit, max_compile_time: float = 600.0) -> tuple[TrappedIonsHardwareEvents, dict]: # (proxy information, details)
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


def run_na_mqt_compiler(hardware: ZonedNeutralAtomArchitecture, circuit: stim.Circuit, max_compile_time: float = 600.0) -> tuple[NeutralAtomsHardwareEvents, dict]:
    """Run the neutral atom compiler (MQT QMAP zoned, routing-aware) on the given circuit and given hardware architecture."""

    # 1.) 1-to-1 transpilation (one stim target group -> one native op) and format change (stim -> mqt)
    qc = QuantumComputation(circuit.num_qubits)
    for op in circuit:
        data = stim.gate_data(op.name)
        for group in op.target_groups():
            atoms = [t.value for t in group]
            if data.is_two_qubit_gate:
                qc.cz(*atoms)
            elif data.produces_measurements or data.is_reset:  # MR counts as one anchor, as in NeutralAtomsModel
                qc.rz(1.0, *atoms)
            else:
                qc.h(*atoms)

    # 2.) compile (all atoms start in the storage zone)
    def compile_in_child(connection) -> None: # enable compilation timeout
        try:
            connection.send(RoutingAwareCompiler(hardware, log_level="E", max_filling_factor=1.0).compile(qc))
        except Exception as error:
            connection.send(error)
    receiver, sender = multiprocessing.Pipe(duplex=False)
    child = multiprocessing.get_context("fork").Process(target=compile_in_child, args=(sender,))
    start = time.perf_counter()
    child.start()
    if not receiver.poll(max_compile_time):
        child.kill()
        raise TimeoutError(f"Neutral atom compilation did not succeed: TIMEOUT after {max_compile_time}s")
    naviz = receiver.recv()
    wall_clock_s = time.perf_counter() - start
    child.join()
    if isinstance(naviz, Exception):
        raise RuntimeError(f"Neutral atom compilation did not succeed: {naviz}")

    # 3.) extract events by replaying schedule
    machine = hardware.to_namachine_string()
    zones = {name: tuple(map(float, corners)) for name, *corners in
             re.findall(r"zone (\w+) \{\s*from: \(([-\d.]+), ([-\d.]+)\)\s*to: \(([-\d.]+), ([-\d.]+)\)", machine)}
    def zone_of(pos: tuple[float, float]) -> str | None: 
        return next((name for name, (x0, y0, x1, y1) in zones.items() if x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1), None)
    ez_traps = [(x, y) for x, y in ((float(x), float(y)) for x, y in re.findall(r"position: \(([-\d.]+), ([-\d.]+)\)", machine)) if zone_of((x, y))]
    def site_of(pos: tuple[float, float]) -> tuple[float, float] | None: 
        if zone_of(pos) is None:
            return None
        trap = min(ez_traps, key=lambda t: math.dist(t, pos))
        partner = min((t for t in ez_traps if t != trap and zone_of(t) == zone_of(trap)), key=lambda t: math.dist(t, trap))
        return min(trap, partner)

    instructions, statement = [], ""
    for line in naviz.splitlines():
        statement += " " + line.strip()
        if statement.count("[") == statement.count("]") and statement.strip():
            words = statement.split()
            targets = [(int(a), (float(x), float(y)) if x else None)
                       for x, y, a in re.findall(r"(?:\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)\s*)?\batom(\d+)\b", statement)]
            instructions.append((words[0] if words[0] == "atom" else words[1], targets, statement))
            statement = ""

    hardware_events = NeutralAtomsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # atom -> number of completed anchor operations
    transports: Counter[tuple[int, int]] = Counter()  # (qubit, completed anchors) -> number of transports (load ... store) of that atom there
    zone_changes: Counter[tuple[int, int]] = Counter()  # ... of which between storage and an entanglement zone
    site_changes: Counter[tuple[int, int]] = Counter()  # ... of which between two different CZ sites (also across entanglement zones)
    storage_moves: Counter[tuple[int, int]] = Counter()  # ... of which within the storage zone
    move_distance: Counter[tuple[int, int]] = Counter()  # (qubit, completed anchors) -> distance (um) the atom travelled there
    positions: dict[int, tuple[float, float]] = {}
    picked_up_at: dict[int, tuple[float, float]] = {}  # atoms currently in the AOD -> where they were picked up
    rearrangement_jobs = 0
    for op, targets, statement in instructions:
        if op == "atom":
            positions.update(targets)
        elif op == "load":
            rearrangement_jobs += not picked_up_at
            picked_up_at.update((atom, positions[atom]) for atom, _ in targets)
        elif op == "move":
            for atom, pos in targets:
                move_distance[(atom, qubit_steps[atom])] += math.dist(positions[atom], pos)
                positions[atom] = pos
        elif op == "store":
            for atom, _ in targets:
                src, dst = site_of(picked_up_at.pop(atom)), site_of(positions[atom])
                anchor = (atom, qubit_steps[atom])
                transports[anchor] += 1
                if (src is None) != (dst is None):
                    zone_changes[anchor] += 1
                    hardware_events.storage_entanglement.append(anchor)
                elif src is None:
                    storage_moves[anchor] += 1
                elif src != dst:
                    site_changes[anchor] += 1
                    hardware_events.inter_entanglement.append(anchor)
        elif op == "cz": 
            pulsed = {name for name in zones if re.search(rf"\b{name}\b", statement)}
            stored = {atom: site_of(pos) for atom, pos in positions.items() if atom not in picked_up_at and zone_of(pos) in pulsed}
            occupancy = Counter(stored.values())
            for atom, site in stored.items():
                if occupancy[site] == 2:
                    qubit_steps[atom] += 1
        elif op == "rz": 
            for atom, _ in targets:
                qubit_steps[atom] += 1

    expected_steps = Counter(t.value for op in circuit for group in op.target_groups() for t in group
                             if stim.gate_data(op.name).is_two_qubit_gate or stim.gate_data(op.name).produces_measurements
                             or stim.gate_data(op.name).is_reset)
    if {qubit: steps for qubit, steps in qubit_steps.items() if steps} != dict(expected_steps):
        raise RuntimeError("Neutral atom schedule replay does not reproduce the anchor operations of the circuit.")

    details = {
        "transports": dict(sorted(transports.items())),  # (qubit, completed anchors) -> nr of atom transports (pick-up, move, drop-off) there
        "zone_changes": dict(sorted(zone_changes.items())),  # ... of which storage <-> entanglement zone
        "entanglement_site_changes": dict(sorted(site_changes.items())),  # ... of which between different CZ sites
        "storage_moves": dict(sorted(storage_moves.items())),  # ... of which within the storage zone (rearrangements)
        "move_distance": dict(sorted(move_distance.items())),  # (qubit, completed anchors) -> distance (um) the atom travelled there
        "rearrangement_jobs": rearrangement_jobs,  # parallel AOD jobs (all atoms moved together count once)
        "wall_clock_s": wall_clock_s,
    }
    return hardware_events, details


def run_na_bloqade_compiler(hardware: dict, circuit: stim.Circuit, max_compile_time: float = 600.0) -> tuple[NeutralAtomsHardwareEvents, dict]:
    """Run Bloqade Lanes on the same trap geometry and CZ sites, using fixed row/column transport lanes (prepared in architecture_preparation.py)."""
    # 1.) 1-to-1 transpilation (one stim target group -> one native op), as for QMAP
    gates = []
    for op in circuit:
        data = stim.gate_data(op.name)
        for group in op.target_groups():
            atoms = [t.value for t in group]
            kind = 2 if data.is_two_qubit_gate else 3 if data.produces_measurements or data.is_reset else 1
            gates.append((kind, atoms[0], atoms[-1]))
    gates = tuple(gates)
    num_qubits = circuit.num_qubits

    # 2.) compile (all atoms start in storage), then collect a plain instruction trace for replay
    architecture, positions, sites = hardware["architecture"], hardware["positions"], hardware["sites"]
    def compile_in_child(connection) -> None:
        try:
            @squin.kernel
            def program():
                atoms = squin.qalloc(num_qubits)
                for gate in gates:
                    if gate[0] == 2:
                        squin.cz(atoms[gate[1]], atoms[gate[2]])
                    elif gate[0] == 3:
                        squin.rz(1.0, atoms[gate[1]])  # measurement/reset anchor, as in QMAP
                    else:
                        squin.ry(1.0, atoms[gate[1]])  # one native R gate; reserves Rz for anchors
                return squin.broadcast.measure(atoms)  # required terminal measurement, excluded from the proxy

            placed = PhysicalNativeToPlace(arch_spec=architecture).emit(program, no_raise=False)
            for atom, statement in enumerate(s for s in placed.callable_region.walk() if isinstance(s, place.NewPinnedQubit)):
                statement.location_address = LocationAddress(atom, 0, 0)
            if num_qubits > hardware["storage_sites"]:
                raise ValueError("The storage zone cannot hold all atoms.")
            SequentialPlacePass(placed.dialects, no_raise=False)(placed)
            placement_strategy = make_physical_placement_strategy(arch_spec=architecture, search_budget=30, return_moves=False)  # atoms stay put between CZ layers
            placement_strategy.restarts = 1  # one deterministic routing search per stage
            compiled = PlaceToMove(
                layout_heuristic=LogicalLayoutHeuristic(arch_spec=architecture),
                placement_strategy=placement_strategy,
            ).emit(placed, no_raise=False)
            frame, _ = AtomInterpreter(compiled.dialects, arch_spec=architecture).run(compiled)
            instructions = []
            for statement in compiled.callable_region.walk():
                if isinstance(statement, move.Move):
                    before = frame.get(statement.current_state).data.qubit_to_locations
                    after = frame.get(statement.result).data.qubit_to_locations
                    instructions.append(("move", [(atom, src.word_id, after[atom].word_id)
                                                   for atom, src in before.items() if src != after[atom]]))
                elif isinstance(statement, move.CZ):
                    controls, targets, _ = frame.get(statement.result).data.get_qubit_pairing(statement.zone_address, architecture)
                    instructions.append(("anchor", [*controls, *targets]))
                elif isinstance(statement, (move.LocalRz, move.GlobalRz)):
                    state = frame.get(statement.result).data
                    atoms = list(state.qubit_to_locations) if isinstance(statement, move.GlobalRz) else [
                        state.locations_to_qubit[loc] for loc in statement.location_addresses]
                    instructions.append(("anchor", atoms))
            connection.send(instructions)
        except Exception as error:
            connection.send(RuntimeError(str(error)))
    receiver, sender = multiprocessing.Pipe(duplex=False)
    child = multiprocessing.get_context("fork").Process(target=compile_in_child, args=(sender,))
    start = time.perf_counter()
    child.start()
    sender.close()
    if not receiver.poll(max_compile_time):
        child.kill()
        child.join()
        receiver.close()
        raise TimeoutError(f"Bloqade compilation did not succeed: TIMEOUT after {max_compile_time}s")
    try:
        instructions = receiver.recv()
    except EOFError as error:
        raise RuntimeError("Bloqade compilation exited without a result.") from error
    finally:
        receiver.close()
        child.join()
    wall_clock_s = time.perf_counter() - start
    if isinstance(instructions, Exception):
        raise RuntimeError(f"Bloqade compilation did not succeed: {instructions}")

    # 3.) extract the same hardware events and movement details by replaying the schedule
    hardware_events = NeutralAtomsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)
    transports: Counter[tuple[int, int]] = Counter()
    zone_changes: Counter[tuple[int, int]] = Counter()
    site_changes: Counter[tuple[int, int]] = Counter()
    storage_moves: Counter[tuple[int, int]] = Counter()
    move_distance: Counter[tuple[int, int]] = Counter()
    rearrangement_jobs = 0
    journey_start: dict[int, int] = {}  # atom -> word where its current journey (all its moves since the last anchor op) began
    def end_journeys() -> None:
        # hardware events: one per journey (net start -> end), since one trip along the fixed lanes is split into many single-hop moves
        for atom, begin in journey_start.items():
            src, dst = sites.get(begin), sites.get(current[atom])
            if (src is None) != (dst is None):
                hardware_events.storage_entanglement.append((atom, qubit_steps[atom]))
            elif src is not None and src != dst:
                hardware_events.inter_entanglement.append((atom, qubit_steps[atom]))
        journey_start.clear()
    current: dict[int, int] = {}  # atom -> current word
    for op, targets in instructions:
        if op == "anchor":
            end_journeys()
            for atom in targets:
                qubit_steps[atom] += 1
        else:
            rearrangement_jobs += 1
            for atom, begin, end in targets:
                journey_start.setdefault(atom, begin)
                current[atom] = end
                # details: real counts per single move
                src, dst = sites.get(begin), sites.get(end)
                anchor = (atom, qubit_steps[atom])
                transports[anchor] += 1
                move_distance[anchor] += math.dist(positions[begin], positions[end])
                if (src is None) != (dst is None):
                    zone_changes[anchor] += 1
                elif src is None:
                    storage_moves[anchor] += 1
                elif src != dst:
                    site_changes[anchor] += 1
    end_journeys()

    expected_steps = Counter(t.value for op in circuit for group in op.target_groups() for t in group
                             if stim.gate_data(op.name).is_two_qubit_gate or stim.gate_data(op.name).produces_measurements
                             or stim.gate_data(op.name).is_reset)
    if {qubit: steps for qubit, steps in qubit_steps.items() if steps} != dict(expected_steps):
        raise RuntimeError("Bloqade schedule replay does not reproduce the anchor operations of the circuit.")

    details = {
        "transports": dict(sorted(transports.items())),
        "zone_changes": dict(sorted(zone_changes.items())),
        "entanglement_site_changes": dict(sorted(site_changes.items())),
        "storage_moves": dict(sorted(storage_moves.items())),
        "move_distance": dict(sorted(move_distance.items())),
        "rearrangement_jobs": rearrangement_jobs,
        "wall_clock_s": wall_clock_s,
    }
    return hardware_events, details


def run_sc_mqt_compiler(hardware: SCArchitecture, circuit: stim.Circuit, max_compile_time: float = 600.0) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
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


def run_sc_sabre_compiler(hardware: CouplingMap, circuit: stim.Circuit, max_compile_time: float = 600.0) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
    """Run the second superconducting qubit compiler (Qiskit SABRE layout and routing) on the given circuit and given hardware architecture.
    SABRE has no time limit, so `max_compile_time` is not enforced."""
    # 1.) 1-to-1 transpilation (one stim target group -> one native op) and format change (stim -> qiskit)
    qc = QuantumCircuit(circuit.num_qubits)
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

    # 2.) compile (only layout + routing, no gate optimization or basis translation, so swaps stay explicit; fixed trials for reproducibility)
    sabre = SabreLayout(hardware, seed=0, max_iterations=4, swap_trials=20, layout_trials=20)
    start = time.perf_counter()
    mapped = PassManager([sabre]).run(qc)
    wall_clock_s = time.perf_counter() - start

    # 3.) extract events by replaying schedule
    layout = {physical: qubit for qubit, physical in enumerate(mapped.layout.initial_index_layout(filter_ancillas=False))}  # physical -> circuit qubit
    hardware_events = SuperconductingQubitsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # qubit -> number of completed anchor operations
    swaps: Counter[tuple[int, int]] = Counter()  # (qubit, completed anchors) -> number of swaps that qubit took part in there
    for instruction in mapped.data:
        physicals = [mapped.find_bit(q).index for q in instruction.qubits]
        qubits = [layout[p] for p in physicals]
        if instruction.operation.name == "swap":
            hardware_events.virtual_swap.append(((qubits[0], qubit_steps[qubits[0]]), (qubits[1], qubit_steps[qubits[1]])))
            swaps.update((qubit, qubit_steps[qubit]) for qubit in qubits)
            layout[physicals[0]], layout[physicals[1]] = qubits[1], qubits[0]
        elif instruction.operation.name == "cx":
            assert hardware.graph.has_edge(*physicals), "mapped CX on uncoupled qubits"
            qubit_steps[qubits[0]] += 1
            qubit_steps[qubits[1]] += 1
        elif instruction.operation.name == "rz":  # measurement / reset marker
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


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_ti_baseline(circuit: stim.Circuit) -> tuple[TrappedIonsHardwareEvents, dict]:
    """Naive per-gate model for comparison: the qubits of every gate enter an interaction zone before and leave it after the gate."""
    hardware_events = TrappedIonsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # qubit -> number of completed anchor operations
    for op in circuit:
        data = stim.gate_data(op.name)
        if not (data.is_single_qubit_gate or data.is_two_qubit_gate):
            continue
        for group in op.target_groups():
            qubits = [t.value for t in group]
            hardware_events.interaction_zone_entry.extend((qubit, qubit_steps[qubit]) for qubit in qubits)
            if data.is_two_qubit_gate or data.produces_measurements or data.is_reset:
                for qubit in qubits:
                    qubit_steps[qubit] += 1
            hardware_events.interaction_zone_eviction.extend((qubit, qubit_steps[qubit]) for qubit in qubits)

    return hardware_events, {}

def run_na_baseline(circuit: stim.Circuit) -> tuple[NeutralAtomsHardwareEvents, dict]:
    """Naive per-gate model for comparison: the qubits of every 2-qubit gate (or measurement) enter the entanglement (or readout) zone from storage before and return after the gate."""
    hardware_events = NeutralAtomsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # qubit -> number of completed anchor operations
    for op in circuit:
        data = stim.gate_data(op.name)
        zone_changes = (hardware_events.storage_entanglement if data.is_two_qubit_gate
                        else hardware_events.storage_readout if data.produces_measurements else None)
        for group in op.target_groups():
            qubits = [t.value for t in group]
            if zone_changes is not None:
                zone_changes.extend((qubit, qubit_steps[qubit]) for qubit in qubits)
            if data.is_two_qubit_gate or data.produces_measurements or data.is_reset:
                for qubit in qubits:
                    qubit_steps[qubit] += 1
            if zone_changes is not None:
                zone_changes.extend((qubit, qubit_steps[qubit]) for qubit in qubits)

    return hardware_events, {}

def run_sc_baseline(circuit: stim.Circuit) -> tuple[SuperconductingQubitsHardwareEvents, dict]:
    """Naive per-gate model for comparison: one virtual swap before and one after every 2-qubit gate (recorded on the two gate qubits)."""
    hardware_events = SuperconductingQubitsHardwareEvents()
    qubit_steps: defaultdict[int, int] = defaultdict(int)  # qubit -> number of completed anchor operations
    for op in circuit:
        data = stim.gate_data(op.name)
        for group in op.target_groups():
            qubits = [t.value for t in group]
            if data.is_two_qubit_gate:
                hardware_events.virtual_swap.append(((qubits[0], qubit_steps[qubits[0]]), (qubits[1], qubit_steps[qubits[1]])))
            if data.is_two_qubit_gate or data.produces_measurements or data.is_reset:
                for qubit in qubits:
                    qubit_steps[qubit] += 1
            if data.is_two_qubit_gate:
                hardware_events.virtual_swap.append(((qubits[0], qubit_steps[qubits[0]]), (qubits[1], qubit_steps[qubits[1]])))

    return hardware_events, {}
