"""Constructs specific hardware."""

import math
from collections import Counter

import networkx as nx

from src.models.hardware import ( TrappedIonsModel, NeutralAtomsModel, SuperconductingQubitsModel)
from mqt.ionshuttler.linear import Architecture as TIArchitecture
from mqt.qmap.sc import Architecture as SCArchitecture

HardwareArchitecture = TIArchitecture | SCArchitecture | dict

# native gate sets (canonical stim gate names), matching the 1-to-1 transpilation in compilation.py
TI_NATIVE_GATES: set[str] = {"SQRT_XX", "SQRT_X", "SQRT_Y", "M", "R"}
NA_NATIVE_GATES: set[str] = {"CZ", "H", "S", "M", "R"}
SC_NATIVE_GATES: set[str] = {"CX", "H", "S", "M", "R"}

def prepare_ti_hardware(parameters: dict, num_qubits: int) -> tuple[TrappedIonsModel, TIArchitecture | dict]:
    """Construct a trapped ion architecture as the abstract model and the concrete MQT IonShuttler hardware.

    parameters: `topology` ("linear" | "ring" | "grid"), `nr_zones`
      - linear: `zone_capacity`, optional `storage_slack` (1)
      - ring:   fixed 2 x 2 junctions
      - grid:   optional `grid_size` ([3, 3])
      - ring / grid: optional `segment_length` (~half filled), `seed` (0); fixed zone capacity 2, `nr_zones` 1-4
    """
    nr_zones = parameters["nr_zones"]

    if parameters["topology"] == "linear":
        # [storage][zone][storage]...[zone][storage] on one chain, one ion per site
        capacity = parameters["zone_capacity"]
        storage_size = -(-num_qubits // (nr_zones + 1)) + parameters.get("storage_slack", 1)
        zones = {f"iz{z}": list(range(storage_size + z * (capacity + storage_size), (z + 1) * (capacity + storage_size))) for z in range(nr_zones)}
        concrete_hardware: TIArchitecture | dict = TIArchitecture(num_sites=storage_size + nr_zones * (capacity + storage_size), processing_zones=zones)

    elif parameters["topology"] in ("ring", "grid"):
        # junction grid [m, n, v, h]: m x n junctions, v / h ions per vertical / horizontal trap segment; a ring is 2 x 2
        capacity = 2
        m, n = (2, 2) if parameters["topology"] == "ring" else parameters.get("grid_size", (3, 3))
        nr_segments = m * (n - 1) + n * (m - 1)
        segment_length = parameters.get("segment_length", -(-2 * num_qubits // nr_segments))
        concrete_hardware = {"arch": [m, n, segment_length, segment_length], "num_pzs": nr_zones, "seed": parameters.get("seed", 0)}

    else:
        raise ValueError(f"Unsupported trapped ion architecture {parameters['topology']!r} (linear, ring, grid).")

    abstract_model = TrappedIonsModel(interaction_zone_capacity=capacity, nr_interaction_sites=nr_zones, native_gates=TI_NATIVE_GATES)
    return abstract_model, concrete_hardware

def prepare_na_hardware(parameters: dict, num_qubits: int) -> tuple[NeutralAtomsModel, dict]:
    """Construct a zoned neutral atom architecture as the abstract model and the concrete hardware (zoned architecture spec, as JSON dict,
    read by MQT QMAP (zoned) and ZAC).

    parameters: `entanglement_zone_size` fixed n x m grid
    
    """
    ez_rows, ez_cols = parameters["entanglement_zone_size"]
    nr_ezs = parameters.get("nr_entanglement_zones", 1)
    if nr_ezs not in (1, 2):
        raise ValueError(f"Unsupported number of entanglement zones {nr_ezs} (1, 2).")
    storage_cols = max(4 * ez_cols, math.ceil(math.sqrt(2 * num_qubits)))
    storage_rows, storage_cols = parameters.get("storage_zone_size", (math.ceil(2 * num_qubits / storage_cols), storage_cols))

    # layout (y grows downwards, all coordinates positive): [EZ 1]  [storage]  [EZ 0] (compiler does not support readout zones)
    margin, gap = 4, 20
    ez_height = 10 * (ez_rows - 1)
    storage_y = margin + (ez_height + gap if nr_ezs == 2 else 0)
    ez_ys = [storage_y + 3 * (storage_rows - 1) + gap, margin][:nr_ezs]
    spec = {
        "name": "zoned",
        "operation_duration": {"rydberg_gate": 0.36, "single_qubit_gate": 52, "atom_transfer": 15},  # us
        "operation_fidelity": {"rydberg_gate": 0.995, "single_qubit_gate": 0.9997, "atom_transfer": 0.999},
        "qubit_spec": {"T": 1.5e6},
        "storage_zones": [{"zone_id": 0, "offset": [margin, storage_y], "dimension": [3 * storage_cols, 3 * storage_rows],
                           "slms": [{"id": 0, "site_separation": [3, 3], "r": storage_rows, "c": storage_cols, "location": [margin, storage_y]}]}],
        "entanglement_zones": [{"zone_id": z, "offset": [margin, y], "dimension": [12 * ez_cols, ez_height + 10],
                                "slms": [{"id": 2 * z + 1, "site_separation": [12, 10], "r": ez_rows, "c": ez_cols, "location": [margin, y]},
                                         {"id": 2 * z + 2, "site_separation": [12, 10], "r": ez_rows, "c": ez_cols, "location": [margin + 2, y]}]}
                               for z, y in enumerate(ez_ys)],
        "aods": [{"id": 0, "site_separation": 2, "r": parameters.get("aod_size", (100, 100))[0], "c": parameters.get("aod_size", (100, 100))[1]}],
        "rydberg_range": [[[margin - 2, y - 2], [margin + 12 * ez_cols, y + ez_height + 2]] for y in ez_ys],
    }
    concrete_hardware = spec

    readout_rows, readout_cols = parameters["readout_zone_size"]
    abstract_model = NeutralAtomsModel(entanglement_zone_capacity=nr_ezs * ez_rows * ez_cols, readout_zone_capacity=readout_rows * readout_cols,
                                       native_gates=NA_NATIVE_GATES)
    return abstract_model, concrete_hardware



def prepare_sc_hardware(parameters: dict, num_qubits: int) -> tuple[SuperconductingQubitsModel, SCArchitecture]:
    """Construct a superconducting qubit architecture as the abstract model and the concrete MQT QMAP hardware.

    parameters: `topology` ("square" | "hex" | "heavy_hex" | "octagonal")
    """
    # underlying (technically infinite) lattice
    side = math.ceil(math.sqrt(num_qubits)) + 3
    if parameters["topology"] == "square":
        lattice = nx.grid_2d_graph(side, side)
        nx.set_node_attributes(lattice, {node: node for node in lattice}, "pos")
    elif parameters["topology"] == "hex": 
        lattice = nx.hexagonal_lattice_graph(side, side)
    elif parameters["topology"] == "heavy_hex":
        hexagons = nx.hexagonal_lattice_graph(side, side)
        lattice = nx.Graph()
        lattice.add_nodes_from(hexagons.nodes(data=True))
        for u, v in hexagons.edges:
            (ux, uy), (vx, vy) = hexagons.nodes[u]["pos"], hexagons.nodes[v]["pos"]
            lattice.add_node((u, v), pos=((ux + vx) / 2, (uy + vy) / 2))
            lattice.add_edges_from([(u, (u, v)), ((u, v), v)])
    elif parameters["topology"] == "octagonal":
        lattice = nx.Graph()
        for row in range(side):
            for col in range(side):
                for i in range(8):
                    angle = (i + 0.5) * math.pi / 4
                    lattice.add_node((row, col, i), pos=(3 * col + math.cos(angle), 3 * row + math.sin(angle)))
                    lattice.add_edge((row, col, i), (row, col, (i + 1) % 8))
                if col > 0:
                    lattice.add_edges_from([((row, col - 1, 0), (row, col, 3)), ((row, col - 1, 7), (row, col, 4))])
                if row > 0:
                    lattice.add_edges_from([((row - 1, col, 1), (row, col, 6)), ((row - 1, col, 2), (row, col, 5))])
    else:
        raise ValueError(f"Unsupported superconducting architecture {parameters['topology']!r} (square, hex, heavy_hex, octagonal).")

    # build finite chip layout based on num_qubits and infinite lattice
    pos = nx.get_node_attributes(lattice, "pos")
    centroid = [sum(p[k] for p in pos.values()) / len(pos) for k in (0, 1)]
    chip, best_key = None, None
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
            chip, best_key = graph, key

    concrete_hardware = SCArchitecture(num_qubits, {edge for u, v in chip.edges for edge in [(u, v), (v, u)]})  # bidirectional
    abstract_model = SuperconductingQubitsModel(number_of_qubits=num_qubits, topology=dict(Counter(d for _, d in chip.degree)),
                                                gate_durations={"SN": 1, "MS": 1, "EN": 1}, native_gates=SC_NATIVE_GATES)
    return abstract_model, concrete_hardware
