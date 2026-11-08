"""Constructs specific hardware."""

from src.models.hardware import ( TrappedIonsModel, NeutralAtomsModel, SuperconductingQubitsModel)
from mqt.ionshuttler.linear import Architecture as TIArchitecture
from mqt.qmap.sc import Architecture as SCArchitecture
from mqt.qmap.na.zoned import ZonedNeutralAtomArchitecture

def prepare_ti_hardware(parameters: dict, num_qubits: int) -> tuple[TrappedIonsModel, TIArchitecture | dict]:
    """Construct a named trapped ion architecture: the abstract model and the concrete MQT IonShuttler hardware.

    parameters["topology"]:
      - "linear" (Quantinuum H1 style): one linear trap, `nr_zones` gate zones of `zone_capacity` sites inside the chain, with
        storage in between (`storage_slack` extra free sites per storage segment, default 1). Concrete: `linear` Architecture
        (one ion per site).
      - "ring" (Quantinuum Helios style): a storage loop of four trap segments (`segment_length` ions each) joined by junctions,
        with `nr_zones` (1-4) gate zones attached outside via junction legs. Concrete: `multi_shuttler` config.
      - "grid" (2D QCCD, e.g. Quantinuum's planned grid devices): `grid_size` (m, n) junctions connected by trap segments, with
        `nr_zones` (1-4) gate zones attached at the sides. Concrete: `multi_shuttler` config.

    Abstract: nr_interaction_sites = number of gate zones; interaction_zone_capacity = sites per zone ("linear"), or 2 for
    "ring" / "grid" (a multi_shuttler gate zone holds at most 2 ions, hardcoded). The topology itself is not modelled abstractly.
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
        segment_length = parameters.get("segment_length", -(-2 * num_qubits // nr_segments))  # default: ~half filled
        concrete_hardware = {"arch": [m, n, segment_length, segment_length], "num_pzs": nr_zones, "seed": parameters.get("seed", 0)}

    else:
        raise ValueError(f"Unknown trapped ion architecture {parameters['topology']!r} (linear, ring, grid).")

    abstract_model = TrappedIonsModel(interaction_zone_capacity=capacity, nr_interaction_sites=nr_zones)
    return abstract_model, concrete_hardware

def prepare_na_hardware(parameters: dict) -> tuple[NeutralAtomsModel, ZonedNeutralAtomArchitecture]:
    abstract_model = NeutralAtomsModel(entanglement_zone_capacity=parameters["entanglement_zone_capacity"], readout_zone_capacity=parameters["readout_zone_capacity"])
    return abstract_model, None



def prepare_sc_hardware(parameters: dict) -> tuple[SuperconductingQubitsModel, SCArchitecture]:
    abstract_model = SuperconductingQubitsModel(number_of_qubits=parameters["number_of_qubits"], gate_durations=parameters["gate_durations"], topology=parameters["topology"])
    return abstract_model, None
