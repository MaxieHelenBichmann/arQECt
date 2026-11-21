"""Decoder helpers that are needed for the quality analysis of the QEC circuit."""

from itertools import combinations, product

import numpy as np

from ldpc import BpOsdDecoder
import stim
from typing import Protocol

from src.models.hardware import HardwareModel

class Decoder(Protocol):
    def decode(self, syndrome: np.ndarray) -> stim.PauliString:
        ...


class BeliefPropOrdStat(Decoder): # minimum-weight lookup for low-weight syndromes, BP-OSD otherwise
    def __init__(self, hardware: HardwareModel, stabilizers: list[stim.PauliString], css_max_weight: int = 3, joint_max_weight: int = 2) -> None:
        n = len(stabilizers[0])

        x =np.array([[s[q] in (1, 2) for q in range(n)]for s in stabilizers], dtype=np.uint8)
        z = np.array([[s[q] in (2, 3) for q in range(n)]for s in stabilizers], dtype=np.uint8)

        px, pz = hardware.noise_model.get_aggregated_probabilities()
        px, pz = max(px, 1e-6), max(pz, 1e-6)

        self.n = n
        self.is_x, self.is_z = ~z.any(axis=1), ~x.any(axis=1) # X-type / Z-type stabilizer rows
        self.css = bool(np.all(self.is_x | self.is_z))

        if self.css:
            hz, hx = z[self.is_z], x[self.is_x]
            self.lookup_x = self._min_weight_lookup(x[self.is_z], z[self.is_z], "X", css_max_weight)
            self.lookup_z = self._min_weight_lookup(x[self.is_x], z[self.is_x], "Z", css_max_weight)
            self.decoder_x = BpOsdDecoder(hz, error_channel=[px] * n, bp_method="product_sum", max_iter=30, osd_method="osd_cs", osd_order=2, input_vector_type="syndrome")
            self.decoder_z = BpOsdDecoder(hx, error_channel=[pz] * n, bp_method="product_sum", max_iter=30, osd_method="osd_cs", osd_order=2, input_vector_type="syndrome")
        else:
            s = np.hstack([x, z])
            self.lookup = self._min_weight_lookup(x, z, "XYZ", joint_max_weight)
            self.decoder = BpOsdDecoder(s, error_channel=[pz] * n + [px] * n, bp_method="product_sum", max_iter=30, osd_method="osd_cs", osd_order=2, input_vector_type="syndrome")

    def decode(self, syndrome: np.ndarray) -> stim.PauliString: # return recovery operation
        syndrome = np.asarray(syndrome, dtype=np.uint8)

        if self.css:
            sx, sz = syndrome[self.is_z], syndrome[self.is_x]
            ex = self.lookup_x[sx.tobytes()][0] if sx.tobytes() in self.lookup_x else np.asarray(self.decoder_x.decode(sx), dtype=np.uint8)
            ez = self.lookup_z[sz.tobytes()][1] if sz.tobytes() in self.lookup_z else np.asarray(self.decoder_z.decode(sz), dtype=np.uint8)
        elif syndrome.tobytes() in self.lookup:
            ex, ez = self.lookup[syndrome.tobytes()]
        else:
            error = np.asarray(self.decoder.decode(syndrome), dtype=np.uint8)
            ez, ex = error[:self.n], error[self.n:]

        return stim.PauliString("".join("IZXY"[2 * int(x) + int(z)] for x, z in zip(ex, ez, strict=True)))

    def _min_weight_lookup(self, x: np.ndarray, z: np.ndarray, paulis: str, max_weight: int) -> dict[bytes, tuple[np.ndarray, np.ndarray]]:
        """syndrome -> minimum-weight error (ex, ez)"""
        lookup = {}
        for weight in range(max_weight + 1):
            for qubits in combinations(range(self.n), weight): # erroneous qubits
                for errors in product(paulis, repeat=weight): # error type per erroneous qubit
                    ex, ez = np.zeros(self.n, dtype=np.uint8), np.zeros(self.n, dtype=np.uint8)
                    for qubit, pauli in zip(qubits, errors):
                        match pauli:
                            case "X":
                                ex[qubit] = 1
                            case "Z":
                                ez[qubit] = 1
                            case "Y":
                                ex[qubit], ez[qubit] = 1, 1
                    syndrome = (z @ ex + x @ ez) % 2
                    lookup.setdefault(syndrome.astype(np.uint8).tobytes(), (ex, ez))
        return lookup
