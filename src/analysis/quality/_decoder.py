"""Decoder helpers that are needed for the quality analysis of the QEC circuit."""

import numpy as np

from ldpc import BpOsdDecoder
import stim
from typing import Protocol

from src.models.hardware import HardwareModel

class Decoder(Protocol):
    def decode(self, syndrome: np.ndarray) -> stim.PauliString:
        ...


class BeliefPropOrdStat(Decoder): # noise-informed decoder
    def __init__(self, hardware: HardwareModel, stabilizers: list[stim.PauliString]) -> None:
        n = len(stabilizers[0])
        
        x =np.array([[s[q] in (1, 2) for q in range(n)]for s in stabilizers], dtype=np.uint8)
        z = np.array([[s[q] in (2, 3) for q in range(n)]for s in stabilizers], dtype=np.uint8)

        s = np.hstack([x, z])

        px, pz = hardware.noise_model.get_aggregated_probabilities()
        px, pz = max(px, 1e-6), max(pz, 1e-6)

        self.n = n
        self.decoder = BpOsdDecoder(s, error_channel=[pz] * n + [px] * n, bp_method="product_sum", max_iter=30, osd_method="osd_cs", osd_order=2, input_vector_type="syndrome")

    def decode(self, syndrome: np.ndarray) -> stim.PauliString: # return recovery operation
        error = np.asarray(self.decoder.decode(syndrome), dtype=np.uint8)
        ez, ex = error[:self.n], error[self.n:]
        
        return stim.PauliString("".join("IZXY"[2 * int(x) + int(z)] for x, z in zip(ex, ez, strict=True)))
        
          