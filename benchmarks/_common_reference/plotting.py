"""Values for consistent visualizations."""

import matplotlib.pyplot as plt

DPI = 200
SURFACE, TEXT, TEXT_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, SERIES, REFERENCE = "#e1e0d9", "#2a78d6", "#e34948"

def save(fig, file: str) -> None:
    fig.savefig(file, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)