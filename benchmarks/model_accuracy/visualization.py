"""Extract and aggregate relevant metrics comparing the abstract execution and concrete compilation, and visualize them."""

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

# chart tokens (light mode): surface, text, recessive grid / axes
_SURFACE, _TEXT, _TEXT_SECONDARY, _MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRID, _BASELINE = "#e1e0d9", "#c3c2b7"
# qubit number of a circuit: one blue ramp, light -> dark, fixed over the range of all benchmark circuits (so the same
# color means the same qubit number in every plot); starts at the lightest step still visible as a dot on the surface
_QUBIT_CMAP = LinearSegmentedColormap.from_list("qubits", ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                                                           "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
_QUBIT_RANGE = (4, 20)


def visualize(platform: list[str], results_dir: Path, max_qubits: int | None = None) -> None:
    if "TI" in platform:
        visualize_ti_aggregated(results_dir, max_qubits)
    if "NA" in platform:
        pass
    if "SC" in platform:
        pass
    
def visualize_ti_aggregated(results_dir: Path, max_qubits: int | None = None) -> None:
    """Scatter abstract vs compiled metrics, aggregated per circuit, one figure per TI hardware config.

    Reads every raw file in `<results_dir>/raw/ti` (one config, one circuit each; only circuits with at most `max_qubits`
    qubits if given), sums its counts per (source, metric), colors each point by the circuit's qubit number, and
    writes one scatter plot per comparison, `<results_dir>/plots/ti_<config>__<comparison>.png` (one point per circuit),
    plus `ti_<config>__aggregated.csv` with the plotted totals.
    """
    # aggregate each raw file into totals per (source, metric), grouped by hardware config
    runs: defaultdict[str, list[dict]] = defaultdict(list)  # config -> one entry per circuit
    for file in sorted((results_dir / "raw" / "ti").glob("*.csv")):
        with open(file, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or rows[0]["platform"] != "TI":
            continue
        if max_qubits is not None and int(rows[0]["num_qubits"]) > max_qubits:
            continue
        totals: Counter[tuple[str, str]] = Counter()
        for row in rows:
            if row["count"]:  # event / anchored rows (detail rows only carry a JSON `value`)
                totals[(row["source"], row["metric"])] += int(row["count"])
        runs[rows[0]["config"]].append({
            "circuit": rows[0]["circuit"],
            "num_qubits": int(rows[0]["num_qubits"]),
            "abstract_iz_entry": totals[("abstract", "interaction_zone_entry")],
            "abstract_iz_eviction": totals[("abstract", "interaction_zone_eviction")],
            "compiled_iz_entry": totals[("compiled", "interaction_zone_entry")],
            "compiled_iz_eviction": totals[("compiled", "interaction_zone_eviction")],
            "compiled_shuttles": totals[("compiled", "shuttles")],
            "compiled_swaps": totals[("compiled", "swaps")],  # counted per ion, i.e. ion moves (see compilation.py)
        })

    # (file name, title, x label, y label, x, y, draw y = x reference)
    comparisons = [
        ("iz_entries", "IZ entries", "abstract IZ entries", "compiled IZ entries",
         lambda r: r["abstract_iz_entry"], lambda r: r["compiled_iz_entry"], True),
        ("iz_evictions", "IZ evictions", "abstract IZ evictions", "compiled IZ evictions",
         lambda r: r["abstract_iz_eviction"], lambda r: r["compiled_iz_eviction"], True),
        ("iz_changes", "IZ changes (entries + evictions)", "abstract IZ changes", "compiled IZ changes",
         lambda r: r["abstract_iz_entry"] + r["abstract_iz_eviction"],
         lambda r: r["compiled_iz_entry"] + r["compiled_iz_eviction"], True),
        ("iz_changes_vs_ion_moves", "IZ changes vs ion moves", "abstract IZ changes", "compiled shuttles + swaps",
         lambda r: r["abstract_iz_entry"] + r["abstract_iz_eviction"],
         lambda r: r["compiled_shuttles"] + r["compiled_swaps"], False),
    ]

    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for config, circuits in sorted(runs.items()):
        slug = re.sub(r"[^a-z0-9]+", "_", config.lower()).strip("_")
        circuits.sort(key=lambda r: r["num_qubits"])  # larger circuits drawn on top
        qubits = [r["num_qubits"] for r in circuits]

        for name, title, x_label, y_label, fx, fy, identity in comparisons:
            xs, ys = [fx(r) for r in circuits], [fy(r) for r in circuits]
            fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=_SURFACE)
            ax.set_facecolor(_SURFACE)
            ax.grid(color=_GRID, linewidth=0.8, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(_BASELINE)
            ax.tick_params(colors=_MUTED, labelcolor=_TEXT_SECONDARY, labelsize=9)

            upper = max([*xs, *ys, 1]) * 1.08
            if identity:  # same quantity on both axes: equal scales, y = x marks a perfect match
                ax.plot([0, upper], [0, upper], color=_BASELINE, linewidth=1, zorder=1)
                ax.set_xlim(0, upper)
                ax.set_ylim(0, upper)
            else:
                ax.set_xlim(0, max([*xs, 1]) * 1.08)
                ax.set_ylim(0, max([*ys, 1]) * 1.08)
            points = ax.scatter(xs, ys, s=56, c=qubits, cmap=_QUBIT_CMAP, vmin=_QUBIT_RANGE[0], vmax=_QUBIT_RANGE[1],
                                edgecolors=_SURFACE, linewidths=1.5, zorder=3)  # surface ring
            colorbar = fig.colorbar(points, ax=ax, ticks=range(_QUBIT_RANGE[0], _QUBIT_RANGE[1] + 1, 4), fraction=0.05, pad=0.03)
            colorbar.set_label("qubits in circuit", color=_TEXT_SECONDARY, fontsize=10)
            colorbar.outline.set_visible(False)
            colorbar.ax.tick_params(colors=_MUTED, labelcolor=_TEXT_SECONDARY, labelsize=9)

            ax.set_title(f"{config}: {title}", loc="left", color=_TEXT, fontsize=12, fontweight="bold")
            ax.set_xlabel(x_label, color=_TEXT_SECONDARY, fontsize=10)
            ax.set_ylabel(y_label, color=_TEXT_SECONDARY, fontsize=10)
            ax.text(0.02, 0.97, _correlation_label(xs, ys), transform=ax.transAxes, va="top", ha="left",
                    color=_TEXT_SECONDARY, fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / f"ti_{slug}__{name}.png", dpi=200, facecolor=_SURFACE)
            plt.close(fig)

        # table view of the plotted totals
        with open(out_dir / f"ti_{slug}__aggregated.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(circuits[0]))
            writer.writeheader()
            writer.writerows(sorted(circuits, key=lambda r: r["circuit"]))


def _correlation_label(xs: list[int], ys: list[int]) -> str:
    """n, Pearson r and Spearman rho (n/a for fewer than 3 points or a constant axis)."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return f"n = {len(xs)}   r = n/a   ρ = n/a"
    r, rho = pearsonr(xs, ys).statistic, spearmanr(xs, ys).statistic
    return f"n = {len(xs)}   r = {r:.2f}   ρ = {rho:.2f}" if not (math.isnan(r) or math.isnan(rho)) else f"n = {len(xs)}"
