"""Plot the agreement of the Detector-Error-Model-based (DEM) quality metrics with the Monte-Carlo-Sampling-based (MC) ground truth, one directory per circuit function.

Per circuit function and metric: scatter of DEM vs MC per run (circuit on one hardware configuration) with the DEM truncation bounds, the MC 95% CI and the speedup of DEM over MC.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, PaddedBox, TextArea, VPacker

from benchmarks._common_reference.plotting import SURFACE, TEXT, TEXT_SECONDARY, MUTED, GRID, SERIES, REFERENCE
from benchmarks.robustness_validity._summarizer import load

BASELINE, CI_BAR = "#c3c2b7", "#8fb6e3"
OUTSIDE = "#eb8a1e" # runs with DEM outside the MC CI (red is the target: MC = DEM)
OUTSIDE_BAR = "#f2b56f"
WEIGHTS = LinearSegmentedColormap.from_list("weights", ["#b5d3f5", SERIES, "#0d366b"]) # lightness: truncation weight the DEM needed (light: few faults)
FT_WEIGHTS = LinearSegmentedColormap.from_list("ft_weights", ["#b9e4c9", "#2e9e5b", "#0f4d2a"]) # same, for fault-tolerant circuits
FT_BAR = "#8cc29f"
WEIGHT_RANGE = (1, 6) # same color for the same weight in every plot
LABELS = {
    "StatePrep": "State preparation", "Encoding": "Encoding (noisy input)", "EncodingCleanInput": "Encoding (clean input)", "Syndrome": "Syndrome extraction",
    "logical_error_rate": "logical error rate", "rejection_rate": "rejection rate", "syndrome_error_rate": "syndrome error rate",
}


def visualize(results_dir: Path) -> None:
    runs = load(results_dir)
    with (results_dir / "summary.csv").open(newline="") as f:
        summary = list(csv.DictReader(f))

    for s in summary:
        function, metric = s["function"], s["metric"]
        rows = [r for r in runs if r["function"] == function and r[f"mc_{metric}"] != "" and r[f"dem_{metric}"] != ""] # LER undefined if no shot accepted
        column = lambda key: np.array([float(r[key]) for r in rows])
        mc, mc_lo, mc_hi = column(f"mc_{metric}"), column(f"mc_{metric}_ci_lower"), column(f"mc_{metric}_ci_upper")
        dem, dem_lo, dem_hi = column(f"dem_{metric}"), column(f"dem_{metric}_lower"), column(f"dem_{metric}_upper")
        weight = column("max_weight").astype(int)
        ft = np.array([r["ft"] == "True" for r in rows])

        out_dir = results_dir / "plots" / function.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"{LABELS[function]}: {LABELS[metric]}, detector error model (DEM) vs Monte Carlo (MC)"
        subtitle = (f"{len(rows)} runs (circuits × hardware configurations),  MC: {int(float(rows[0]['max_shots'])):,} shots,  "
                    f"DEM: fault sets with up to {weight.min()}–{weight.max()} faults")
        plot_scatter(s, mc, mc_lo, mc_hi, dem, dem_lo, dem_hi, weight, ft, metric, title, subtitle, out_dir / f"{metric}__scatter.png")
    print(f"Saved plots to {results_dir / 'plots'}")


def plot_scatter(s: dict, mc, mc_lo, mc_hi, dem, dem_lo, dem_hi, weight, ft, metric: str, title: str, subtitle: str, file: Path) -> None:
    """DEM vs MC (ground truth, x) per run: on the diagonal if both agree, DEM bounds vertical, MC 95% CI horizontal. Fault-tolerant circuits green, others blue."""
    fig, ax = plt.subplots(figsize=(7.6, 6.2), facecolor=SURFACE)
    inside = (dem >= mc_lo - 1e-12) & (dem <= mc_hi + 1e-12) # tolerance: float rounding of the Wilson bounds at rates 0 and 1
    sampled = mc > 0 # without sampled failure, MC only gives an upper bound (drawn at the upper end of its CI)
    x = np.where(sampled, mc, mc_hi)

    values = np.concatenate([dem_lo, dem_hi, mc_hi, mc[sampled]])
    limits = (values[values > 0].min() / 2, values.max() * 2)
    ax.plot(limits, limits, color=REFERENCE, linewidth=3, zorder=2, label="exact agreement (DEM = MC)")

    # intervals in the background (behind the diagonal and the markers)
    outside = ~inside & sampled
    # group color for every interval, runs outside the MC CI additionally with an orange dotted line on top
    for group, bar_color, alpha, dashes in [(~ft, CI_BAR, 0.4, "solid"), (ft, FT_BAR, 0.4, "solid"), (outside, OUTSIDE_BAR, 0.9, (0, (1.5, 1.5)))]:
        for bars in [ax.errorbar(x[group], dem[group], xerr=np.clip([np.where(sampled, mc - mc_lo, 0)[group], (mc_hi - x)[group]], 0, None), fmt="none", ecolor=bar_color, alpha=alpha, linewidth=1, capsize=1.5, capthick=0.8, zorder=1), # clip float rounding
                     ax.errorbar(x[group], dem[group], yerr=np.clip([(dem - dem_lo)[group], (dem_hi - dem)[group]], 0, None), fmt="none", ecolor=bar_color, alpha=alpha, linewidth=1.5, capsize=1.5, capthick=0.8, zorder=1)]:
            for line in bars[2]:
                line.set_linestyle(dashes)
    shade = lambda w, cmap: cmap((np.clip(w, *WEIGHT_RANGE) - WEIGHT_RANGE[0]) / (WEIGHT_RANGE[1] - WEIGHT_RANGE[0]))
    for group, cmap in [(~ft, WEIGHTS), (ft, FT_WEIGHTS)]:
        ax.scatter(x[group & inside & sampled], dem[group & inside & sampled], color=shade(weight[group & inside & sampled], cmap), s=30, alpha=0.55, edgecolors="none", zorder=5)
        ax.scatter(x[group & outside], dem[group & outside], color=shade(weight[group & outside], cmap), s=30, alpha=0.65, edgecolors=OUTSIDE, linewidths=1.3, zorder=6)
        ax.scatter(x[group & ~sampled], dem[group & ~sampled], facecolors="none", edgecolors=shade(weight[group & ~sampled], cmap), s=30, linewidths=1.2, alpha=0.65, zorder=5)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()

    handles = [Line2D([], [], color=REFERENCE, linewidth=3),
               Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=OUTSIDE, markeredgewidth=1.3, color=OUTSIDE_BAR, linestyle=(0, (1.5, 1.5)))]
    labels = ["exact agreement (DEM = MC)", "run: DEM LER outside the MC 95% CI"]
    if not sampled.all(): # MC estimate 0 not drawable on a log axis
        handles.append(Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=MUTED, linestyle="none"))
        labels.append("run: MC sampled 0 failures --> drawn at its upper CI end")
    handles.append(Line2D([], [], marker="o", color=SERIES, linestyle="none"))
    labels.append("run of a non-fault-tolerant circuit" if ft.any() else "run of a circuit")
    if ft.any():
        handles.append(Line2D([], [], marker="o", color=FT_WEIGHTS(0.5), linestyle="none"))
        labels.append("run of a fault-tolerant circuit")

    # discrete scale: one color per truncation weight the DEM needed (one scale per circuit group)
    levels = np.arange(WEIGHT_RANGE[0], WEIGHT_RANGE[1] + 1)
    for cmap, group_label in ([(WEIGHTS, "non-FT"), (FT_WEIGHTS, "FT")] if ft.any() else [(WEIGHTS, "")]):
        bar = fig.colorbar(ScalarMappable(BoundaryNorm(np.append(levels, levels[-1] + 1) - 0.5, len(levels)), ListedColormap(shade(levels, cmap))),
                           ax=ax, ticks=levels, fraction=0.035, pad=0.03)
        bar.ax.set_title(group_label, color=TEXT_SECONDARY, fontsize=8)
        if cmap is WEIGHTS: # outermost scale
            bar.set_label("DEM: fault sets enumerated up to ... faults", color=TEXT_SECONDARY, fontsize=9)
        bar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9, length=0)
        bar.outline.set_edgecolor(BASELINE)
    width = (dem_hi - dem_lo)[dem > 0] / dem[dem > 0]
    lines = [f"DEM inside MC 95% CI     {float(s['dem_in_mc_ci']):>8.1%}   (expected ≈ 95%)",
             f"MC CI and DEM overlap     {float(s['intervals_overlap']):>8.1%}",
             f"median |DEM − MC| / MC    {float(s['median_rel_deviation']):>8.2%}",
             f"DEM bounds width / DEM    {np.median(width):>8.2g}   (median),  {np.max(width):.2g} (max)"]
    if s["median_speedup"] != "": # older raw results have no times
        lines.append(f"speedup DEM vs MC         {float(s['median_speedup']):>7.3g}×   (median),  {float(s['min_speedup']):.3g}× – {float(s['max_speedup']):.3g}× (range, MC time / DEM time)")
    style(fig, ax, title, subtitle, f"MC {LABELS[metric]} (+ 95% confidence interval)", f"DEM {LABELS[metric]} (+ bounds due to truncated fault sets)", handles, labels, "Agreement: how close are the circuits to y = x?", lines)
    fig.savefig(file, dpi=400, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25) # higher dpi than the shared save()
    plt.close(fig)


def style(fig, ax, title: str, subtitle: str, x_label: str, y_label: str, handles, labels, header: str, lines: list[str]) -> None:
    """Style the axes, then put the legend and the score box (bold header) below it (like the model accuracy plots)."""
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)
    ax.set_title(title, loc="center", color=TEXT, fontsize=12, fontweight="bold", pad=22)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, va="bottom", ha="center", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=10)
    fig.tight_layout()

    # everything below the axes is placed from the measured bottom of the previous block (saved with bbox_inches="tight")
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    left, bottom = to_figure.transform(ax.get_tightbbox(renderer).p0)
    legend = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(left, bottom - 0.01), frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY,
                        handlelength=2.0, ncol=2, columnspacing=1.8, borderaxespad=0)
    bottom = to_figure.transform(legend.get_window_extent(renderer).p0)[1] - 0.025
    box = PaddedBox(TextArea("\n".join(lines), textprops=dict(color=TEXT_SECONDARY, fontsize=7.5, family="monospace")), pad=6, draw_frame=True,
                    patch_attrs=dict(facecolor=SURFACE, edgecolor=SERIES, linewidth=1))
    blocks = [TextArea(header, textprops=dict(color=TEXT, fontsize=9, fontweight="bold")), box]
    fig.add_artist(AnchoredOffsetbox("upper left", child=VPacker(children=blocks, sep=8, pad=0, align="left"), bbox_to_anchor=(left, bottom),
                                     bbox_transform=fig.transFigure, frameon=False, pad=0, borderpad=0))
