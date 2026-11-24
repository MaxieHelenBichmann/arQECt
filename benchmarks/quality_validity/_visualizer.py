"""Plot the agreement between the Monte-Carlo-Sampling-based (MC) and the Detector-Error-Model-based (DEM) quality metrics, one directory per circuit function.

Per circuit function and metric: scatter of MC vs DEM per run (circuit on one hardware configuration) with the MC 95% CI and the DEM truncation bounds.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, PaddedBox, TextArea, VPacker

from benchmarks._common_reference.plotting import SURFACE, TEXT, TEXT_SECONDARY, MUTED, GRID, SERIES, REFERENCE, save
from benchmarks.quality_validity._summarizer import load

BASELINE, CI_BAR = "#c3c2b7", "#a9a8a0"
OUTSIDE, ZERO = "#eb8a1e", "#9cc3f0" # runs with DEM outside the MC CI (red is the target: MC = DEM), runs without sampled MC failure
LABELS = {
    "StatePrep": "State preparation", "Encoding": "Encoding", "Syndrome": "Syndrome extraction",
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

        out_dir = results_dir / "plots" / function.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"{LABELS[function]}: {LABELS[metric]}, Monte Carlo (MC) vs detector error model (DEM)"
        subtitle = (f"{len(rows)} runs (circuits × hardware configurations),  MC: {int(float(rows[0]['max_shots'])):,} shots,  "
                    f"DEM: fault sets with up to {rows[0]['max_weight']} faults")
        plot_scatter(s, mc, mc_lo, mc_hi, dem, dem_lo, dem_hi, metric, title, subtitle, out_dir / f"{metric}__scatter.png")
    print(f"Saved plots to {results_dir / 'plots'}")


def plot_scatter(s: dict, mc, mc_lo, mc_hi, dem, dem_lo, dem_hi, metric: str, title: str, subtitle: str, file: Path) -> None:
    """MC vs DEM per run: on the diagonal if both agree, MC 95% CI vertical, DEM bounds horizontal."""
    fig, ax = plt.subplots(figsize=(7.6, 6.2), facecolor=SURFACE)
    inside = (dem >= mc_lo - 1e-12) & (dem <= mc_hi + 1e-12) # tolerance: float rounding of the Wilson bounds at rates 0 and 1
    sampled = mc > 0 # without sampled failure, MC only gives an upper bound (drawn at the upper end of its CI)
    y = np.where(sampled, mc, mc_hi)

    values = np.concatenate([dem_lo, dem_hi, mc_hi, mc[sampled]])
    limits = (values[values > 0].min() / 2, values.max() * 2)
    ax.plot(limits, limits, color=REFERENCE, linewidth=3, zorder=2, label="exact agreement (MC = DEM)")

    # intervals in the background (behind the diagonal and the markers)
    ax.errorbar(dem, y, yerr=np.clip([np.where(sampled, mc - mc_lo, 0), mc_hi - y], 0, None), fmt="none", ecolor=CI_BAR, alpha=0.4, linewidth=1, zorder=1) # clip float rounding
    ax.errorbar(dem, y, xerr=np.clip([dem - dem_lo, dem_hi - dem], 0, None), fmt="none", ecolor=CI_BAR, alpha=0.4, linewidth=1.5, zorder=1)
    for mask, color in [(inside & sampled, SERIES), (~inside & sampled, OUTSIDE), (~sampled, ZERO)]:
        ax.scatter(dem[mask], y[mask], color=color, s=30, alpha=0.8, edgecolors="none", zorder=5)

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()

    handles = [Line2D([], [], color=REFERENCE, linewidth=3), Line2D([], [], color=CI_BAR, linewidth=1), Line2D([], [], color=CI_BAR, linewidth=1.5),
               Line2D([], [], marker="o", color=SERIES, linestyle="none"), Line2D([], [], marker="o", color=OUTSIDE, linestyle="none")]
    labels = ["exact agreement (MC = DEM)", "MC 95% confidence interval (Wilson, vertical)", "DEM bounds (truncated fault sets, horizontal)",
              "run: DEM inside the MC 95% CI", "run: DEM outside the MC 95% CI"]
    if not sampled.all(): # MC estimate 0 not drawable on a log axis
        handles.append(Line2D([], [], marker="o", color=ZERO, linestyle="none"))
        labels.append("run: MC sampled 0 failures --> drawn at its upper CI end")
    width = (dem_hi - dem_lo)[dem > 0] / dem[dem > 0]
    lines = [f"DEM inside MC 95% CI     {float(s['dem_in_mc_ci']):>8.1%}   (expected ≈ 95%)",
             f"MC CI and DEM overlap     {float(s['intervals_overlap']):>8.1%}",
             f"median |MC − DEM| / DEM   {float(s['median_rel_deviation']):>8.2%}",
             f"DEM bounds width / DEM    {np.median(width):>8.2g}   (median),  {np.max(width):.2g} (max)"]
    style(fig, ax, title, subtitle, f"DEM {LABELS[metric]}", f"MC {LABELS[metric]}", handles, labels, "Agreement: are the runs on the diagonal?", lines)
    save(fig, str(file))


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
