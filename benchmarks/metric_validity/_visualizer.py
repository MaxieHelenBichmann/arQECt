"""Plot the summaries of the concordance (scorecard) and proportionality of the static metrics per platform / compiler."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from benchmarks._common_reference.plotting import SURFACE, TEXT, TEXT_SECONDARY, MUTED, GRID, SERIES, REFERENCE, save
from benchmarks.metric_validity._summarizer import HYPOTHESES

COLUMNS = [("TI", "IonShuttler"), ("NA", "QMAP"), ("NA", "Bloqade"), ("SC", "QMAP"), ("SC", "SABRE")]
DESIGNS = {"comparable_task": "comparable-task pairs (same QEC code and task)",
           "matched_workload": "matched-workload pairs (equal qubits, 2q gates within 1.25x)"}
DIVERGING = LinearSegmentedColormap.from_list("diverging", [REFERENCE, "#f6d3cf", "#f7f6f2", "#cfe0f5", SERIES])
UNAVAILABLE = "#efeeea"


# ------------------------------------------------------------------------------------------------------------------------------------------------

# short display names of the metrics in the plots
LABELS = {
    "critical_path_depth": "critical path depth",
    "2q_depth": "2q depth",
    "critical_path_duration": "critical path duration",
    "parallelism_saturation": "parallelism saturation",
    "capacity_constrained_depth_bound": "capacity depth bound",
    "min_idle_steps": "min. idle steps",
    "min_scheduling_window_congestion": "min. window congestion",
    "max_scheduling_window_congestion": "max. window congestion",
    "average_asap_qubit_reuse": "ASAP qubit reuse",
    "average_alap_qubit_reuse": "ALAP qubit reuse",
    "average_asap_qubit_interaction": "ASAP qubit interaction",
    "nr_gates": "nr. gates",
    "share_2q_gates": "share 2q gates",
    "fraction_of_partner_capacity_exceeding_qubits": "capacity-exceeding qubits",
    "weighted_fraction_of_partner_capacity_exceeding_qubits": "capacity-exceeding qubits (weighted)",
    "fraction_of_effective_partner_capacity_exceeding_qubits": "eff. capacity-exceeding qubits",
    "max_partner_capacity_ratio": "max. partner capacity ratio",
    "average_interaction_partner_diversity": "partner diversity",
    "average_interaction_entropy": "interaction entropy",
    "average_effective_partner_number": "eff. partner number",
    "statically_servable_interaction_fraction": "statically servable interactions",
    "capacity_constrained_qubit_partition": "capacity qubit partition",
    "non_bipartite_interaction_fraction": "non-bipartite interactions",
    "average_interaction_reuse_distance": "interaction reuse distance",
    "capacity_constrained_reuse": "capacity reuse",
    "partner_switches_per_qubit": "partner switches / qubit",
    "capacity_constrained_partner_switches_per_qubit": "partner switches / qubit (capacity)",
    "_activity_imbalance": "activity imbalance",
    "N_2q": "nr. 2q gates",
}

# ------------------------------------------------------------------------------------------------------------------------------------------------

def visualize(results_dir: Path) -> None:
    with (results_dir / "summary.csv").open(newline="") as f:
        summary = [row for row in csv.DictReader(f) if row["scope"] == "platform"]
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    costs = ["Runtime costs (aggregated per compiled circuit, not normalized):",
             "   - routing  = qubit moves (SC swaps, TI shuttles, NA atom transports)",
             "   - schedule = compiled schedule length",
             "   - idle     = time qubits wait between their gates (in compiler timesteps)"]
    # per plot: value, its neutral value and color range, file name, subtitle, color bar label, notes
    plots = [("C", 0.5, Normalize(0.25, 0.75), "scorecard",
              "Concordance: Comparing two circuits, does the static metric predict the right order of their actual runtime costs?",
              "weighted share of circuit pairs whose predicted-worse circuit actually costs more (0.5 = random)",
              [costs, ["bold: 95% CI excludes 0.5", "hatched: fewer than 3 code families to compare"]]),
             ("P", 0.0, Normalize(-1, 1), "proportionality",
              "Proportionality: Comparing two circuits, does a larger difference in the static metric correspond to a larger difference in runtime cost?",
              "weighted Spearman correlation between the metric difference of a pair and its signed cost difference " r"$\dfrac{y_{\mathrm{worse}} - y_{\mathrm{better}}}{y_{\mathrm{worse}} + y_{\mathrm{better}}}$" " (0 = no relation)",
              [costs, ["metric difference = distance of the two circuits in the metric's ranking",
                       "signed cost difference > 0 if the predicted-worse circuit costs more",
                       "hatched: fewer than 3 circuits to compare"]])]

    for design, title in DESIGNS.items():
        cells = {(r["metric"], r["platform"], r["compiler"]): r for r in summary if r["design"] in (design, "")}
        for value, neutral, norm, name, subtitle, label, notes in plots:
            fig, ax = draw_rows(len(COLUMNS))
            for x, (platform, compiler) in enumerate(COLUMNS):
                ax.text(x + 0.5, -0.3, f"{platform}\n{compiler}", ha="center", va="bottom", color=TEXT, fontsize=8.5, fontweight="bold")
                for y, metric in rows():
                    row = cells.get((metric, platform, compiler))
                    if row is None or row["status"] != "defined":
                        cell(ax, x, y, 1, UNAVAILABLE, row["status"] if row else "no runs", MUTED, 6.5)
                        continue
                    estimate, low, high, sparse = float(row[value]), float(row[f"{value}_lo"]), float(row[f"{value}_hi"]), row["sparse"] == "True"
                    if np.isnan(estimate):
                        cell(ax, x, y, 1, UNAVAILABLE, "–", MUTED, 6.5)
                        continue
                    significant = low > neutral or high < neutral
                    cell(ax, x, y, 1, DIVERGING(norm(estimate)), f"{estimate:.2f}", "black", 7.5, bold=significant, hatch=sparse)
                    if not sparse:
                        ax.text(x + 0.95, y + 0.85, row["n_families"], ha="right", va="bottom", color=TEXT_SECONDARY, fontsize=5)
            caption(fig, ax, norm, label, notes, f"Validation of static runtime-cost metrics: {name}, {title}", subtitle)
            save(fig, str(out_dir / f"{name}_{design}.png"))
    print(f"Saved plots to {out_dir}")


def rows() -> list[tuple[int, str]]:
    """(row position, metric) in the fixed order, leaving one row per representation header."""
    result, y = [], 0
    for hypotheses in HYPOTHESES.values():
        y += 1
        result += [(y + i, metric) for i, metric in enumerate(hypotheses)]
        y += len(hypotheses)
    return result


def draw_rows(width: float):
    """Figure with the representation headers and metric labels (with preferred direction) left of the cells."""
    fig, ax = plt.subplots(figsize=(3 + 1.6 * width, 0.27 * len(rows()) + 3), facecolor=SURFACE)
    y = 0
    for representation, hypotheses in HYPOTHESES.items():
        ax.text(-0.08, y + 0.6, representation, ha="right", va="center", color=TEXT, fontsize=8, fontweight="bold")
        ax.axhline(y + 0.98, color=GRID, linewidth=0.8)
        for i, (metric, (direction, _)) in enumerate(hypotheses.items()):
            ax.text(-0.08, y + i + 1.5, f"{LABELS.get(metric, metric)}  {'↓' if direction == 1 else '↑'}", ha="right", va="center",
                    color=TEXT_SECONDARY, fontsize=7.5)
        y += len(hypotheses) + 1
    ax.set_xlim(0, width)
    ax.set_ylim(y, 0)
    ax.axis("off")
    return fig, ax


def cell(ax, x: float, y: int, w: float, color, text: str, text_color: str, size: float, bold: bool = False, hatch: bool = False) -> None:
    ax.add_patch(Rectangle((x + 0.03, y + 0.06), w - 0.06, 0.88, facecolor=color, edgecolor="white" if hatch else "none", hatch="////" if hatch else None,
                           linewidth=0))
    ax.text(x + w / 2, y + 0.53, text, ha="center", va="center", color=text_color, fontsize=size, fontweight="bold" if bold else "normal")


def caption(fig, ax, norm, label: str, notes: list[list[str]], title: str, subtitle: str) -> None:
    colorbar = fig.colorbar(ScalarMappable(norm, DIVERGING), ax=ax, orientation="horizontal", fraction=0.012, pad=0.01, aspect=50)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=7)
    colorbar.set_label(label, color=TEXT_SECONDARY, fontsize=7.5)
    # two note blocks side by side below the color bar and its label, centered around it with a gap of 30 points (offsets in points)
    for block, dx, ha in zip(notes, (-15, 15), ("right", "left")):
        colorbar.ax.annotate("\n".join(block), xy=(0.5, 0), xycoords="axes fraction", xytext=(dx, -42), textcoords="offset points",
                             ha=ha, va="top", multialignment="left", color=MUTED, fontsize=7, family="monospace")
    # title and subtitle centered on the saved (cropped) png, i.e. on everything drawn so far (offsets in points above the column headers)
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    axes = ax.get_position()  # in figure fraction (not affected by the tight cropping when saving)
    x = ((bbox.x0 + bbox.x1) / 2 / fig.get_figwidth() - axes.x0) / axes.width
    ax.annotate(title, xy=(x, 1), xycoords="axes fraction", xytext=(0, 64), textcoords="offset points", ha="center",
                va="bottom", color=TEXT, fontsize=12, fontweight="bold")
    ax.annotate(subtitle, xy=(x, 1), xycoords="axes fraction", xytext=(0, 42), textcoords="offset points", ha="center",
                va="bottom", color=TEXT, fontsize=9, fontweight="bold")

