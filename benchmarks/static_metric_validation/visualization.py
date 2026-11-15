"""Concordance, proportionality and decision regret of the static metrics on circuit pairs (plan §4, §5), and their plots:
scorecard (concordance C: right order of the costs) and proportionality (P: larger metric gap = larger cost difference).
Only the runtime cost aspect is validated (hardware events of the compiled circuit), not the compilation cost (compiler work / freedom).

Pairs: comparable task (circuits in the same data folder) and matched workload (equal qubits, similar number of 2q gates).
Weights balance task groups / code families; uncertainty comes from resampling code families (first data folder level).
Rows of the plots are the metrics in the fixed hypothesis order, columns the platform / compiler combinations (configs with equal weight).
"""

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy.stats import rankdata

from benchmarks.common_reference.plotting import SURFACE, TEXT, TEXT_SECONDARY, MUTED, GRID, SERIES, REFERENCE, save

COLUMNS = [("TI", "IonShuttler"), ("NA", "QMAP"), ("NA", "Bloqade"), ("SC", "QMAP"), ("SC", "SABRE")]
DESIGNS = {"comparable_task": "comparable-task pairs (circuits for the same task and QEC code)",
           "matched_workload": "matched-workload pairs (circuit with the same number of qubits, and roughly same number of 2q gates within 1.25x)"}
DIVERGING = LinearSegmentedColormap.from_list("diverging", [REFERENCE, "#f6d3cf", "#f7f6f2", "#cfe0f5", SERIES])
UNAVAILABLE = "#efeeea"


# ------------------------------------------------------------------------------------------------------------------------------------------------

# targets: raw (aggregated, unnormalized) values associated with runtime cost of a compiled circuit
TARGETS = {
    "routing": {"TI": ["shuttles"], "NA": ["transports"], "SC": ["swaps"]}, # qubit movements
    "zones": {"TI": ["interaction_zone_entry", "interaction_zone_eviction"], "NA": ["zone_changes"]}, # zone changes
    "schedule": {"TI": ["schedule_length"], "NA": ["schedule_length"], "SC": ["schedule_length"]}, # schedule length
    "idle": {"TI": ["idle_time"], "NA": ["idle_time"], "SC": ["idle_time"]},  # idle times
}

# hypotheses: per metric (that approximates runtime costs) whether lower (1) or higher (-1) values are better, and the cost it should predict/correspond to
# runtime-cost metrics not evaluated: share_measurements, share_resets, share_mid_circuit_measurements (measurements / resets will not be changed during the compilation, so there is nothing to verify),
# estimated_nr_gates_post_transpilation, estimated_nr_2q_gates_post_transpilation (utilized compilers do not support transpilations themselves)
HYPOTHESES = {
    "Dependency DAG": {
        "critical_path_depth": (1, "schedule"),
        "2q_depth": (1, "schedule"),
        "critical_path_duration": (1, "schedule"),
        "parallelism_saturation": (1, "schedule"),
        "capacity_constrained_depth_bound": (1, "schedule"),
        "min_idle_steps": (1, "idle"),
        "min_scheduling_window_congestion": (1, "schedule"),
        "max_scheduling_window_congestion": (1, "schedule"),
        "average_asap_qubit_reuse": (-1, "idle"),
        "average_alap_qubit_reuse": (-1, "idle"),
        "average_asap_qubit_interaction": (1, "routing"),
    },
    "Gate distribution": {
        "nr_gates": (1, "schedule"),
        "share_2q_gates": (1, "routing"),
    },
    "Weighted interaction graph": {
        "fraction_of_partner_capacity_exceeding_qubits": (1, "routing"),
        "weighted_fraction_of_partner_capacity_exceeding_qubits": (1, "routing"),
        "fraction_of_effective_partner_capacity_exceeding_qubits": (1, "routing"),
        "max_partner_capacity_ratio": (1, "routing"),
        "average_interaction_partner_diversity": (1, "routing"),
        "average_interaction_entropy": (1, "routing"),
        "average_effective_partner_number": (1, "routing"),
        "interaction_concentration": (-1, "routing"),
        "statically_servable_interaction_fraction": (-1, "routing"),
        "capacity_constrained_qubit_partition": (1, "routing"),
        "non_bipartite_interaction_fraction": (1, "routing"),
        "nr_connected_components": (-1, "routing"),
        "largest_component_fraction": (1, "routing"),
    },
    "Qubit timelines": {
        "average_interaction_reuse_distance": (1, "routing"),
        "share_partner_reusing_qubits": (-1, "routing"),
        "capacity_constrained_reuse": (-1, "routing"),
        "partner_switching_frequency": (1, "routing"),
        "zone_switching_frequency": (1, "zones"),
        "_activity_imbalance": (1, "schedule"),
    },
    "Workload baselines": {
        "N_2q": (1, "routing"),
    },
}

# ------------------------------------------------------------------------------------------------------------------------------------------------

def visualize(results_dir: Path) -> None:
    summary = summarize(results_dir)
    write_csv(results_dir / "summary.csv", summary)
    summary = [row for row in summary if row["scope"] == "platform"]
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    costs = ["Runtime costs (aggregated per compiled circuit, not normalized):",
             "   - routing  = qubit moves (SC swaps, TI shuttles, NA atom transports)",
             "   - zones    = zone changes (TI interaction zones, NA functional zones)",
             "   - schedule = compiled schedule length",
             "   - idle     = time qubits wait between their gates (in compiler timesteps)",
             ""]
    # per plot: value, its neutral value and color range, file name, subtitle, color bar label, notes
    plots = [("C", 0.5, Normalize(0.25, 0.75), "scorecard",
              "Correlation: Comparing two circuits, does the static metric predict the right order of their actual runtime costs?",
              "weighted share of circuit pairs whose predicted-worse circuit actually costs more (0.5 = random)",
              costs + ["bold: 95% CI excludes 0.5; hatched: fewer than 3 code families to compare"]),
             ("P", 0.0, Normalize(-1, 1), "proportionality",
              "Proportionality: Comparing two circuits, does a larger difference in the static metric correspond to a larger difference in runtime cost?",
              "weighted Spearman correlation between the metric difference of a pair and its signed cost difference " r"$\dfrac{y_{\mathrm{worse}} - y_{\mathrm{better}}}{y_{\mathrm{worse}} + y_{\mathrm{better}}}$" " (0 = no relation)",
              costs + ["metric difference = distance of the two circuits in the metric's ranking",
                       "signed cost difference > 0 if the predicted-worse circuit costs more",
                       "hatched: fewer than 3 circuits to compare"])]

    for design, title in DESIGNS.items():
        cells = {(r["metric"], r["platform"], r["compiler"]): r for r in summary if r.get("design") in (design, None)}
        for value, neutral, norm, name, subtitle, label, notes in plots:
            fig, ax = draw_rows(len(COLUMNS), f"Validation of static (runtime cost) metrics - {name} of {title}", subtitle)
            for x, (platform, compiler) in enumerate(COLUMNS):
                ax.text(x + 0.5, -0.3, f"{platform}\n{compiler}", ha="center", va="bottom", color=TEXT, fontsize=8.5, fontweight="bold")
                for y, metric in rows():
                    row = cells.get((metric, platform, compiler))
                    if row is None or row["status"] != "defined":
                        cell(ax, x, y, 1, UNAVAILABLE, row["status"] if row else "no runs", MUTED, 6.5)
                        continue
                    if np.isnan(row[value]):
                        cell(ax, x, y, 1, UNAVAILABLE, "–", MUTED, 6.5)
                        continue
                    significant = row[f"{value}_lo"] > neutral or row[f"{value}_hi"] < neutral
                    cell(ax, x, y, 1, DIVERGING(norm(row[value])), f"{row[value]:.2f}", "black", 7.5, bold=significant, hatch=row["sparse"])
                    if not row["sparse"]:
                        ax.text(x + 0.95, y + 0.85, str(row["n_families"]), ha="right", va="bottom", color=TEXT_SECONDARY, fontsize=5)
            caption(fig, ax, norm, label, notes)
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


def draw_rows(width: float, title: str, subtitle: str):
    """Figure with the representation headers and metric labels (with preferred direction) left of the cells."""
    fig, ax = plt.subplots(figsize=(3 + 1.6 * width, 0.27 * len(rows()) + 3), facecolor=SURFACE)
    y = 0
    for representation, hypotheses in HYPOTHESES.items():
        ax.text(-0.08, y + 0.6, representation, ha="right", va="center", color=TEXT, fontsize=8, fontweight="bold")
        ax.axhline(y + 0.98, color=GRID, linewidth=0.8)
        for i, (metric, (direction, _)) in enumerate(hypotheses.items()):
            ax.text(-0.08, y + i + 1.5, f"{metric}  {'↓' if direction == 1 else '↑'}", ha="right", va="center", color=TEXT_SECONDARY, fontsize=7.5)
        y += len(hypotheses) + 1
    ax.set_xlim(0, width)
    ax.set_ylim(y, 0)
    ax.axis("off")
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=52)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 36), textcoords="offset points", ha="left", va="bottom",
                color=TEXT, fontsize=9, fontweight="bold")
    return fig, ax


def cell(ax, x: float, y: int, w: float, color, text: str, text_color: str, size: float, bold: bool = False, hatch: bool = False) -> None:
    ax.add_patch(Rectangle((x + 0.03, y + 0.06), w - 0.06, 0.88, facecolor=color, edgecolor="white" if hatch else "none", hatch="////" if hatch else None,
                           linewidth=0))
    ax.text(x + w / 2, y + 0.53, text, ha="center", va="center", color=text_color, fontsize=size, fontweight="bold" if bold else "normal")


def caption(fig, ax, norm, label: str, notes: list[str]) -> None:
    colorbar = fig.colorbar(ScalarMappable(norm, DIVERGING), ax=ax, orientation="horizontal", fraction=0.012, pad=0.01, aspect=50)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=7)
    colorbar.set_label(label, color=TEXT_SECONDARY, fontsize=7.5)
    fig.text(0.02, colorbar.ax.get_position().y0 - 0.035, "\n".join(notes), color=MUTED, fontsize=7, va="top")


def summarize(results_dir: Path) -> list[dict]:
    """Per config / compiler and per platform / compiler (configs with equal weight): C, P, decisiveness and regret of every metric,
    read from the raw results."""
    warnings.simplefilter("ignore", RuntimeWarning)  # empty bootstrap replicates are expected
    strata = defaultdict(dict)  # (platform, config, compiler) -> circuit -> {group, family, structural metrics, compiled values, targets}
    for file in sorted((results_dir / "raw").glob("*/*.csv")):
        with file.open(newline="") as f:
            rows = list(csv.DictReader(f))
        platform = rows[0]["platform"]
        circuit = {"group": rows[0]["group"], "family": rows[0]["group"].split("/")[0], "num_qubits": int(rows[0]["num_qubits"]),
                   "N_2q": int(rows[0]["N_2q"]), **{row["metric"]: None if row["value"] == "" else float(row["value"]) for row in rows}}
        for target, values in TARGETS.items():
            if platform in values and all(value in circuit for value in values[platform]):
                circuit[target] = sum(circuit[value] for value in values[platform])
        strata[(platform, rows[0]["config"], rows[0]["compiler"])][rows[0]["circuit"]] = circuit

    families = sorted({c["family"] for circuits in strata.values() for c in circuits.values()})
    rng = np.random.default_rng(0)
    multiplicity = dict(zip(families, rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=1000).T))

    summary = []
    for (platform, config, compiler), circuits in sorted(strata.items()):
        for hypotheses in HYPOTHESES.values():
            for metric, (direction, target) in hypotheses.items():
                base = {"scope": "config", "platform": platform, "config": config, "compiler": compiler, "metric": metric, "direction": direction}
                if all(c.get(target) is None for c in circuits.values()):
                    summary.append({**base, "status": f"{target} n/a"})
                    continue
                for design in ("comparable_task", "matched_workload"):
                    row = evaluate(circuits, metric, direction, target, design, multiplicity)
                    summary.append({**base, "design": design, "target": target, **row})
        print(f"[{platform}] Analyzed {config} / {compiler}: {len(circuits)} compiled circuits")

    # platform / compiler summaries: configs with equal weight (mean of their estimates and of their bootstrap replicates)
    grouped = defaultdict(list)
    for row in summary:
        grouped[(row["platform"], row["compiler"], row["metric"], row.get("design"))].append(row)
    for rows in grouped.values():
        defined = [r for r in rows if r["status"] == "defined"]
        platform_row = {k: rows[0].get(k) for k in ("platform", "compiler", "metric", "direction", "design", "target")}
        if not defined:
            summary.append({**platform_row, "scope": "platform", "config": "ALL", "status": rows[0]["status"]})
            continue
        replicates = {k: np.nanmean([r[f"_{k}"] for r in defined], axis=0) for k in ("C", "P")}
        n_families = max(r["n_families"] for r in defined)
        summary.append({**platform_row, "scope": "platform", "config": "ALL", "status": "defined",
                        **{k: np.nanmean([r[k] for r in defined]) for k in ("C", "P", "decisiveness", "regret_rel", "regret_rel_N_2q", "regret_rel_random")},
                        **{f"{k}_lo": np.nanpercentile(v, 2.5) for k, v in replicates.items()},
                        **{f"{k}_hi": np.nanpercentile(v, 97.5) for k, v in replicates.items()},
                        "n_pairs": sum(r["n_pairs"] for r in defined), "n_families": n_families, "sparse": n_families < 3,
                        "n_configs": len(defined)})
    for row in summary:
        row.pop("_C", None)
        row.pop("_P", None)
    return summary


def evaluate(circuits: dict, metric: str, direction: int, target: str, design: str, multiplicity: dict) -> dict:
    """Concordance C, proportionality P (both with family-bootstrap CI), decisiveness and relative regret of one metric on one pair set."""
    names = [n for n, c in circuits.items() if c[metric] is not None and c.get(target) is not None]
    if metric in ("average_interaction_reuse_distance", "capacity_constrained_reuse"):  # 0 means "no partner reuse", not the best value
        names = [n for n in names if circuits[n]["share_partner_reusing_qubits"] > 0]
    group = np.array([circuits[n]["group"] for n in names])
    family = np.array([circuits[n]["family"] for n in names])
    s = direction * np.array([circuits[n][metric] for n in names], dtype=float)  # oriented: larger predicts worse
    y = np.array([circuits[n][target] for n in names], dtype=float)
    left, right = np.triu_indices(len(names), 1)
    if design == "comparable_task":
        keep = group[left] == group[right]
    else:
        q, n2q = np.array([circuits[n]["num_qubits"] for n in names]), np.array([circuits[n]["N_2q"] for n in names])
        keep = (q[left] == q[right]) & (np.maximum(n2q[left], n2q[right]) <= 1.25 * np.minimum(n2q[left], n2q[right]))  # equal qubits, 2q gates within 1.25x
    left, right = left[keep], right[keep]
    if len(left) == 0:
        return {"status": "no pairs"}

    # weights: comparable task 1 / (groups * pairs in group); matched workload a_i + a_j with a_i = 1 / (families * circuits in family * partners)
    if design == "comparable_task":
        _, inverse, counts = np.unique(group[left], return_inverse=True, return_counts=True)
        w = 1 / (len(counts) * counts[inverse])
    else:
        partners = np.bincount(np.concatenate([left, right]), minlength=len(names))
        anchored = partners > 0
        per_family = {f: np.sum(anchored & (family == f)) for f in set(family[anchored])}
        a = np.array([1 / (len(per_family) * per_family[f] * k) if k else 0 for f, k in zip(family, partners)])
        w = a[left] + a[right]
    # bootstrap: a pair counts once per copy of its family (same family) or per combination of copies (different families)
    m_i, m_j = np.array([multiplicity[f] for f in family[left]]).T, np.array([multiplicity[f] for f in family[right]]).T
    W = np.vstack([w, w * np.where(family[left] == family[right], m_i, m_i * m_j)])

    untied = ~np.isclose(s[left], s[right])
    y_worse, y_better = np.where(s[left] > s[right], y[left], y[right]), np.where(s[left] > s[right], y[right], y[left])
    target_tied = np.isclose(y_worse, y_better)
    score = untied * np.where(target_tied, 0.5, y_worse > y_better)
    C = (W @ score) / (W @ untied)
    families_untied = len(set(family[left[untied]]) | set(family[right[untied]]))
    row = {"status": "defined" if untied.any() else "ties", "C": C[0], "C_lo": np.nanpercentile(C[1:], 2.5), "C_hi": np.nanpercentile(C[1:], 97.5),
           "decisiveness": w @ untied / w.sum(), "target_tie_share": w @ (untied & target_tied) / (w @ untied),
           "regret_rel": np.average(regret(s, y, left, right), weights=w),
           "regret_rel_N_2q": np.average(regret(np.array([circuits[n]["N_2q"] for n in names]), y, left, right), weights=w),
           "regret_rel_random": np.average(np.divide(np.abs(y[left] - y[right]) / 2, np.maximum(y[left], y[right]), out=np.zeros(len(left)), where=np.maximum(y[left], y[right]) > 0), weights=w),
           "n_circuits": len(set(left) | set(right)), "n_pairs": len(left), "n_untied_pairs": int(untied.sum()), "n_families": families_untied,
           "sparse": families_untied < 3, "_C": C[1:]}
    if not untied.any():
        return row

    # proportionality P: weighted Spearman correlation between the metric gap (percentile distance of the two circuits) and the signed
    # relative cost difference d_y of the untied pairs (same weights and bootstrap replicates as C)
    percentile = rankdata(s) / len(s)
    g = rankdata(np.abs(percentile[left] - percentile[right])[untied])
    d = rankdata(np.divide(y_worse - y_better, y_worse + y_better, out=np.zeros(len(left)), where=y_worse + y_better > 0)[untied])
    V = W[:, untied] / W[:, untied].sum(axis=1, keepdims=True)
    covariance = V @ (g * d) - (V @ g) * (V @ d)
    P = covariance / np.sqrt((V @ g ** 2 - (V @ g) ** 2) * (V @ d ** 2 - (V @ d) ** 2))
    row.update({"P": P[0], "P_lo": np.nanpercentile(P[1:], 2.5), "P_hi": np.nanpercentile(P[1:], 97.5), "_P": P[1:]})
    return row


def regret(s: np.ndarray, y: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Relative regret of choosing the circuit with the smaller oriented score (expected cost of a random choice on ties)."""
    low, high = np.minimum(y[left], y[right]), np.maximum(y[left], y[right])
    selected = np.where(np.isclose(s[left], s[right]), (y[left] + y[right]) / 2, np.where(s[left] < s[right], y[left], y[right]))
    return np.divide(selected - low, high, out=np.zeros(len(left)), where=high > 0)


def write_csv(file: Path, rows: list[dict]) -> None:
    with file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
