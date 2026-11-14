"""Concordance, gap response and decision regret of the static metrics on circuit pairs (plan §4, §5), and their plots:
F1 (validation scorecard: concordance C) and F2 (gap response: signed cost separation per metric gap bin).

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

GAP_BINS = ("small", "medium", "large")  # metric percentile gap < 1/3, < 2/3, <= 1
COLUMNS = [("TI", "IonShuttler"), ("NA", "QMAP"), ("NA", "Bloqade"), ("SC", "QMAP"), ("SC", "SABRE")]
DESIGNS = {"comparable_task": "comparable-task pairs (circuits of the same task folder)",
           "matched_workload": "matched-workload pairs (equal qubits, 2q gates within 1.25x)"}
DIVERGING = LinearSegmentedColormap.from_list("diverging", [REFERENCE, "#f6d3cf", "#f7f6f2", "#cfe0f5", SERIES])
UNAVAILABLE = "#efeeea"


# ------------------------------------------------------------------------------------------------------------------------------------------------
# hypotheses: per metric whether lower (1) or higher (-1) values are better, and the compiled cost it should predict/correspond to
HYPOTHESES = {
    "Dependency DAG": {
        "critical_path_depth": (1, "schedule"),
        "2q_depth": (1, "schedule"),
        "critical_path_duration": (1, "schedule"),
        "parallelism_saturation": (1, "schedule"),
        "capacity_constrained_depth_bound": (1, "schedule"),
        "min_idle_steps": (1, "idle"),
        "average_scheduling_slack": (-1, "schedule"),
        "average_weighted_scheduling_slack": (-1, "schedule"),
        "no_slack_fraction": (1, "schedule"),
        "min_scheduling_window_congestion": (1, "schedule"),
        "max_scheduling_window_congestion": (1, "schedule"),
        "average_asap_qubit_reuse": (-1, "idle"),
        "average_alap_qubit_reuse": (-1, "idle"),
        "average_asap_qubit_interaction": (1, "routing"),
    },
    "Gate distribution": {
        "nr_gates": (1, "schedule"),
        "share_2q_gates": (1, "routing"),
        "share_measurements": (1, "readout"),
        "share_resets": (1, "readout"),
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
        "share_mid_circuit_measurements": (1, "readout"),
        "_activity_imbalance": (1, "schedule"),
    },
    "Native translation (deferred)": {
        "share_native_gates": (-1, "native"),
        "share_native_2q_gates": (-1, "native"),
        "estimated_nr_gates_post_transpilation": (1, "native"),
        "estimated_nr_2q_gates_post_transpilation": (1, "native"),
        "_average_1q_run_length": (-1, "native"),
    },
    "Workload baselines": {
        "num_qubits": (1, "routing"),
        "N_2q": (1, "routing"),
        "N_gate": (1, "routing"),
    },
}

# ------------------------------------------------------------------------------------------------------------------------------------------------

def visualize(results_dir: Path) -> None:
    summary, gaps = summarize(results_dir)
    write_csv(results_dir / "summary.csv", summary)
    write_csv(results_dir / "gap_summary.csv", gaps)
    summary = [row for row in summary if row["scope"] == "platform"]
    gaps = [row for row in gaps if row["scope"] == "platform"]
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for design, title in DESIGNS.items():
        cells = {(r["metric"], r["platform"], r["compiler"]): r for r in summary if r.get("design") in (design, None)}
        bins = {(r["metric"], r["platform"], r["compiler"], r["gap_bin"]): r for r in gaps if r["design"] == design}

        # F1: concordance per metric and platform / compiler
        fig, ax = draw_rows(len(COLUMNS), f"Static metric validation scorecard: {title}")
        norm = Normalize(0.25, 0.75)
        for x, (platform, compiler) in enumerate(COLUMNS):
            ax.text(x + 0.5, -0.3, f"{platform}\n{compiler}", ha="center", va="bottom", color=TEXT, fontsize=8.5, fontweight="bold")
            for y, metric in rows():
                row = cells.get((metric, platform, compiler))
                if row is None or row["status"] != "defined":
                    cell(ax, x, y, 1, UNAVAILABLE, row["status"] if row else "no runs", MUTED, 6.5)
                    continue
                significant = float(row["C_lo"]) > 0.5 or float(row["C_hi"]) < 0.5
                cell(ax, x, y, 1, DIVERGING(norm(float(row["C"]))), f"{float(row['C']):.2f}", MUTED if row["sparse"] else TEXT, 7.5,
                     bold=significant, hatch=row["sparse"])
                ax.text(x + 0.95, y + 0.85, str(row["n_families"]), ha="right", va="bottom", color=TEXT_SECONDARY, fontsize=5)
        caption(fig, ax, norm, "C: weighted share of metric-untied pairs whose predicted-worse circuit costs more (0.5 = random)", [
            "Costs: routing = moves per input 2q gate (SC swaps, TI shuttles, NA atom transports); zones = zone changes per qubit operation.",
            "Bold: family-bootstrap 95% CI excludes 0.5. Small number: code families with untied pairs; hatched: fewer than 3.",
            "Grey: no estimate (cost not measured by the compilation wrappers, all pairs tied, or no pairs)."])
        save(fig, str(out_dir / f"f1_scorecard_{design}.png"))

        # F2: median signed separation d_y (color) and median |d_y| (number) per metric percentile gap bin
        width = len(GAP_BINS) * 0.6 + 0.25
        fig, ax = draw_rows(len(COLUMNS) * width, f"Gap response: {title}")
        norm = Normalize(-0.5, 0.5)
        for g, (platform, compiler) in enumerate(COLUMNS):
            ax.text(g * width + 0.9, -0.9, f"{platform} {compiler}", ha="center", va="bottom", color=TEXT, fontsize=8.5, fontweight="bold")
            for b, gap_bin in enumerate(GAP_BINS):
                ax.text(g * width + (b + 0.5) * 0.6, -0.15, gap_bin, ha="center", va="bottom", color=TEXT_SECONDARY, fontsize=6.5)
            for y, metric in rows():
                row = cells.get((metric, platform, compiler))
                if row is None or row["status"] != "defined":
                    cell(ax, g * width, y, 1.8, UNAVAILABLE, row["status"] if row else "no runs", MUTED, 6.5)
                    continue
                for b, gap_bin in enumerate(GAP_BINS):
                    gap = bins.get((metric, platform, compiler, gap_bin))
                    if gap is None:
                        cell(ax, g * width + b * 0.6, y, 0.6, "#f7f6f2", "–", MUTED, 6)
                        continue
                    sparse = gap["n_families"] < 3
                    cell(ax, g * width + b * 0.6, y, 0.6, DIVERGING(norm(float(gap["median_signed_d"]))), f"{float(gap['median_abs_d']):.2f}",
                         MUTED if sparse else TEXT, 6, hatch=sparse)
        caption(fig, ax, norm, "median signed separation d_y = (y_worse - y_better) / (y_worse + y_better) of the predicted pair (1/3 = twice the cost)", [
            "Bins: metric percentile gap of the two circuits (small < 1/3 <= medium < 2/3 <= large); all metric-untied pairs incl. reversals.",
            "Number: median |d_y|, i.e. whether physical differences grow with the metric gap. Hatched: fewer than 3 families; '–': empty bin."])
        save(fig, str(out_dir / f"f2_gap_response_{design}.png"))
    print(f"Saved plots to {out_dir}")


def rows() -> list[tuple[int, str]]:
    """(row position, metric) in the fixed order, leaving one row per representation header."""
    result, y = [], 0
    for hypotheses in HYPOTHESES.values():
        y += 1
        result += [(y + i, metric) for i, metric in enumerate(hypotheses)]
        y += len(hypotheses)
    return result


def draw_rows(width: float, title: str):
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
    ax.set_title(title, loc="left", color=TEXT, fontsize=11, fontweight="bold", pad=34)
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


def summarize(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """Per config / compiler and per platform / compiler (configs with equal weight): C, decisiveness and regret of every metric
    (summary), and its gap-response profile (gaps), read from the raw results."""
    warnings.simplefilter("ignore", RuntimeWarning)  # empty bootstrap replicates are expected
    strata = defaultdict(dict)  # (platform, config, compiler) -> circuit -> {group, family, metrics, workload, compiled targets}
    for file in sorted((results_dir / "raw").glob("*/*.csv")):
        with file.open(newline="") as f:
            rows = list(csv.DictReader(f))
        values = {row["metric"]: row["value"] for row in rows}
        if values["status"] in ("success", "reused"):
            strata[(rows[0]["platform"], rows[0]["config"], rows[0]["compiler"])][rows[0]["circuit"]] = {
                "group": rows[0]["group"], "family": rows[0]["group"].split("/")[0], "num_qubits": int(rows[0]["num_qubits"]),
                **{row["metric"]: None if row["value"] == "" else float(row["value"]) for row in rows if row["source"] != "run"}}

    families = sorted({c["family"] for circuits in strata.values() for c in circuits.values()})
    rng = np.random.default_rng(0)
    multiplicity = dict(zip(families, rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=1000).T))

    summary, gaps = [], []
    for (platform, config, compiler), circuits in sorted(strata.items()):
        for hypotheses in HYPOTHESES.values():
            for metric, (direction, target) in hypotheses.items():
                base = {"scope": "config", "platform": platform, "config": config, "compiler": compiler, "metric": metric, "direction": direction}
                if all(c.get(target) is None for c in circuits.values()):
                    summary.append({**base, "status": f"{target} n/a"})
                    continue
                for design in ("comparable_task", "matched_workload"):
                    row, gap = evaluate(circuits, metric, direction, target, design, multiplicity)
                    summary.append({**base, "design": design, "target": target, **row})
                    gaps += [{**base, "design": design, "target": target, **g} for g in gap]
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
        replicates = np.nanmean([r["_C"] for r in defined], axis=0)
        n_families = max(r["n_families"] for r in defined)
        summary.append({**platform_row, "scope": "platform", "config": "ALL", "status": "defined",
                        **{k: np.mean([r[k] for r in defined]) for k in ("C", "decisiveness", "regret_rel", "regret_rel_N_2q", "regret_rel_random")},
                        "C_lo": np.nanpercentile(replicates, 2.5), "C_hi": np.nanpercentile(replicates, 97.5),
                        "n_pairs": sum(r["n_pairs"] for r in defined), "n_families": n_families, "sparse": n_families < 3,
                        "n_configs": len(defined)})
    for row in summary:
        row.pop("_C", None)

    pooled = defaultdict(list)
    for g in gaps:
        pooled[(g["platform"], g["compiler"], g["metric"], g["design"], g["target"], g["gap_bin"])].append(g)
    for (platform, compiler, metric, design, target, gap_bin), rows in pooled.items():
        d = np.concatenate([r.pop("_d") for r in rows])
        w = np.concatenate([r.pop("_w") for r in rows])  # weights sum to 1 per config, so configs count equally
        gaps.append({"scope": "platform", "platform": platform, "config": "ALL", "compiler": compiler, "metric": metric, "design": design,
                     "target": target, "gap_bin": gap_bin,
                     "C": np.average(np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)), weights=w) if w.sum() else np.nan,
                     "median_signed_d": weighted_median(d, w), "median_abs_d": weighted_median(np.abs(d), w),
                     "n_pairs": len(d), "n_families": max(r["n_families"] for r in rows)})
    return summary, gaps


def evaluate(circuits: dict, metric: str, direction: int, target: str, design: str, multiplicity: dict) -> tuple[dict, list]:
    """Concordance C, decisiveness and relative regret (with family-bootstrap CI) and the gap-response profile of one metric on one pair set."""
    names = [n for n, c in circuits.items() if c[metric] is not None and c[target] is not None]
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
        return {"status": "no pairs"}, []

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
        return row, []

    # gap response: metric percentile gap (tied midranks among the eligible circuits) against the signed relative cost gap
    percentile = rankdata(s) / len(s)
    g = np.abs(percentile[left] - percentile[right])[untied]
    d = np.divide(y_worse - y_better, y_worse + y_better, out=np.zeros(len(left)), where=y_worse + y_better > 0)[untied]
    w_untied = w[untied] / w[untied].sum()
    profile = []
    for b, gap_bin in enumerate(GAP_BINS):
        inside = np.digitize(g, [1 / 3, 2 / 3]) == b
        if not inside.any():
            continue
        profile.append({"gap_bin": gap_bin, "C": np.average(score[untied][inside], weights=w_untied[inside]),
                        "median_signed_d": weighted_median(d[inside], w_untied[inside]), "median_abs_d": weighted_median(np.abs(d[inside]), w_untied[inside]),
                        "n_pairs": int(inside.sum()), "n_families": len(set(family[left[untied][inside]]) | set(family[right[untied][inside]])),
                        "_d": d[inside], "_w": w_untied[inside]})
    return row, profile


def regret(s: np.ndarray, y: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Relative regret of choosing the circuit with the smaller oriented score (expected cost of a random choice on ties)."""
    low, high = np.minimum(y[left], y[right]), np.maximum(y[left], y[right])
    selected = np.where(np.isclose(s[left], s[right]), (y[left] + y[right]) / 2, np.where(s[left] < s[right], y[left], y[right]))
    return np.divide(selected - low, high, out=np.zeros(len(left)), where=high > 0)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    cumulative = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def write_csv(file: Path, rows: list[dict]) -> None:
    with file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
