"""Compare abstract hardware events with compiled events and physical movement.

Aggregated event counts assess agreement or proxy scaling (correlation + proportionality).
Location scores compare occupied slots; timing alignment also distinguishes event kinds.
"""

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import kendalltau, spearmanr, theilslopes

SURFACE, TEXT, TEXT_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SERIES, FIT, REFERENCE = "#e1e0d9", "#c3c2b7", "#2a78d6", "#0d366b", "#e34948"
PER_GATE = "#2f9a5e"
RING_SIZE, RING_WIDTH = 40, 1.4
QUBIT_CMAP = LinearSegmentedColormap.from_list(
    "qubits",
    ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)
QUBIT_MIN, QUBIT_MAX = 4, 50
QUBIT_TICKS = (4, 10, 20, 30, 40, 50)
COUNT_TICKS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)
DPI = 200
EQUIVALENCE_FACTOR = 1.5
BAND_SHARE = 0.8
BOOTSTRAP_SAMPLES = 500
F1_TOLERANCES = (0, 1, 2, 3)
ON_TIME = 0.25
ALIGNMENT_GAP = 2.0

ABSTRACT_MARKER = dict(marker="o", markersize=7, linestyle="none", color=SERIES)
PER_GATE_MARKER = dict(
    marker="o", markersize=6, linestyle="none", markerfacecolor="none", markeredgecolor=PER_GATE, markeredgewidth=1.4
)
TI_IZ_CHANGES = ("interaction_zone_entry", "interaction_zone_eviction")
TI_ION_MOVES = ("shuttles", "swaps")  # swaps count once per participating ion
NA_ZONE_CHANGES = ("storage_entanglement", "storage_readout", "entanglement_readout", "inter_entanglement")
NA_COMPARABLE_ZONE_CHANGES = ("storage_entanglement", "inter_entanglement")  # compiler has no readout zone


def visualize(platform: list[str], results_dir: Path, max_qubits: int | None = None) -> None:
    for hardware in ("TI", "NA", "SC"):
        if hardware not in platform:
            continue
        prefix = hardware.lower()
        out_dir = results_dir / prefix / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        for old_file in out_dir.glob(f"{prefix}_*__*"):
            if old_file.is_file() and old_file.suffix in (".png", ".csv"):
                old_file.unlink()

        runs = defaultdict(list)
        for file in sorted((results_dir / prefix / "raw").glob("*.csv")):
            with file.open(newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and (max_qubits is None or int(rows[0]["num_qubits"]) <= max_qubits):
                runs[rows[0]["config"]].append(rows)
        for config, circuits in sorted(runs.items()):
            name = re.sub(r"[^a-z0-9]+", "_", config.lower()).strip("_")
            file_prefix = f"{out_dir}/{prefix}_{name}"
            visualize_aggregated(hardware, config, circuits, file_prefix)
            visualize_distribution(hardware, config, circuits, file_prefix)


def circuits_subtitle(circuits: list[list[dict]]) -> str:
    speedups = []
    for rows in circuits:
        wall_clock_s = {row["source"]: float(row["value"]) for row in rows if row["metric"] == "wall_clock_s"}
        speedups.append(wall_clock_s["compiled"] / wall_clock_s["abstract"])
    speedup = f"{np.mean(speedups):.3g}×" if speedups else "n/a"
    return f"number of circuits = {len(circuits)}   mean speedup (compile / abstract time): {speedup}"


def style(fig, ax, title: str, subtitle: str, x_label: str, y_label: str, notes: tuple[str, ...] = ()) -> None:
    for note in notes:
        ax.plot([], [], linestyle="none", label=note)
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, va="bottom", ha="left", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.13),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
        handlelength=2.4,
    )
    fig.tight_layout()


def lin_ccc(xs: list[float], ys: list[float]) -> float | None:
    """Lin's CCC on positive log counts: 2 cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))²).

    Unlike correlation, this penalizes scale bias; zero pairs are omitted.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 3:
        return None
    log_x, log_y = np.log10(pairs).T
    if np.ptp(log_x) == np.ptp(log_y) == 0:
        return None if log_x[0] == log_y[0] else 0.0
    covariance = np.mean((log_x - log_x.mean()) * (log_y - log_y.mean()))
    denominator = log_x.var() + log_y.var() + (log_x.mean() - log_y.mean()) ** 2
    return float(2 * covariance / denominator) if denominator else None


def correlation(xs: list[float], ys: list[float], rng: np.random.Generator) -> str:
    """Kendall τ-b with a circuit-bootstrap percentile CI, and Spearman ρ (including zeros)."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return "no variation"
    xs, ys = np.asarray(xs), np.asarray(ys)
    tau = kendalltau(xs, ys).statistic
    resamples = (rng.integers(0, len(xs), len(xs)) for _ in range(BOOTSTRAP_SAMPLES))
    low, high = np.nanpercentile([kendalltau(xs[i], ys[i]).statistic for i in resamples], [2.5, 97.5])
    return f"Kendall τ = {tau:.2f}, 95% CI [{low:.2f}, {high:.2f}], Spearman ρ = {spearmanr(xs, ys).statistic:.2f}"


def power_law_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float, float, float, int] | None:
    """Theil-Sen fit log(y) = log(c) + b log(x) over positive pairs.

    Returns b, its 95% CI, c, the BAND_SHARE quantile of multiplicative residuals,
    and the number of omitted zero pairs. The residual band describes this sample.
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2:
        return None
    log_x, log_y = np.log10(pairs).T
    b, log_c, b_low, b_high = theilslopes(log_y, log_x)
    k = 10 ** np.quantile(np.abs(log_y - (log_c + b * log_x)), BAND_SHARE)
    return b, b_low, b_high, 10**log_c, k, len(xs) - len(pairs)


def log_count_range(counts: list[int]) -> tuple[float, float, float | None]:
    lo = min([c for c in counts if c > 0], default=1) / 1.3
    return lo, max([*counts, 1]) * 1.3, lo / 2.2 if 0 in counts else None


def visualize_aggregated(platform: str, config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    if not circuits:
        return
    if platform == "TI":
        metrics = {
            "iz_entries": ("interaction_zone_entry",),
            "iz_evictions": ("interaction_zone_eviction",),
            "iz_changes": TI_IZ_CHANGES,
        }
        compiled_metrics = {**metrics, "ion_moves": TI_ION_MOVES}
        plots = [
            (
                "iz_entries",
                "IZ entries",
                "iz_entries",
                "iz_entries",
                "abstract IZ entries",
                "compiled IZ entries",
                True,
            ),
            (
                "iz_evictions",
                "IZ evictions",
                "iz_evictions",
                "iz_evictions",
                "abstract IZ evictions",
                "compiled IZ evictions",
                True,
            ),
            (
                "iz_changes",
                "IZ changes (entries + evictions)",
                "iz_changes",
                "iz_changes",
                "abstract IZ changes",
                "compiled IZ changes",
                True,
            ),
            (
                "iz_changes_vs_ion_moves",
                "IZ changes vs ion moves",
                "iz_changes",
                "ion_moves",
                "abstract IZ changes",
                "compiled shuttles + swaps",
                False,
            ),
        ]
    elif platform == "NA":
        comparable = {
            "storage_entanglement": ("storage_entanglement",),
            "inter_entanglement": ("inter_entanglement",),
            "zone_changes": NA_COMPARABLE_ZONE_CHANGES,
        }
        metrics = {**comparable, "all_zone_changes": NA_ZONE_CHANGES}
        compiled_metrics = {**comparable, "atom_moves": ("transports",)}
        plots = [
            (
                "storage_entanglement",
                "storage ↔ entanglement zone changes",
                "storage_entanglement",
                "storage_entanglement",
                "abstract storage ↔ EZ changes",
                "compiled storage ↔ EZ changes",
                True,
            ),
            (
                "inter_entanglement",
                "entanglement site changes",
                "inter_entanglement",
                "inter_entanglement",
                "abstract EZ site changes",
                "compiled EZ site changes",
                True,
            ),
            (
                "zone_changes",
                "zone changes (storage ↔ EZ + EZ site)",
                "zone_changes",
                "zone_changes",
                "abstract zone changes",
                "compiled zone changes",
                True,
            ),
            (
                "zone_changes_vs_atom_moves",
                "zone changes vs atom moves",
                "all_zone_changes",
                "atom_moves",
                "abstract zone changes (incl. readout)",
                "compiled atom transports",
                False,
            ),
        ]
    else:
        metrics = compiled_metrics = {"swaps": ("virtual_swap",)}
        plots = [("swaps", "virtual swaps", "swaps", "swaps", "abstract virtual swaps", "compiled swaps", False)]

    scores = []
    for rows in circuits:
        totals: defaultdict[tuple[str, str], float] = defaultdict(float)
        wall_clock_s = {}
        for row in rows:
            if row["count"]:
                totals[(row["source"], row["metric"])] += float(row["count"])
            elif row["metric"] == "wall_clock_s":
                wall_clock_s[row["source"]] = float(row["value"])
        score = {
            "circuit": rows[0]["circuit"],
            "num_qubits": int(rows[0]["num_qubits"]),
            "anchors": sum(
                int(row["value"]) for row in rows if row["source"] == "circuit" and row["metric"] == "anchors"
            ),
        }
        for source in ("abstract", "compiled", "per_gate"):
            for name, kinds in (compiled_metrics if source == "compiled" else metrics).items():
                count = int(sum(totals[(source, kind)] for kind in kinds))
                score[f"{source}_{name}"] = count // 2 if platform == "SC" else count
        score.update(abstract_wall_clock_s=wall_clock_s["abstract"], compiled_wall_clock_s=wall_clock_s["compiled"])
        scores.append(score)
    scores.sort(key=lambda c: c["num_qubits"])
    qubits = [c["num_qubits"] for c in scores]
    sizes = np.array([c["anchors"] for c in scores])
    subtitle = circuits_subtitle(circuits)

    for name, title, x_key, y_key, x_label, y_label, same_quantity in plots:
        xs = [c[f"abstract_{x_key}"] for c in scores]
        xs_per_gate = [c[f"per_gate_{x_key}"] for c in scores]
        ys = [c[f"compiled_{y_key}"] for c in scores]
        fig, ax = plt.subplots(figsize=(7, 7.6) if same_quantity else (7.6, 9), facecolor=SURFACE)
        x_range = log_count_range([*xs, *xs_per_gate, *(ys if same_quantity else [])])
        y_range = x_range if same_quantity else log_count_range(ys)
        x_lo, x_end, x_zero = x_range
        y_lo, _, y_zero = y_range
        for axis, (lo, end, zero_at) in ((ax.xaxis, x_range), (ax.yaxis, y_range)):
            axis_name = axis.axis_name
            ax.set(**{f"{axis_name}scale": "log", f"{axis_name}lim": (lo / 3.2 if zero_at else lo, end)})
            ticks = [t for t in COUNT_TICKS if lo <= t <= end and (end / lo < 1000 or str(t)[0] == "1")]
            axis.set_ticks(
                [zero_at, *ticks] if zero_at else ticks, [*(["0"] if zero_at else []), *(f"{t:g}" for t in ticks)]
            )
            if zero_at:
                (ax.axvline if axis_name == "x" else ax.axhline)(
                    lo / 1.4, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1.5
                )
        ax.minorticks_off()
        # Only drawing positions use the zero strip; statistics use the original counts.
        px = [x if x else x_zero for x in xs]
        px_per_gate = [x if x else x_zero for x in xs_per_gate]
        py = [y if y else y_zero for y in ys]
        ax.scatter(
            px_per_gate, py, s=RING_SIZE, facecolors="none", edgecolors=PER_GATE, linewidths=RING_WIDTH, zorder=3
        )
        points = ScalarMappable(Normalize(QUBIT_MIN, QUBIT_MAX), QUBIT_CMAP)
        for i, (x, y, qubit) in enumerate(zip(px, py, qubits)):
            zorder = 4 + i * 3e-5
            ax.scatter([x], [y], s=56, color=SURFACE, edgecolors=SURFACE, linewidths=1.6, zorder=zorder)
            ax.scatter([x], [y], s=56, color=points.to_rgba(qubit), linewidths=0, zorder=zorder + 2e-5)

        models = (("abstract", xs, ABSTRACT_MARKER), ("per-gate", xs_per_gate, PER_GATE_MARKER))
        ys_per_anchor = (np.array(ys)[sizes > 0] / sizes[sizes > 0]).tolist()
        if same_quantity:
            diagonal = np.array([x_zero or x_lo, x_end])
            ax.plot(
                diagonal,
                diagonal,
                color=REFERENCE,
                linewidth=3,
                zorder=2,
                label="exact match",
            )
            ax.fill_between(
                diagonal,
                diagonal / EQUIVALENCE_FACTOR,
                diagonal * EQUIVALENCE_FACTOR,
                color=REFERENCE,
                alpha=0.12,
                linewidth=0,
                zorder=1,
                label=f"within {EQUIVALENCE_FACTOR:g}× of the compiled count",
            )
            headers = ["Agreement: how close are the circuits to y = x?"]
            ax.plot([], [], linestyle="none", label=headers[0])
            for model, counts, marker in models:
                errors = [abs(x - y) / y for x, y in zip(counts, ys) if y > 0]
                within = np.mean(
                    [x <= EQUIVALENCE_FACTOR * y and y <= EQUIVALENCE_FACTOR * x for x, y in zip(counts, ys)]
                )
                ratios = [x / y for x, y in zip(counts, ys) if y > 0]
                agreement_text = [
                    f"median rel. error {np.median(errors):.0%}" if errors else "no compiled events",
                    f"{within:.0%} within {EQUIVALENCE_FACTOR:g}×",
                ]
                if ratios:
                    agreement_text.append(f"median ratio ×{np.median(ratios):.2f}")
                ax.plot([], [], **marker, label=f"{model} model: {', '.join(agreement_text)}")
                per_anchor = (np.array(counts)[sizes > 0] / sizes[sizes > 0]).tolist()
                ccc = [lin_ccc(counts, ys), lin_ccc(per_anchor, ys_per_anchor)]
                zeros = sum(x == 0 or y == 0 for x, y in zip(counts, ys))
                ax.plot(
                    [],
                    [],
                    linestyle="none",
                    label=f"    Lin's CCC (log): raw counts {'–' if ccc[0] is None else f'{ccc[0]:.2f}'},"
                    f"\n        counts per anchor: {'–' if ccc[1] is None else f'{ccc[1]:.2f}'}"
                    + (f", {zeros} with a 0 left out" if zeros else ""),
                )
            ax.plot(
                [], [], linestyle="none", label="relative errors and model / compiled ratios use compiled count > 0"
            )
        else:
            fit = power_law_fit(xs, ys)
            if fit:
                b, _, _, c, k, _ = fit
                fit_xs = np.geomspace(x_lo, x_end, 200)
                fit_ys = c * fit_xs**b
                ax.fill_between(
                    fit_xs,
                    np.maximum(fit_ys / k, y_lo),
                    np.maximum(fit_ys * k, y_lo),
                    color=FIT,
                    alpha=0.1,
                    linewidth=0,
                    zorder=1,
                    label=f"{BAND_SHARE:.0%} of positive-count circuits (in sample, within ×{k:.2g})",
                )
                ax.plot(
                    fit_xs,
                    np.where(fit_ys >= y_lo, fit_ys, np.nan),
                    color=FIT,
                    alpha=0.6,
                    linewidth=2,
                    zorder=2,
                    label=f"power-law fit y = {c:.2g} · x^{b:.2f} of the abstract model (robust, Theil-Sen)",
                )
                ratio = np.median([y / x for x, y in zip(xs, ys) if x > 0 and y > 0])
                ax.plot(
                    fit_xs,
                    np.where(ratio * fit_xs >= y_lo, ratio * fit_xs, np.nan),
                    color=REFERENCE,
                    linewidth=3,
                    zorder=2,
                    label=f"proportional (b = 1) with the median ratio y / x = {ratio:.2g}",
                )
            rng = np.random.default_rng(0)
            headers = [
                "Correlation: is the order of the circuits the same?",
                "Scaling: log-log fit y = c · x^b (b = 1: proportional)",
            ]
            ax.plot([], [], linestyle="none", label=headers[0])
            for model, counts, marker in models:
                per_anchor = (np.array(counts)[sizes > 0] / sizes[sizes > 0]).tolist()
                ax.plot([], [], **marker, label=f"{model} model, raw counts: {correlation(counts, ys, rng)}")
                ax.plot(
                    [],
                    [],
                    linestyle="none",
                    label=f"    counts per anchor:\n        {correlation(per_anchor, ys_per_anchor, rng)}",
                )
            ax.plot([], [], linestyle="none", label=headers[1])
            for model, counts, marker in models:
                per_anchor = (np.array(counts)[sizes > 0] / sizes[sizes > 0]).tolist()
                for normalized, values, targets in ((False, counts, ys), (True, per_anchor, ys_per_anchor)):
                    fit = power_law_fit(values, targets)
                    fit_text = "too few positive counts or no x variation"
                    if fit:
                        b, b_low, b_high, _, k, zeros = fit
                        fit_text = (
                            f"b = {b:.2f}, 95% CI [{b_low:.2f}, {b_high:.2f}], {BAND_SHARE:.0%} within ×{k:.2g}"
                            + (f", {zeros} with a 0 left out" if zeros else "")
                        )
                    if normalized:
                        ax.plot([], [], linestyle="none", label=f"    counts per anchor:\n        {fit_text}")
                    else:
                        ax.plot([], [], **marker, label=f"{model} model, raw counts: {fit_text}")
        if np.any(sizes == 0):
            ax.plot(
                [], [], linestyle="none", label=f"per-anchor scores omit {np.sum(sizes == 0)} circuits without anchors"
            )

        colorbar = fig.colorbar(points, ax=ax, ticks=QUBIT_TICKS, fraction=0.05, pad=0.03)
        colorbar.set_label("qubits in circuit", color=TEXT_SECONDARY, fontsize=10)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)
        style(fig, ax, f"{config}: {title}", subtitle, x_label, y_label)
        for text in ax.get_legend().get_texts():
            if text.get_text() in headers:
                text.set_fontweight("bold")
                text.set_color(TEXT)
        # Shorten connectors in display coordinates so they stop at the ring, including on log axes.
        ring_radius = (math.sqrt(RING_SIZE) / 2 + RING_WIDTH / 2) * fig.dpi / 72
        for i, (x, x_per_gate, y) in enumerate(zip(px, px_per_gate, py)):
            (screen_x, screen_y), (screen_per_gate, _) = ax.transData.transform([(x, y), (x_per_gate, y)])
            distance = screen_per_gate - screen_x
            if abs(distance) > ring_radius:
                end = ax.transData.inverted().transform(
                    (screen_per_gate - math.copysign(ring_radius, distance), screen_y)
                )[0]
                ax.plot(
                    [x, end], [y, y], color=BASELINE, linewidth=0.8, solid_capstyle="butt", zorder=4 + i * 3e-5 + 1e-5
                )
        fig.savefig(f"{file_prefix}__{name}.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)

    with open(f"{file_prefix}__aggregated.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(sorted(scores, key=lambda c: c["circuit"]))


def f1(predicted: set[tuple[int, int]], actual: set[tuple[int, int]], tolerance: int) -> float:
    """Harmonic mean of slot precision and recall within ±tolerance anchors on the same qubit.

    Matches may be reused: this measures occupied-slot coverage, ignoring event count and kind.
    Both empty is undefined; one empty is zero agreement.
    """
    if not predicted and not actual:
        return float("nan")
    if not predicted or not actual:
        return 0.0
    precision = sum(any((q, a + d) in actual for d in range(-tolerance, tolerance + 1)) for q, a in predicted) / len(
        predicted
    )
    recall = sum(any((q, a + d) in predicted for d in range(-tolerance, tolerance + 1)) for q, a in actual) / len(
        actual
    )
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def align(abstract: list[int], compiled: list[int], gap: float = ALIGNMENT_GAP) -> tuple[list[int], int, int]:
    """Pair sorted anchor sequences in order, minimizing |shift| plus `gap` per unpaired event.

    Shifts are abstract − compiled. Pairing wins ties at 2 * gap anchors.
    Returns shifts, unpaired abstract count, and unpaired compiled count.
    """
    n, m = len(abstract), len(compiled)
    cost = np.zeros((n + 1, m + 1))
    cost[:, 0], cost[0, :] = np.arange(n + 1) * gap, np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(
                cost[i - 1, j - 1] + abs(abstract[i - 1] - compiled[j - 1]), cost[i - 1, j] + gap, cost[i, j - 1] + gap
            )
    shifts, i, j = [], n, m
    while i and j:
        if cost[i, j] == cost[i - 1, j - 1] + abs(abstract[i - 1] - compiled[j - 1]):
            shifts.append(abstract[i - 1] - compiled[j - 1])
            i, j = i - 1, j - 1
        elif cost[i, j] == cost[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return shifts, n - len(shifts), m - len(shifts)


def visualize_distribution(platform: str, config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    comparable: tuple[str, ...]
    proxy: tuple[str, ...]
    movement: tuple[str, ...]
    if platform == "TI":
        comparable = proxy = TI_IZ_CHANGES
        movement = TI_ION_MOVES
        event, event_key, move = "IZ change", "iz_changes", "ion moves"
    elif platform == "NA":
        comparable, proxy = NA_COMPARABLE_ZONE_CHANGES, NA_ZONE_CHANGES
        movement = ("transports",)
        event, event_key, move = "zone change", "zone_changes", "atom transports"
    else:
        comparable = proxy = ("virtual_swap",)
        movement = ("swaps",)
        event, event_key, move = "swap", "swaps", "swaps"

    scores = []
    shifts, missing, extra = [], 0, 0
    qubit_shifts_per_circuit = []
    qubit_shares: list[tuple[float, float, int]] = []
    amount_circuits = []
    rows_of = {rows[0]["circuit"]: rows for rows in circuits}
    nan = float("nan")
    for rows in circuits:
        num_qubits = int(rows[0]["num_qubits"])
        changes: dict[str, Counter[tuple[int, int]]] = {"abstract": Counter(), "compiled": Counter()}
        abstract_events: Counter[tuple[int, int]] = Counter()
        moves: Counter[tuple[int, int]] = Counter()
        events: defaultdict[tuple[str, str, int], list[int]] = defaultdict(list)
        anchors = [0] * num_qubits
        has_anchors = False
        for row in rows:
            source, metric = row["source"], row["metric"]
            if source == "circuit" and metric == "anchors":
                anchors[int(row["qubit"])] = int(row["value"])
                has_anchors = True
            elif row["count"] and source in changes:
                slot = (int(row["qubit"]), int(row["anchor"]))
                if metric in comparable:
                    count = int(row["count"])
                    changes[source][slot] += count
                    events[(source, metric, slot[0])].extend([slot[1]] * count)
                if source == "abstract" and metric in proxy:
                    abstract_events[slot] += int(row["count"])
                elif source == "compiled" and metric in movement:
                    moves[slot] += int(row["count"])
        if not has_anchors:
            print(f"Skipped {config} / {rows[0]['circuit']}: no anchor rows in the raw file (re-run the runner)")
            continue

        circuit_shifts, circuit_missing, circuit_extra = [], 0, 0
        if platform != "SC":
            for kind in comparable:
                for qubit in range(num_qubits):
                    paired, unpaired_abstract, unpaired_compiled = align(
                        sorted(events[("abstract", kind, qubit)]), sorted(events[("compiled", kind, qubit)])
                    )
                    circuit_shifts.extend(paired)
                    circuit_extra += unpaired_abstract
                    circuit_missing += unpaired_compiled
            shifts.extend(circuit_shifts)
            missing += circuit_missing
            extra += circuit_extra

        # Preserve the noise-comparison window: slots before each qubit's last anchor.
        abstract_per_qubit = [[abstract_events[(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
        moves_per_qubit = [[moves[(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
        qubit_shifts = [
            float(np.average(range(len(a)), weights=a) - np.average(range(len(m)), weights=m))
            for a, m in zip(abstract_per_qubit, moves_per_qubit)
            if sum(a) and sum(m)
        ]
        if qubit_shifts:
            qubit_shifts_per_circuit.append(qubit_shifts)
        abstract_total, moves_total = sum(map(sum, abstract_per_qubit)), sum(map(sum, moves_per_qubit))
        if abstract_total and moves_total:
            amount_circuits.append(rows)
            qubit_shares.extend(
                (sum(m) / moves_total, sum(a) / abstract_total, num_qubits)
                for a, m in zip(abstract_per_qubit, moves_per_qubit)
                if a
            )

        score = {"circuit": rows[0]["circuit"], "num_qubits": num_qubits}
        if platform == "SC":
            score.update(
                abstract_swaps=sum(changes["abstract"].values()) // 2,
                compiled_swaps=sum(changes["compiled"].values()) // 2,
            )
        else:
            score.update(
                {
                    f"f1_{'exact' if t == 0 else f'pm{t}'}": f1(set(changes["abstract"]), set(changes["compiled"]), t)
                    for t in F1_TOLERANCES
                }
            )
            score.update(
                {
                    f"paired_{event_key}": len(circuit_shifts),
                    "share_on_time": circuit_shifts.count(0) / len(circuit_shifts) if circuit_shifts else nan,
                    "mean_shift": float(np.mean(circuit_shifts)) if circuit_shifts else nan,
                    f"missing_{event_key}": circuit_missing,
                    f"extra_{event_key}": circuit_extra,
                }
            )
        score.update(
            {
                "qubits_early": float(np.mean(np.array(qubit_shifts) < -ON_TIME)) if qubit_shifts else nan,
                "qubits_on_time": float(np.mean(np.abs(qubit_shifts) <= ON_TIME)) if qubit_shifts else nan,
                "qubits_late": float(np.mean(np.array(qubit_shifts) > ON_TIME)) if qubit_shifts else nan,
                "event_shift_median": float(np.median(qubit_shifts)) if qubit_shifts else nan,
                "noise_shift_q25": float(np.percentile(qubit_shifts, 25)) if qubit_shifts else nan,
                "noise_shift_q75": float(np.percentile(qubit_shifts, 75)) if qubit_shifts else nan,
                "timed_qubits": len(qubit_shifts),
                "qubits_with_anchors": sum(a > 0 for a in anchors),
            }
        )
        scores.append(score)
    if not scores:
        return
    scores.sort(key=lambda c: (c["num_qubits"], c["circuit"]))

    if platform != "SC":
        plot_location_agreement(config, scores, rows_of, event, file_prefix)
        plot_timing_shift(config, scores, rows_of, event, shifts, missing, extra, file_prefix)
    scored = [c for c in scores if c["timed_qubits"]]
    plot_event_timing(
        config,
        qubit_shifts_per_circuit,
        [rows_of[c["circuit"]] for c in scored],
        move,
        sum(c["timed_qubits"] for c in scores),
        sum(c["qubits_with_anchors"] for c in scores),
        file_prefix,
    )
    abstract_label = "abstract swaps" if platform == "SC" else f"abstract {event}s"
    plot_event_amount(
        qubit_shares,
        f"{config}: event amount vs compiled {move}",
        circuits_subtitle(amount_circuits),
        f"qubit's share of the compiled {move}",
        f"qubit's share of the {abstract_label}",
        "above: more abstract events than its compiled share, below: less",
        f"{file_prefix}__event_amount.png",
        zero_events="swaps" if platform == "SC" else None,
    )

    with open(f"{file_prefix}__distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        for c in sorted(scores, key=lambda c: c["circuit"]):
            writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()})


def plot_location_agreement(config, scores, rows_of, event: str, file_prefix: str) -> None:
    keys = [key for key in scores[0] if key.startswith("f1_")]
    scored = [c for c in scores if not np.isnan(c["f1_exact"])]
    f1s = np.array([[c[k] for k in keys] for c in scored]).reshape(-1, len(keys))
    fig, ax = plt.subplots(figsize=(6, 7), facecolor=SURFACE)
    ax.axhline(1, color=REFERENCE, linewidth=4, zorder=1.6, label=f"abstract and concrete {event}s in the same slots")
    for row in f1s:
        ax.plot(F1_TOLERANCES, row, color=SERIES, alpha=0.18, linewidth=1, zorder=2)
    ax.plot([], [], color=SERIES, alpha=0.4, linewidth=1, label="one circuit")
    if len(f1s):
        ax.fill_between(
            F1_TOLERANCES,
            np.percentile(f1s, 25, axis=0),
            np.percentile(f1s, 75, axis=0),
            color=SERIES,
            alpha=0.15,
            linewidth=0,
            zorder=2,
            label="middle 50% of circuits",
        )
        medians = np.median(f1s, axis=0)
        ax.plot(F1_TOLERANCES, medians, color=SERIES, linewidth=2.5, marker="o", zorder=3, label="median circuit")
        for t, m in zip(F1_TOLERANCES, medians):
            ax.annotate(
                f"{m:.2f}",
                (t, m),
                textcoords="offset points",
                xytext=(0, -16),
                ha="center",
                color=TEXT_SECONDARY,
                fontsize=9,
            )
    ax.set_xticks(F1_TOLERANCES, [f"±{t}" if t else "exact" for t in F1_TOLERANCES])
    ax.set_ylim(0, 1.04)
    style(
        fig,
        ax,
        f"{config}: {event} location agreement",
        circuits_subtitle([rows_of[c["circuit"]] for c in scored]),
        "allowed anchor offset",
        f"F1 score of {event} slots",
        notes=(
            "slot = (qubit, number of completed anchors)",
            f"A / C = slots with ≥ 1 abstract / compiled {event}",
            "precision = share of abstract occupied slots with a compiled match",
            "     within ±offset on the same qubit",
            "recall = share of compiled occupied slots with an abstract match",
            "     within ±offset on the same qubit (matches may be reused)",
            "F1 = 2 · precision · recall / (precision + recall)",
        ),
    )
    fig.savefig(f"{file_prefix}__location_agreement.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_timing_shift(
    config, scores, rows_of, event: str, shifts: list[int], missing: int, extra: int, file_prefix: str
) -> None:
    by_shift = Counter(shifts)
    span = max([1, *(abs(s) for s in shifts)])
    values = list(range(-span, span + 1))
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=SURFACE)
    ax.bar(values, [by_shift[v] for v in values], width=0.8, color=SERIES, zorder=3, label=f"paired {event}s")
    ax.bar(
        [-span - 2, span + 2],
        [missing, extra],
        width=0.8,
        color=MUTED,
        zorder=3,
        label=f"unpaired {event}s: {missing} missing, {extra} extra",
    )
    ax.axvline(0, color=REFERENCE, linewidth=3, zorder=4)
    ax.set_xticks([-span - 2, *values, span + 2], ["missing", *(f"{v:+d}" if v else "0" for v in values), "extra"])
    summary = f"no paired {event}s"
    if shifts:
        summary = (
            f"{len(shifts)} paired: {shifts.count(0) / len(shifts):.0%} on time,"
            f" {sum(s < 0 for s in shifts) / len(shifts):.0%} early, {sum(s > 0 for s in shifts) / len(shifts):.0%} late,"
            f" mean shift {np.mean(shifts):+.2f} anchors"
        )
    style(
        fig,
        ax,
        f"{config}: {event} timing",
        circuits_subtitle([rows_of[c["circuit"]] for c in scores]),
        "timing shift in anchors",
        f"{event}s (all circuits)",
        notes=(
            summary,
            "shift = abstract - compiled anchor, paired by qubit and event kind",
            "< 0: events too early, > 0: too late",
            "missing: unmatched compiled event, extra: unmatched abstract event",
            f"pairing allows shifts up to ±{2 * ALIGNMENT_GAP:g} anchors",
        ),
    )
    fig.savefig(f"{file_prefix}__timing_shift.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_event_timing(
    config: str,
    qubit_shifts_per_circuit: list[list[float]],
    circuits: list[list[dict]],
    move: str,
    timed_qubits: int,
    active_qubits: int,
    file_prefix: str,
) -> None:
    edges = [-np.inf, -2, -1, -0.5, -ON_TIME, np.nextafter(ON_TIME, np.inf), 0.5, 1, 2, np.inf]
    bin_labels = ["< -2", "-2…-1", "-1…-½", "−½…−¼", "±¼", "¼…½", "½…1", "1…2", "≥ 2"]
    center = bin_labels.index("±¼")
    per_circuit = [np.histogram(s, edges)[0] / len(s) for s in qubit_shifts_per_circuit]
    positions = np.arange(len(bin_labels))
    jitter = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    ax.axvline(center, color=REFERENCE, linewidth=4, zorder=1.6, label=f"weight center of the qubit's compiled {move}")
    if per_circuit:
        ax.bar(
            positions,
            np.mean(per_circuit, axis=0),
            width=0.7,
            color=SERIES,
            alpha=0.25,
            zorder=2,
            label="average circuit",
        )
        for shares in per_circuit:
            drawn = shares > 0
            ax.scatter(
                positions[drawn] + jitter.uniform(-0.22, 0.22, drawn.sum()),
                shares[drawn],
                s=18,
                color=SERIES,
                alpha=0.8,
                edgecolors=SURFACE,
                linewidths=0.6,
                zorder=3,
            )
    ax.scatter([], [], s=18, color=SERIES, label="circuit")
    ax.set_xticks(positions, bin_labels)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, 1.04)
    style(
        fig,
        ax,
        f"{config}: event timing, abstract vs compiled",
        circuits_subtitle(circuits),
        "mean abstract anchor − mean compiled movement anchor",
        "share of qubits with both event types",
        notes=(
            f"{timed_qubits} / {active_qubits} qubits with anchors have both event types",
            "events before each qubit's last anchor; circuits weighted equally",
            "< 0: abstract events too early on a qubit, > 0: too late",
        ),
    )
    fig.savefig(f"{file_prefix}__event_timing.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_event_amount(
    qubit_shares: list[tuple[float, float, int]],
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    above_note: str,
    file: str,
    zero_events: str | None = None,
) -> None:
    shares = np.array([(m, a) for m, a, _ in qubit_shares]).reshape(-1, 2)
    qubits = np.array([n for _, _, n in qubit_shares])
    nonzero = shares[shares > 0]
    lo = nonzero.min() / 1.3 if nonzero.size else 0.01
    has_zeros = bool((shares == 0).any())
    zero_at, end = lo / 2.2, 1.3
    start = lo / 3.2 if has_zeros else lo
    rng = np.random.default_rng(0)
    plotted = np.where(shares > 0, shares, zero_at * rng.uniform(0.85, 1.18, shares.shape))

    fig, ax = plt.subplots(figsize=(6.6, 6.9) if zero_events else (6.6, 6), facecolor=SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    diagonal = np.array([zero_at if has_zeros else lo, end])
    ax.fill_between(
        diagonal,
        diagonal / EQUIVALENCE_FACTOR,
        diagonal * EQUIVALENCE_FACTOR,
        color=REFERENCE,
        alpha=0.12,
        linewidth=0,
        zorder=1,
        label=f"within {EQUIVALENCE_FACTOR:g}× of its compiled share",
    )
    ax.plot(diagonal, diagonal, color=REFERENCE, linewidth=3, zorder=2, label="exact match")
    for edge in (ax.axhline, ax.axvline) if has_zeros else ():
        edge(lo / 1.4, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1.5)

    order = np.argsort(-qubits)
    points = ax.scatter(
        plotted[order, 0],
        plotted[order, 1],
        s=26,
        c=qubits[order],
        cmap=QUBIT_CMAP,
        vmin=QUBIT_MIN,
        vmax=QUBIT_MAX,
        alpha=0.8,
        edgecolors=SURFACE,
        linewidths=0.6,
        zorder=3,
    )
    colorbar = fig.colorbar(points, ax=ax, ticks=QUBIT_TICKS, fraction=0.05, pad=0.03)
    colorbar.set_label("qubits in circuit", color=TEXT_SECONDARY, fontsize=10)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)

    ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1) if t >= lo]
    for set_lim, set_ticks in ((ax.set_xlim, ax.set_xticks), (ax.set_ylim, ax.set_yticks)):
        set_lim(start, end)
        set_ticks(
            [zero_at, *ticks] if has_zeros else ticks, [*(["0"] if has_zeros else []), *(f"{t:g}" for t in ticks)]
        )
    ax.minorticks_off()

    notes = [f"number of qubits = {len(shares)}"]
    zero_header = f"Zero shares: the qubit has no {zero_events} of that kind in its circuit"
    if len(shares):
        m, a = shares[:, 0], shares[:, 1]
        notes.append(
            f"> {EQUIVALENCE_FACTOR:g}× its compiled share: {np.mean(a > EQUIVALENCE_FACTOR * m):.0%}, "
            f"< 1/{EQUIVALENCE_FACTOR:g}×: {np.mean(EQUIVALENCE_FACTOR * a < m):.0%}"
            + ("" if zero_events else f",   zero share: {np.mean((m == 0) | (a == 0)):.0%}")
        )
    notes += [
        above_note,
        "one data point = one qubit, shares within its circuit (log scale)",
        "events before each qubit's last anchor; qubits with ≥ 1 anchor",
    ]
    if len(shares) and zero_events:
        notes += [
            zero_header,
            f"bottom row (missed): compiled {zero_events}, but no abstract ones: {np.mean((m > 0) & (a == 0)):.0%}",
            f"left column (invented): abstract {zero_events}, but no compiled ones: {np.mean((m == 0) & (a > 0)):.0%}",
            f"corner (correct): neither, the qubit is not involved: {np.mean((m == 0) & (a == 0)):.0%}",
        ]
    style(fig, ax, title, subtitle, x_label, y_label, notes=tuple(notes))
    for text in ax.get_legend().get_texts():
        if text.get_text() == zero_header:
            text.set_fontweight("bold")
            text.set_color(TEXT)
    fig.savefig(file, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
