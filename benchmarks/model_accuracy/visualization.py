"""Compare abstract hardware events with compiled events and physical movement.

Aggregated event counts assess agreement or proxy scaling (correlation + proportionality).
Location scores compare occupied slots; timing alignment also distinguishes event kinds.
Every plot shows all compilers of a platform, each in its own color and marker, with its scores in its own box.
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, PaddedBox, TextArea, VPacker
from scipy.stats import kendalltau, spearmanr, theilslopes

SURFACE, TEXT, TEXT_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SERIES, FIT, REFERENCE = "#e1e0d9", "#c3c2b7", "#2a78d6", "#0d366b", "#e34948"
PER_GATE_COLORS = ["#13917f", "#86a01b"]  # per-gate model per compiler: cool teal-green, warm olive-green
RING_SIZE, RING_WIDTH = 40, 1.4
# one color ramp (shade = qubits in circuit), color and marker per compiler, with QMAP first (blue)
COMPILER_CMAPS = [
    LinearSegmentedColormap.from_list(
        "qubits",
        ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
    ),
    LinearSegmentedColormap.from_list(
        "qubits_violet",
        ["#d39ae0", "#c784d6", "#bb6ecc", "#ad55bf", "#9c2fae", "#8a279a", "#772185", "#641b70", "#51165b"],
    ),
]
COMPILER_COLORS = [SERIES, "#9c2fae"]
COMPILER_FITS = [FIT, "#9c2fae"]  # dark navy vs mid violet, so the fit lines stay apart
COMPILER_MARKERS = ["o", "s"]
COMPILER_GLYPHS = ["●", "■"]
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

TI_IZ_CHANGES = ("interaction_zone_entry", "interaction_zone_eviction")
TI_ION_MOVES = ("shuttles", "swaps")  # swaps count once per participating ion
NA_ZONE_CHANGES = ("storage_entanglement", "storage_readout", "entanglement_readout", "inter_entanglement")


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


def compiler_groups(items: list, compiler_of) -> list[tuple[str, int, list]]:
    """Split items by compiler: (compiler, index for its color / marker, items), with QMAP first."""
    compilers = sorted({compiler_of(item) for item in items}, key=lambda compiler: (compiler != "QMAP", compiler))
    return [(compiler, g, [item for item in items if compiler_of(item) == compiler]) for g, compiler in enumerate(compilers)]


def circuits_subtitle(circuits: list[list[dict]]) -> str:
    speedups = defaultdict(list)
    for rows in circuits:
        wall_clock_s = {row["source"]: float(row["value"]) for row in rows if row["metric"] == "wall_clock_s"}
        speedups[rows[0]["compiler"]].append(wall_clock_s["compiled"] / wall_clock_s["abstract"])
    means = {
        compiler: float(np.mean(s))
        for compiler, s in sorted(speedups.items(), key=lambda item: (item[0] != "QMAP", item[0]))
    }
    # plain numbers, no scientific notation: 3 significant digits below 100, whole numbers with thousands separators above
    speedup = ", ".join(f"{c} {m:,.0f}×" if m >= 100 else f"{c} {m:.3g}×" for c, m in means.items()) or "n/a"
    circuit_names = {rows[0]["circuit"] for rows in circuits}
    return f"number of circuits = {len(circuit_names)}   mean speedup (compile / abstract time): {speedup}"


def compiler_box(compiler: str, g: int, lines: list[str]) -> PaddedBox:
    """A framed score box of one compiler: its marker and name, then monospaced lines."""
    title = TextArea(
        f"{COMPILER_GLYPHS[g]} {compiler}", textprops=dict(color=COMPILER_COLORS[g], fontsize=9.5, fontweight="bold")
    )
    body = TextArea("\n".join(lines), textprops=dict(color=TEXT_SECONDARY, fontsize=7.5, family="monospace"))
    return PaddedBox(
        VPacker(children=[title, body], sep=4, pad=0, align="left"),
        pad=6,
        draw_frame=True,
        patch_attrs=dict(facecolor=SURFACE, edgecolor=COMPILER_COLORS[g], linewidth=1),
    )


def style(
    fig, ax, title: str, subtitle: str, x_label: str, y_label: str, sections=(), notes: tuple[str, ...] = (), compilers=()
) -> None:
    """Style the axes, then put a compact legend, the sections (bold header or None, compiler boxes side by side),
    and the notes below it. Legend entries labelled "<compiler>: ..." form one column per compiler (in `compilers` order),
    the shared entries (reference lines) head the columns."""
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, va="bottom", ha="left", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlabel(x_label, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=10)
    fig.tight_layout()

    # everything below the axes is placed from the measured bottom of the previous block (saved with bbox_inches="tight")
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    left = to_figure.transform(ax.get_tightbbox(renderer).p0)[0]
    bottom = to_figure.transform(ax.get_tightbbox(renderer).p0)[1] - 0.01
    handles, labels = ax.get_legend_handles_labels()
    columns = [[label for label in labels if label.startswith(f"{compiler}: ")] for compiler in compilers]
    shared = [label for label in labels if not any(label in column for column in columns)]
    handle_of = dict(zip(labels, handles))
    if len(columns) == 2:  # filled column by column: shared entries pairwise on top, then QMAP left, other compiler right
        if len(shared) % 2:  # an invisible blank entry next to an odd shared entry keeps the compiler rows aligned
            shared, handle_of[""] = [*shared, ""], Line2D([], [], linestyle="none")
        order = shared[0::2] + columns[0] + shared[1::2] + columns[1]
    else:
        order = shared + [label for column in columns for label in column]
    handles, labels = [handle_of[label] for label in order], order
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(left, bottom),
            frameon=False,
            fontsize=8.5,
            labelcolor=TEXT_SECONDARY,
            handlelength=2.0,
            ncol=2,
            columnspacing=1.8,
            borderaxespad=0,
        )
        bottom = to_figure.transform(legend.get_window_extent(renderer).p0)[1] - 0.015
    blocks = []
    for header, boxes in sections:
        if header:
            blocks.append(TextArea(header, textprops=dict(color=TEXT, fontsize=9, fontweight="bold")))
        blocks.append(HPacker(children=list(boxes), sep=10, pad=0, align="top"))
    if notes:
        blocks.append(TextArea("\n".join(notes), textprops=dict(color=MUTED, fontsize=7.5)))
    if blocks:
        fig.add_artist(
            AnchoredOffsetbox(
                "upper left",
                child=VPacker(children=blocks, sep=8, pad=0, align="left"),
                bbox_to_anchor=(left, bottom),
                bbox_transform=fig.transFigure,
                frameon=False,
                pad=0,
                borderpad=0,
            )
        )


def save(fig, file: str) -> None:
    fig.savefig(file, dpi=DPI, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def qubit_colorbars(fig, ax, mappables: list[tuple[str, int, ScalarMappable]]) -> None:
    """Thin equal colorbars side by side right of the axes, marked by the compiler's glyph; only the last one is labelled."""
    for _, g, points in mappables:
        cax = ax.inset_axes((1.03 + 0.05 * g, 0.15, 0.025, 0.7))
        colorbar = fig.colorbar(points, cax=cax, ticks=QUBIT_TICKS)
        colorbar.outline.set_visible(False)
        cax.set_title(COMPILER_GLYPHS[g], color=COMPILER_COLORS[g], fontsize=9)
        last = g == len(mappables) - 1
        cax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=8, labelright=last, length=3 if last else 0)
        if last:
            colorbar.set_label("qubits in circuit", color=TEXT_SECONDARY, fontsize=9)


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


def correlation(xs: list[float], ys: list[float], rng: np.random.Generator) -> tuple[float, float, float, float] | None:
    """Kendall τ-b with a circuit-bootstrap percentile CI, and Spearman ρ (including zeros)."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    tau = kendalltau(xs, ys).statistic
    resamples = (rng.integers(0, len(xs), len(xs)) for _ in range(BOOTSTRAP_SAMPLES))
    low, high = np.nanpercentile([kendalltau(xs[i], ys[i]).statistic for i in resamples], [2.5, 97.5])
    return tau, low, high, spearmanr(xs, ys).statistic


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
            ("iz_entries", "IZ entries", "iz_entries", "iz_entries", "abstract IZ entries", "compiled IZ entries", True),
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
        metrics = {"zone_changes": NA_ZONE_CHANGES}
        compiled_metrics = {**metrics, "atom_moves": ("transports",)}
        plots = [
            (
                "zone_changes",
                "zone changes",
                "zone_changes",
                "zone_changes",
                "abstract zone changes",
                "compiled zone changes",
                True,
            ),
            (
                "zone_changes_vs_atom_moves",
                "zone changes vs atom moves",
                "zone_changes",
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
            "compiler": rows[0]["compiler"],
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
    groups = compiler_groups(scores, lambda c: c["compiler"])
    subtitle = circuits_subtitle(circuits)

    for name, title, x_key, y_key, x_label, y_label, same_quantity in plots:
        all_xs = [c[f"{source}_{x_key}"] for c in scores for source in ("abstract", "per_gate")]
        all_ys = [c[f"compiled_{y_key}"] for c in scores]
        fig, ax = plt.subplots(figsize=(7.6, 6.2), facecolor=SURFACE)
        x_range = log_count_range([*all_xs, *(all_ys if same_quantity else [])])
        y_range = x_range if same_quantity else log_count_range(all_ys)
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

        if same_quantity:
            diagonal = np.array([x_zero or x_lo, x_end])
            ax.plot(diagonal, diagonal, color=REFERENCE, linewidth=3, zorder=2, label="exact match (y = x)")
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

        rng = np.random.default_rng(0)
        if same_quantity:
            sections = [("Agreement: how close are the circuits to y = x?", [])]
        else:
            sections = [
                ("Correlation: is the order of the circuits the same?", []),
                ("Scaling: log-log fit y = c · x^b (b = 1: proportional)", []),
            ]
        mappables = []
        for compiler, g, group in groups:
            xs = [c[f"abstract_{x_key}"] for c in group]
            xs_per_gate = [c[f"per_gate_{x_key}"] for c in group]
            ys = [c[f"compiled_{y_key}"] for c in group]
            sizes = np.array([c["anchors"] for c in group])
            # Only drawing positions use the zero strip; statistics use the original counts.
            px = [x if x else x_zero for x in xs]
            px_per_gate = [x if x else x_zero for x in xs_per_gate]
            py = [y if y else y_zero for y in ys]
            marker = COMPILER_MARKERS[g]
            ax.plot([], [], marker=marker, markersize=7, linestyle="none", color=COMPILER_COLORS[g], alpha=0.7,
                    label=f"{compiler}: abstract model")
            ax.plot([], [], marker=marker, markersize=6, linestyle="none", markerfacecolor="none", alpha=0.7,
                    markeredgecolor=PER_GATE_COLORS[g], markeredgewidth=1.4, label=f"{compiler}: per-gate model")
            ax.scatter(
                px_per_gate,
                py,
                s=RING_SIZE,
                marker=marker,
                facecolors="none",
                edgecolors=PER_GATE_COLORS[g],
                linewidths=RING_WIDTH,
                alpha=0.7,
                zorder=3,
            )
            points = ScalarMappable(Normalize(QUBIT_MIN, QUBIT_MAX), COMPILER_CMAPS[g])
            ax.scatter(
                px,
                py,
                s=56,
                marker=marker,
                c=[points.to_rgba(c["num_qubits"]) for c in group],
                edgecolors="none",
                linewidths=0,
                alpha=0.7,  # overlapping points of both compilers stay visible
                zorder=5 - g,  # first compiler (smaller circles) on top
            )
            mappables.append((compiler, g, points))

            ys_per_anchor = (np.array(ys)[sizes > 0] / sizes[sizes > 0]).tolist()
            models = (("abstract", xs), ("per-gate", xs_per_gate))
            if same_quantity:
                lines = [f"{'':<10}{'rel.err':>8}{'≤' + f'{EQUIVALENCE_FACTOR:g}×':>7}{'ratio':>8}{'CCC':>6}{'/anchor':>9}"]
                zeros = 0
                for model, counts in models:
                    errors = [abs(x - y) / y for x, y in zip(counts, ys) if y > 0]
                    within = np.mean(
                        [x <= EQUIVALENCE_FACTOR * y and y <= EQUIVALENCE_FACTOR * x for x, y in zip(counts, ys)]
                    )
                    ratios = [x / y for x, y in zip(counts, ys) if y > 0]
                    per_anchor = (np.array(counts)[sizes > 0] / sizes[sizes > 0]).tolist()
                    ccc = [lin_ccc(counts, ys), lin_ccc(per_anchor, ys_per_anchor)]
                    zeros = max(zeros, sum(x == 0 or y == 0 for x, y in zip(counts, ys)))
                    lines.append(
                        f"{model:<10}"
                        + (f"{np.median(errors):>8.0%}" if errors else f"{'–':>8}")
                        + f"{within:>7.0%}"
                        + (f"{'×' + f'{np.median(ratios):.2f}':>8}" if ratios else f"{'–':>8}")
                        + "".join(f"{'–' if v is None else f'{v:.2f}':>{w}}" for v, w in zip(ccc, (6, 9)))
                    )
                if zeros:
                    lines.append(f"CCC leaves out up to {zeros} circuits with a 0")
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
                        color=COMPILER_FITS[g],
                        alpha=0.08,
                        linewidth=0,
                        zorder=1,
                    )
                    ax.plot(
                        fit_xs,
                        np.where(fit_ys >= y_lo, fit_ys, np.nan),
                        color=COMPILER_FITS[g],
                        alpha=0.85,
                        linewidth=2,
                        zorder=2,
                        label=f"{compiler}: fit y = {c:.2g} · x^{b:.2f}, {BAND_SHARE:.0%} within ×{k:.2g}",
                    )
                    ratio = np.median([y / x for x, y in zip(xs, ys) if x > 0 and y > 0])
                    ax.plot(
                        fit_xs,
                        np.where(ratio * fit_xs >= y_lo, ratio * fit_xs, np.nan),
                        color=REFERENCE,
                        linewidth=2.5 if g == 0 else 2,
                        linestyle="solid" if g == 0 else (0, (5, 3)),
                        zorder=2,
                        label=f"{compiler}: proportional, median y / x = {ratio:.2g}",
                    )
                correlation_lines = [f"{'':<10}{'τ [95% CI]':<22}{'ρ':>5}"]
                lines = [f"{'':<10}{'b [95% CI]':<22}{'×' + f'{BAND_SHARE:.0%}':>5}"]
                zeros = 0
                for model, counts in models:
                    per_anchor = (np.array(counts)[sizes > 0] / sizes[sizes > 0]).tolist()
                    for label, values, targets in ((model, counts, ys), ("  /anchor", per_anchor, ys_per_anchor)):
                        corr, fit = correlation(values, targets, rng), power_law_fit(values, targets)
                        correlation_lines.append(
                            f"{label:<10}"
                            + (f"{f'{corr[0]:5.2f} [{corr[1]:.2f}, {corr[2]:.2f}]':<22}{corr[3]:>5.2f}" if corr else "no variation")
                        )
                        lines.append(
                            f"{label:<10}" + (f"{f'{fit[0]:5.2f} [{fit[1]:.2f}, {fit[2]:.2f}]':<22}{fit[4]:>5.2g}" if fit else "–")
                        )
                        zeros = max(zeros, fit[5] if fit else 0)
                if zeros:
                    lines.append(f"fits leave out up to {zeros} circuits with a 0")
                sections[0][1].append(compiler_box(compiler, g, correlation_lines))
            if np.any(sizes == 0):
                lines.append(f"/anchor leaves out {np.sum(sizes == 0)} circuits without anchors")
            sections[-1][1].append(compiler_box(compiler, g, lines))

        if same_quantity:
            notes = (
                "rel.err = median |model − compiled| / compiled,  ratio = median model / compiled (both: compiled > 0)",
                f"≤{EQUIVALENCE_FACTOR:g}× = share of circuits within a factor {EQUIVALENCE_FACTOR:g},"
                "  CCC = Lin's concordance of log counts (raw / per anchor)",
            )
        else:
            notes = (
                "τ = Kendall τ-b (circuit bootstrap 95% CI), ρ = Spearman rank correlation",
                f"b = Theil-Sen slope of log y = log c + b log x (1 = proportional), ×{BAND_SHARE:.0%} = factor around"
                f" the fit holding {BAND_SHARE:.0%} of the circuits",
                "/anchor = counts divided by the anchors of the circuit",
            )
        if platform == "NA":
            notes += ("Totals include readout transitions; compiler adapters do not model readout transport.",)
        qubit_colorbars(fig, ax, mappables)
        compilers = ", ".join(compiler for compiler, *_ in groups)
        style(
            fig, ax, f"{config}: {title}", subtitle, x_label, f"{y_label} ({compilers})", sections, notes,
            compilers=[compiler for compiler, *_ in groups],
        )
        save(fig, f"{file_prefix}__{name}.png")

    with open(f"{file_prefix}__aggregated.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(sorted(scores, key=lambda c: (c["circuit"], c["compiler"])))


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
        comparable = proxy = NA_ZONE_CHANGES
        movement = ("transports",)
        event, event_key, move = "zone change", "zone_changes", "atom transports"
    else:
        comparable = proxy = ("virtual_swap",)
        movement = ("swaps",)
        event, event_key, move = "swap", "swaps", "swaps"

    results = []  # per compiler: (compiler, g, results of its circuits)
    nan = float("nan")
    for compiler, g, compiler_circuits in compiler_groups(circuits, lambda rows: rows[0]["compiler"]):
        scores = []
        shifts, missing, extra = [], 0, 0
        qubit_shifts_per_circuit = []
        qubit_shares: list[tuple[float, float, int]] = []
        amount_circuits = []
        for rows in compiler_circuits:
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
                print(f"Skipped {config} / {rows[0]['circuit']} / {compiler}: no anchor rows in the raw file (re-run the runner)")
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

            score = {"compiler": compiler, "circuit": rows[0]["circuit"], "num_qubits": num_qubits}
            if platform == "SC":
                score.update(
                    abstract_swaps=sum(changes["abstract"].values()) // 2,
                    compiled_swaps=sum(changes["compiled"].values()) // 2,
                )
            else:
                score.update(
                    {
                        f"f1_{'exact' if t == 0 else f'pm{t}'}": f1(
                            set(changes["abstract"]), set(changes["compiled"]), t
                        )
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
        if scores:
            scores.sort(key=lambda c: (c["num_qubits"], c["circuit"]))
            results.append(
                (
                    compiler,
                    g,
                    dict(
                        scores=scores,
                        shifts=shifts,
                        missing=missing,
                        extra=extra,
                        qubit_shifts_per_circuit=qubit_shifts_per_circuit,
                        qubit_shares=qubit_shares,
                        amount_circuits=amount_circuits,
                    ),
                )
            )
    if not results:
        return

    subtitle = circuits_subtitle(circuits)
    if platform != "SC":
        plot_location_agreement(config, subtitle, results, event, file_prefix)
        plot_timing_shift(config, subtitle, results, event, file_prefix)
    plot_event_timing(config, subtitle, results, move, file_prefix)
    abstract_label = "abstract swaps" if platform == "SC" else f"abstract {event}s"
    plot_event_amount(
        results,
        f"{config}: event amount vs compiled {move}",
        circuits_subtitle([rows for _, _, result in results for rows in result["amount_circuits"]]),
        f"qubit's share of the compiled {move}",
        f"qubit's share of the {abstract_label}",
        f"{file_prefix}__event_amount.png",
        zero_events="swaps" if platform == "SC" else None,
    )

    all_scores = [c for _, _, result in results for c in result["scores"]]
    with open(f"{file_prefix}__distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_scores[0]))
        writer.writeheader()
        for c in sorted(all_scores, key=lambda c: (c["circuit"], c["compiler"])):
            writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()})


def plot_location_agreement(config: str, subtitle: str, results, event: str, file_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    ax.axhline(1, color=REFERENCE, linewidth=4, zorder=1.6, label=f"abstract and compiled {event}s in the same slots")
    boxes = []
    for compiler, g, result in results:
        scored = [c for c in result["scores"] if not np.isnan(c["f1_exact"])]
        keys = [key for key in result["scores"][0] if key.startswith("f1_")]
        f1s = np.array([[c[k] for k in keys] for c in scored]).reshape(-1, len(keys))
        offsets = np.array(F1_TOLERANCES) + (g - (len(results) - 1) / 2) * 0.08  # side by side per compiler
        color = COMPILER_COLORS[g]
        ax.plot([], [], color=color, alpha=0.4, linewidth=1, label=f"{compiler}: one circuit")
        ax.fill_between([], [], color=color, alpha=0.2, linewidth=0, label=f"{compiler}: middle 50% of circuits")
        ax.plot([], [], color=color, linewidth=2.5, marker=COMPILER_MARKERS[g], label=f"{compiler}: median circuit")
        for row in f1s:
            ax.plot(offsets, row, color=color, alpha=0.12, linewidth=1, zorder=2)
        lines = [f"circuits scored: {len(f1s)}"]
        if len(f1s):
            ax.fill_between(
                offsets,
                np.percentile(f1s, 25, axis=0),
                np.percentile(f1s, 75, axis=0),
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
            medians = np.median(f1s, axis=0)
            ax.plot(offsets, medians, color=color, linewidth=2.5, marker=COMPILER_MARKERS[g], zorder=4 - g)
            lines.append("median F1  " + "  ".join(f"{'±' + str(t) if t else 'exact'} {m:.2f}" for t, m in zip(F1_TOLERANCES, medians)))
        boxes.append(compiler_box(compiler, g, lines))
    ax.set_xticks(F1_TOLERANCES, [f"±{t}" if t else "exact" for t in F1_TOLERANCES])
    ax.set_ylim(0, 1.04)
    style(
        fig,
        ax,
        f"{config}: {event} location agreement",
        subtitle,
        "allowed anchor offset",
        f"F1 score of {event} slots",
        [(None, boxes)],
        notes=(
            "slot = (qubit, number of completed anchors); occupied = ≥ 1 event there",
            "precision / recall = share of abstract / compiled occupied slots with a match in the other within ±offset"
            " on the same qubit",
            "(matches may be reused), F1 = 2 · precision · recall / (precision + recall)",
        ),
        compilers=[c for c, _, _ in results],
    )
    save(fig, f"{file_prefix}__location_agreement.png")


def plot_timing_shift(config: str, subtitle: str, results, event: str, file_prefix: str) -> None:
    span = max([1, *(abs(s) for _, _, result in results for s in result["shifts"])])
    values = list(range(-span, span + 1))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(6.8, 4.8), facecolor=SURFACE)
    boxes = []
    for compiler, g, result in results:
        shifts, missing, extra = result["shifts"], result["missing"], result["extra"]
        offset = (g - (len(results) - 1) / 2) * width
        by_shift = Counter(shifts)
        color = COMPILER_COLORS[g]
        ax.fill_between([], [], color=color, linewidth=0, label=f"{compiler}: paired {event}s")
        ax.fill_between([], [], color=color, linewidth=0, alpha=0.45, label=f"{compiler}: unpaired {event}s")
        ax.bar(np.array(values) + offset, [by_shift[v] for v in values], width=width, color=color, zorder=3)
        ax.bar(
            np.array([-span - 2, span + 2]) + offset, [missing, extra], width=width, color=color, alpha=0.45, zorder=3
        )
        lines = [f"no paired {event}s"]
        if shifts:
            lines = [
                f"{len(shifts)} paired: {shifts.count(0) / len(shifts):.0%} on time, {sum(s < 0 for s in shifts) / len(shifts):.0%}"
                f" early, {sum(s > 0 for s in shifts) / len(shifts):.0%} late",
                f"mean shift {np.mean(shifts):+.2f} anchors",
            ]
        lines.append(f"unpaired: {missing} missing, {extra} extra")
        boxes.append(compiler_box(compiler, g, lines))
    ax.axvline(0, color=REFERENCE, linewidth=3, zorder=4)
    ax.set_xticks([-span - 2, *values, span + 2], ["missing", *(f"{v:+d}" if v else "0" for v in values), "extra"])
    style(
        fig,
        ax,
        f"{config}: {event} timing",
        subtitle,
        "timing shift in anchors",
        f"{event}s (all circuits)",
        [(None, boxes)],
        notes=(
            "shift = abstract − compiled anchor, paired by qubit and event kind (< 0: too early, > 0: too late)",
            f"missing / extra = unmatched compiled / abstract event, pairing allows shifts up to ±{2 * ALIGNMENT_GAP:g}"
            " anchors",
        ),
        compilers=[c for c, _, _ in results],
    )
    save(fig, f"{file_prefix}__timing_shift.png")


def plot_event_timing(config: str, subtitle: str, results, move: str, file_prefix: str) -> None:
    edges = [-np.inf, -2, -1, -0.5, -ON_TIME, np.nextafter(ON_TIME, np.inf), 0.5, 1, 2, np.inf]
    bin_labels = ["< -2", "-2…-1", "-1…-½", "−½…−¼", "±¼", "¼…½", "½…1", "1…2", "≥ 2"]
    center = bin_labels.index("±¼")
    positions = np.arange(len(bin_labels))
    width = 0.8 / len(results)
    jitter = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)
    ax.axvline(center, color=REFERENCE, linewidth=4, zorder=1.6, label=f"weight center of the qubit's compiled {move}")
    boxes = []
    for compiler, g, result in results:
        per_circuit = [np.histogram(s, edges)[0] / len(s) for s in result["qubit_shifts_per_circuit"]]
        offset = (g - (len(results) - 1) / 2) * width
        color = COMPILER_COLORS[g]
        ax.fill_between([], [], color=color, linewidth=0, alpha=0.25, label=f"{compiler}: average circuit")
        ax.scatter([], [], s=16, marker=COMPILER_MARKERS[g], color=color, alpha=0.65, label=f"{compiler}: one circuit")
        if per_circuit:
            ax.bar(positions + offset, np.mean(per_circuit, axis=0), width=width, color=color, alpha=0.25, zorder=2)
            for shares in per_circuit:
                drawn = shares > 0
                ax.scatter(
                    positions[drawn] + offset + jitter.uniform(-0.3, 0.3, drawn.sum()) * width,
                    shares[drawn],
                    s=16,
                    marker=COMPILER_MARKERS[g],
                    color=color,
                    alpha=0.65,
                    edgecolors="none",
                    linewidths=0,
                    zorder=3 - 0.1 * g,
                )
        scores = result["scores"]
        timed, active = sum(c["timed_qubits"] for c in scores), sum(c["qubits_with_anchors"] for c in scores)
        lines = [f"{timed} / {active} qubits with anchors have both event types"]
        if per_circuit:
            lines.append(f"average circuit: {np.mean(per_circuit, axis=0)[center]:.0%} of its qubits within ±¼")
        boxes.append(compiler_box(compiler, g, lines))
    ax.set_xticks(positions, bin_labels)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, 1.04)
    style(
        fig,
        ax,
        f"{config}: event timing, abstract vs compiled",
        subtitle,
        "mean abstract anchor − mean compiled movement anchor",
        "share of qubits with both event types",
        [(None, boxes)],
        notes=(
            "events before each qubit's last anchor; circuits weighted equally",
            "< 0: abstract events too early on a qubit, > 0: too late",
        ),
        compilers=[c for c, _, _ in results],
    )
    save(fig, f"{file_prefix}__event_timing.png")


def plot_event_amount(
    results, title: str, subtitle: str, x_label: str, y_label: str, file: str, zero_events: str | None = None
) -> None:
    all_shares = np.array([(m, a) for _, _, result in results for m, a, _ in result["qubit_shares"]]).reshape(-1, 2)
    nonzero = all_shares[all_shares > 0]
    lo = nonzero.min() / 1.3 if nonzero.size else 0.01
    has_zeros = bool((all_shares == 0).any())
    zero_at, end = lo / 2.2, 1.3
    start = lo / 3.2 if has_zeros else lo
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6.6, 5.6), facecolor=SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    diagonal = np.array([zero_at if has_zeros else lo, end])
    ax.plot(diagonal, diagonal, color=REFERENCE, linewidth=3, zorder=2, label="exact match")
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
    for edge in (ax.axhline, ax.axvline) if has_zeros else ():
        edge(lo / 1.4, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1.5)

    boxes, mappables = [], []
    for compiler, g, result in results:
        shares = np.array([(m, a) for m, a, _ in result["qubit_shares"]]).reshape(-1, 2)
        qubits = np.array([n for _, _, n in result["qubit_shares"]])
        plotted = np.where(shares > 0, shares, zero_at * rng.uniform(0.85, 1.18, shares.shape))
        order = np.argsort(-qubits)
        points = ax.scatter(
            plotted[order, 0],
            plotted[order, 1],
            s=22,
            marker=COMPILER_MARKERS[g],
            c=qubits[order],
            cmap=COMPILER_CMAPS[g],
            vmin=QUBIT_MIN,
            vmax=QUBIT_MAX,
            alpha=0.6,
            edgecolors="none",
            linewidths=0,
            zorder=4 - g,
        )
        mappables.append((compiler, g, points))
        ax.scatter([], [], s=22, marker=COMPILER_MARKERS[g], color=COMPILER_COLORS[g], alpha=0.6, label=f"{compiler}: one qubit")
        lines = [f"qubits: {len(shares)}"]
        if len(shares):
            m, a = shares[:, 0], shares[:, 1]
            lines.append(
                f"> {EQUIVALENCE_FACTOR:g}× its compiled share: {np.mean(a > EQUIVALENCE_FACTOR * m):.0%},"
                f" < 1/{EQUIVALENCE_FACTOR:g}×: {np.mean(EQUIVALENCE_FACTOR * a < m):.0%}"
            )
            if zero_events:
                lines += [
                    f"missed (bottom row):   {np.mean((m > 0) & (a == 0)):>4.0%}",
                    f"invented (left col.):  {np.mean((m == 0) & (a > 0)):>4.0%}",
                    f"not involved (corner): {np.mean((m == 0) & (a == 0)):>4.0%}",
                ]
            else:
                lines.append(f"zero share: {np.mean((m == 0) | (a == 0)):.0%}")
        boxes.append(compiler_box(compiler, g, lines))
    qubit_colorbars(fig, ax, mappables)

    ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1) if t >= lo]
    for set_lim, set_ticks in ((ax.set_xlim, ax.set_xticks), (ax.set_ylim, ax.set_yticks)):
        set_lim(start, end)
        set_ticks(
            [zero_at, *ticks] if has_zeros else ticks, [*(["0"] if has_zeros else []), *(f"{t:g}" for t in ticks)]
        )
    ax.minorticks_off()

    notes = [
        "one data point = one qubit, shares within its circuit (log scale); above: more abstract events than its"
        " compiled share",
        "events before each qubit's last anchor; qubits with ≥ 1 anchor",
    ]
    if zero_events:
        notes.append(
            f"zero shares: missed = compiled {zero_events} but no abstract ones, invented = the reverse,"
            " not involved = neither (correct)"
        )
    style(fig, ax, title, subtitle, x_label, y_label, [(None, boxes)], tuple(notes), compilers=[c for c, _, _ in results])
    save(fig, file)
