"""Plot the summaries of abstract hardware events, compiled events and physical movement."""

import csv
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

from benchmarks._common_reference.plotting import SURFACE, TEXT, TEXT_SECONDARY, MUTED, GRID, SERIES, REFERENCE, save
from benchmarks.model_accuracy._summarizer import (
    ALIGNMENT_GAP,
    BAND_SHARE,
    F1_TOLERANCES,
    ON_TIME,
    compiler_groups,
    correlation,
    lin_ccc,
    power_law_fit,
)

BASELINE, FIT =  "#c3c2b7", "#0d366b"
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
EQUIVALENCE_FACTOR = 1.5


def visualize(platform: list[str], results_dir: Path) -> None:
    for hardware in ("TI", "NA", "SC"):
        if hardware not in platform:
            continue
        for file in sorted((results_dir / "plots" / hardware.lower()).glob("*__aggregated.csv")):
            file_prefix = str(file).removesuffix("__aggregated.csv")
            scores = read(file)
            visualize_aggregated(hardware, scores, file_prefix)
            visualize_distribution(hardware, scores, file_prefix)


def read(file) -> list[dict]:
    """Rows of a summary csv with numbers parsed (empty = nan)."""
    with open(file, newline="") as f:
        return [
            {
                k: v if k in ("config", "compiler", "circuit") else int(v) if v.lstrip("-").isdigit() else float(v or "nan")
                for k, v in row.items()
            }
            for row in csv.DictReader(f)
        ]


def circuits_subtitle(scores: list[dict]) -> str:
    speedups = defaultdict(list)
    for c in scores:
        speedups[c["compiler"]].append(c["compiled_wall_clock_s"] / c["abstract_wall_clock_s"])
    means = {
        compiler: float(np.mean(s))
        for compiler, s in sorted(speedups.items(), key=lambda item: (item[0] != "QMAP", item[0]))
    }
    # plain numbers, no scientific notation: 3 significant digits below 100, whole numbers with thousands separators above
    speedup = ", ".join(f"{c} {m:,.0f}×" if m >= 100 else f"{c} {m:.3g}×" for c, m in means.items()) or "n/a"
    circuit_names = {c["circuit"] for c in scores}
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


def log_count_range(counts: list[int]) -> tuple[float, float, float | None]:
    lo = min([c for c in counts if c > 0], default=1) / 1.3
    return lo, max([*counts, 1]) * 1.3, lo / 2.2 if 0 in counts else None


def visualize_aggregated(platform: str, scores: list[dict], file_prefix: str) -> None:
    config = scores[0]["config"]
    scores = sorted(scores, key=lambda c: c["num_qubits"])
    if platform == "TI":
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
        plots = [("swaps", "virtual swaps", "swaps", "swaps", "abstract virtual swaps", "compiled swaps", False)]

    groups = compiler_groups(scores, lambda c: c["compiler"])
    subtitle = circuits_subtitle(scores)

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


def visualize_distribution(platform: str, aggregated: list[dict], file_prefix: str) -> None:
    if platform == "TI":
        event, move = "IZ change", "ion moves"
    elif platform == "NA":
        event, move = "zone change", "atom transports"
    else:
        event, move = "swap", "swaps"
    if not Path(f"{file_prefix}__distribution.csv").exists():
        return

    config = aggregated[0]["config"]
    shifts = read(f"{file_prefix}__timing_shifts.csv") if platform != "SC" else []
    qubits = read(f"{file_prefix}__qubits.csv")
    results = []  # per compiler: (compiler, g, results of its circuits)
    for compiler, g, scores in compiler_groups(read(f"{file_prefix}__distribution.csv"), lambda c: c["compiler"]):
        qubit_shifts_per_circuit = defaultdict(list)
        for q in qubits:
            if q["compiler"] == compiler and not np.isnan(q["timing_shift"]):
                qubit_shifts_per_circuit[q["circuit"]].append(q["timing_shift"])
        results.append(
            (
                compiler,
                g,
                dict(
                    scores=sorted(scores, key=lambda c: (c["num_qubits"], c["circuit"])),
                    shifts=[s["shift"] for s in shifts if s["compiler"] == compiler],
                    missing=sum(v for c in scores for k, v in c.items() if k.startswith("missing_")),
                    extra=sum(v for c in scores for k, v in c.items() if k.startswith("extra_")),
                    qubit_shifts_per_circuit=list(qubit_shifts_per_circuit.values()),
                    qubit_shares=[
                        (q["moves_share"], q["abstract_share"], q["num_qubits"])
                        for q in qubits
                        if q["compiler"] == compiler and not np.isnan(q["moves_share"])
                    ],
                ),
            )
        )
    amount_circuits = {(q["circuit"], q["compiler"]) for q in qubits if not np.isnan(q["moves_share"])}

    subtitle = circuits_subtitle(aggregated)
    if platform != "SC":
        plot_location_agreement(config, subtitle, results, event, file_prefix)
        plot_timing_shift(config, subtitle, results, event, file_prefix)
    plot_event_timing(config, subtitle, results, move, file_prefix)
    abstract_label = "abstract swaps" if platform == "SC" else f"abstract {event}s"
    plot_event_amount(
        results,
        f"{config}: event amount vs compiled {move}",
        circuits_subtitle([c for c in aggregated if (c["circuit"], c["compiler"]) in amount_circuits]),
        f"qubit's share of the compiled {move}",
        f"qubit's share of the {abstract_label}",
        f"{file_prefix}__event_amount.png",
        zero_events="swaps" if platform == "SC" else None,
    )


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
