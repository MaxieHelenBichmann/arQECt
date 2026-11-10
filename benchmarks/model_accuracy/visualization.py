"""Extract and aggregate relevant metrics comparing the abstract execution and concrete compilation, and visualize them.

I want to answer two questions:
A) How well does the abstract hardware model approximate certain concrete hardware events compared to the actually compiled circuit?
B) How well do the abstract hardware events function as a proxy for the relevant concrete hardware events in the compiled circuit?

And "well" is defined as resulting in a suitable (either similar (A) or having a similar scaling behavior (B)) number of hardware events
aggregated over the whole circuit (important for structural cost computation), as well as having a suitable distribution of hardware events
throughout the circuit (important for noise injection).

As a comparison, the aggregated plots also show a naive per-gate model (another, not hardware-aware abstraction): the qubits of every
relevant gate enter the relevant zone before and leave it after the gate (TI: any gate, NA: 2-qubit gates and measurements),
SC: a virtual swap before and after every 2-qubit gate.

All data comes from the raw result files written by the runner (one file per hardware config and circuit).
"""

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from scipy.stats import kendalltau, spearmanr, theilslopes

# colors: surface, text, grid / axes, data series, fit line, reference (optimal outcome) line
SURFACE, TEXT, TEXT_SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SERIES, FIT, REFERENCE = "#e1e0d9", "#c3c2b7", "#2a78d6", "#0d366b", "#e34948"
PER_GATE = "#2f9a5e"  # naive per-gate model (comparison)
RING_SIZE, RING_WIDTH = 40, 1.4  # per-gate model marker: area (pt²) and ring line width (pt)
# blue ramp for the qubit number of a circuit, with a fixed range so the same color means the same qubit number in every plot
QUBIT_CMAP = LinearSegmentedColormap.from_list("qubits", ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                                                          "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
QUBIT_MIN, QUBIT_MAX = 4, 50
QUBIT_TICKS = (4, 10, 20, 30, 40, 50)
DPI = 200

# TI raw result metrics
TI_IZ_CHANGES = ("interaction_zone_entry", "interaction_zone_eviction")
TI_ION_MOVES = ("shuttles", "swaps")  # swaps are counted per ion
F1_TOLERANCES = [0, 1, 2, 3]  # allowed anchor offsets of the location agreement
ON_TIME = 0.25  # a qubit's abstract events are on time if their weight center is within ±ON_TIME anchors of its ion moves

# NA raw result metrics
NA_ZONE_CHANGES = ("storage_entanglement", "storage_readout", "entanglement_readout", "inter_entanglement")  # abstract events
NA_COMPARABLE_ZONE_CHANGES = ("storage_entanglement", "inter_entanglement")  # the ones the compiler has (no readout zone)
NA_ATOM_MOVES = "transports"  # compiled detail: atom pick-up, move, drop-off (any zone)

# SC raw result metrics (one row per swapped qubit, i.e. two per swap)
SC_SWAPS = "virtual_swap"  # abstract and compiled event
SC_COMPILED_SWAPS = "swaps"  # compiled detail, the same SWAPs as the compiled virtual_swap events


def visualize(platform: list[str], results_dir: Path, max_qubits: int | None = None) -> None:
    """Write the plots (and their CSV tables) of the given platforms; removes the previous plots and CSVs first."""
    if "TI" in platform:
        out_dir = results_dir / "ti" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
         
        for old_file in out_dir.glob("ti_*"):  # remove previous TI outputs (also of configs that no longer have results)
            old_file.unlink()
        runs = defaultdict(list)  # config -> the raw rows of each circuit
        for file in sorted((results_dir / "ti" / "raw").glob("*.csv")):
            with open(file, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and (max_qubits is None or int(rows[0]["num_qubits"]) <= max_qubits):
                runs[rows[0]["config"]].append(rows)
        for config, circuits in sorted(runs.items()):
            file_prefix = f"{out_dir}/ti_{re.sub(r'[^a-z0-9]+', '_', config.lower()).strip('_')}"
            visualize_ti_aggregated(config, circuits, file_prefix)
            visualize_ti_distribution(config, circuits, file_prefix)
    if "NA" in platform:
        out_dir = results_dir / "na" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        for old_file in out_dir.glob("na_*"):  # remove previous NA outputs (also of configs that no longer have results)
            old_file.unlink()
        runs = defaultdict(list)  # config -> the raw rows of each circuit
        for file in sorted((results_dir / "na" / "raw").glob("*.csv")):
            with open(file, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and (max_qubits is None or int(rows[0]["num_qubits"]) <= max_qubits):
                runs[rows[0]["config"]].append(rows)
        for config, circuits in sorted(runs.items()):
            file_prefix = f"{out_dir}/na_{re.sub(r'[^a-z0-9]+', '_', config.lower()).strip('_')}"
            visualize_na_aggregated(config, circuits, file_prefix)
            visualize_na_distribution(config, circuits, file_prefix)
    if "SC" in platform:
        out_dir = results_dir / "sc" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        for old_file in out_dir.glob("sc_*"):  # remove previous SC outputs (also of configs that no longer have results)
            old_file.unlink()
        runs = defaultdict(list)  # config -> the raw rows of each circuit
        for file in sorted((results_dir / "sc" / "raw").glob("*.csv")):
            with open(file, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and (max_qubits is None or int(rows[0]["num_qubits"]) <= max_qubits):
                runs[rows[0]["config"]].append(rows)
        for config, circuits in sorted(runs.items()):
            file_prefix = f"{out_dir}/sc_{re.sub(r'[^a-z0-9]+', '_', config.lower()).strip('_')}"
            visualize_sc_aggregated(config, circuits, file_prefix)
            visualize_sc_distribution(config, circuits, file_prefix)


def style(fig, ax, title: str, subtitle: str, x_label: str, y_label: str, notes: tuple[str, ...] = ()) -> None:
    """Common look of all plots: axes, title + subtitle, axis labels and the legend below the plot (`notes` become
    text-only legend entries)."""
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
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
              handlelength=2.4)
    fig.tight_layout()


def plot_event_amount(qubit_shares: list[tuple[float, float, int]], title: str, x_label: str, y_label: str,
                      above_note: str, file: str, zero_events: str | None = None) -> None:
    """Per qubit its share of the circuit's compiled events (x) vs of the abstract events (y), colored by the qubit
    number of its circuit. Shares are ~1/n, so both axes are logarithmic: the diagonal stays the exact match and a
    factor between the shares becomes a fixed distance from it. Zero shares get their own strip at the axis start; with
    `zero_events` (the name of the events, e.g. "swaps") the legend explains the zero strip and the share of each part."""
    shares = np.array([(m, a) for m, a, _ in qubit_shares]).reshape(-1, 2)
    qubits = np.array([n for _, _, n in qubit_shares])
    nonzero = shares[shares > 0]
    lo = nonzero.min() / 1.3 if nonzero.size else 0.01  # lower end of the log part
    has_zeros = bool((shares == 0).any())
    zero_at, end = lo / 2.2, 1.3  # zero strip, below a break at lo / 1.4
    start = lo / 3.2 if has_zeros else lo
    rng = np.random.default_rng(0)
    plotted = np.where(shares > 0, shares, zero_at * rng.uniform(0.85, 1.18, shares.shape))  # jitter the zero pile

    fig, ax = plt.subplots(figsize=(6.6, 6.9) if zero_events else (6.6, 6), facecolor=SURFACE)  # longer legend
    ax.set_xscale("log")
    ax.set_yscale("log")
    diagonal = np.array([zero_at if has_zeros else lo, end])  # through the zero strip's (0, 0) as well
    ax.fill_between(diagonal, diagonal / 1.5, diagonal * 1.5, color=REFERENCE, alpha=0.12, linewidth=0, zorder=1,
                    label="within 1.5× of its compiled share")
    ax.plot(diagonal, diagonal, color=REFERENCE, linewidth=3, zorder=2, label="exact match")
    for edge in (ax.axhline, ax.axvline) if has_zeros else ():  # break between the zero strip and the log axis
        edge(lo / 1.4, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1.5)

    order = np.argsort(-qubits)  # smaller circuits (larger shares, fewer points) drawn on top
    points = ax.scatter(plotted[order, 0], plotted[order, 1], s=26, c=qubits[order], cmap=QUBIT_CMAP,
                        vmin=QUBIT_MIN, vmax=QUBIT_MAX, alpha=0.8, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    colorbar = fig.colorbar(points, ax=ax, ticks=QUBIT_TICKS, fraction=0.05, pad=0.03)
    colorbar.set_label("qubits in circuit", color=TEXT_SECONDARY, fontsize=10)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)

    ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1) if t >= lo]
    for set_lim, set_ticks in ((ax.set_xlim, ax.set_xticks), (ax.set_ylim, ax.set_yticks)):
        set_lim(start, end)
        set_ticks([zero_at, *ticks] if has_zeros else ticks, [*(["0"] if has_zeros else []), *(f"{t:g}" for t in ticks)])
    ax.minorticks_off()

    subtitle = f"number of qubits = {len(shares)}"
    notes = [above_note, "one data point = one qubit, shares within its circuit (log scale)"]
    zero_header = f"Zero shares: the qubit has no {zero_events} of that kind in its circuit"
    if len(shares):
        m, a = shares[:, 0], shares[:, 1]
        subtitle += f"   > 1.5× its compiled share: {np.mean(a > 1.5 * m):.0%}, < 1/1.5×: {np.mean(1.5 * a < m):.0%}"
        if zero_events:  # the zero shares are explained in the legend instead
            notes += [zero_header,
                      f"bottom row (missed): compiled {zero_events}, but no abstract ones: {np.mean((m > 0) & (a == 0)):.0%}",
                      f"left column (invented): abstract {zero_events}, but no compiled ones: {np.mean((m == 0) & (a > 0)):.0%}",
                      f"corner (correct): neither, the qubit is not involved: {np.mean((m == 0) & (a == 0)):.0%}"]
        else:
            subtitle += f"   zero share: {np.mean((m == 0) | (a == 0)):.0%}"
    style(fig, ax, title, subtitle, x_label, y_label, notes=tuple(notes))
    for text in ax.get_legend().get_texts():
        if text.get_text() == zero_header:
            text.set_fontweight("bold")
            text.set_color(TEXT)
    fig.savefig(file, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


COUNT_TICKS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000)


def log_count_range(counts: list[int]) -> tuple[float, float, float | None]:
    """Range of a logarithmic count axis: (start of the log part, end, axis position of 0 or None if there are no zeros).
    As in plot_event_amount, zeros get their own strip at the axis start, below a break."""
    lo = min([c for c in counts if c > 0], default=1) / 1.3
    return lo, max([*counts, 1]) * 1.3, lo / 2.2 if 0 in counts else None


def set_log_count_axis(ax, axis: str, lo: float, end: float, zero_at: float | None) -> None:
    """Make the x or y axis a logarithmic count axis (see log_count_range), with a dashed break before the zero strip."""
    set_scale, set_lim, set_ticks, break_line = ((ax.set_xscale, ax.set_xlim, ax.set_xticks, ax.axvline) if axis == "x"
                                                 else (ax.set_yscale, ax.set_ylim, ax.set_yticks, ax.axhline))
    set_scale("log")
    set_lim(lo / 3.2 if zero_at else lo, end)
    ticks = [t for t in COUNT_TICKS if lo <= t <= end and (end / lo < 1000 or str(t)[0] == "1")]  # wide range: decades only
    set_ticks([zero_at, *ticks] if zero_at else ticks, [*(["0"] if zero_at else []), *(f"{t:g}" for t in ticks)])
    ax.minorticks_off()
    if zero_at:
        break_line(lo / 1.4, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1.5)


def on_log_count_axis(counts: list[int], zero_at: float | None) -> list[float]:
    """Axis positions of counts: zeros are drawn in the zero strip."""
    return [c if c > 0 else zero_at for c in counts]


def circuit_zorder(i: int) -> float:
    """Drawing order of the i-th circuit's dot in the aggregated plots: outline at +0, connecting line at +1e-5, fill at +2e-5."""
    return 4 + i * 3e-5


def connect_per_gate(fig, ax, xs: list[float], xs_per_gate: list[float], ys: list[float]) -> None:
    """Connect the two points of each circuit with a horizontal line: from the abstract model's dot (under its fill, over its
    outline) to the outer edge of the per-gate model's ring. Needs the final layout, as the ring radius is converted from points."""
    ring_radius = (math.sqrt(RING_SIZE) / 2 + RING_WIDTH / 2) * fig.dpi / 72  # in pixels (the axes may be logarithmic)
    for i, (x, x_per_gate, y) in enumerate(zip(xs, xs_per_gate, ys)):
        (px, py), (px_per_gate, _) = ax.transData.transform([(x, y), (x_per_gate, y)])
        if abs(px_per_gate - px) > ring_radius:
            x_line_end = ax.transData.inverted().transform((px_per_gate - math.copysign(ring_radius, px_per_gate - px), py))[0]
            ax.plot([x, x_line_end], [y, y], color=BASELINE, linewidth=0.8,
                    solid_capstyle="butt", zorder=circuit_zorder(i) + 1e-5)


EQUIVALENCE_FACTOR = 1.5  # a model's count is "close" to the compiled one if within this factor (band around y = x)


def lin_ccc(xs: list[float], ys: list[float]) -> float | None:
    """Lin's concordance correlation coefficient of the logged values (pairs with a 0 left out), or None if too few: 1 if all
    lie on y = x; unlike a correlation it also drops for an offset or a slope other than 1, so it measures equivalence.
    Logged, as the counts span orders of magnitude (otherwise the largest circuits dominate)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 3:
        return None
    log_x, log_y = np.log10(pairs).T
    covariance = np.mean((log_x - log_x.mean()) * (log_y - log_y.mean()))
    return 2 * covariance / (log_x.var() + log_y.var() + (log_x.mean() - log_y.mean()) ** 2)


def equivalence_scores(ax, xs: list[int], xs_per_gate: list[int], ys: list[int], sizes: list[int], x_start: float,
                       x_end: float) -> list[str]:
    """For a model count (x) of the same compiled quantity (y): draw the band within EQUIVALENCE_FACTOR around y = x and add
    the legend section equivalence (how close are the circuits to y = x?): median relative error (|x - y| / y, circuits
    with y = 0 left out), share within the band (an exact 0 counts as within), median ratio x / y (systematically too
    many or too few) and Lin's CCC for the counts and per anchor (circuit size removed, as both grow with it).
    Returns the section header."""
    band_xs = np.geomspace(x_start, x_end, 2)  # as long as the y = x line
    ax.fill_between(band_xs, band_xs / EQUIVALENCE_FACTOR, band_xs * EQUIVALENCE_FACTOR, color=REFERENCE, alpha=0.12,
                    linewidth=0, zorder=1, label=f"within {EQUIVALENCE_FACTOR:g}× of the compiled count")
    per_anchor = lambda counts: [n / max(size, 1) for n, size in zip(counts, sizes)]
    header = "Equivalence: how close are the circuits to y = x?"
    ax.plot([], [], linestyle="none", label=header)
    for name, model_xs, marker in (("abstract", xs, ABSTRACT_MARKER), ("per-gate", xs_per_gate, PER_GATE_MARKER)):
        errors = [abs(x - y) / y for x, y in zip(model_xs, ys) if y]
        within = np.mean([x == y or (x > 0 and y > 0 and max(x / y, y / x) <= EQUIVALENCE_FACTOR) for x, y in zip(model_xs, ys)])
        ratios = [x / y for x, y in zip(model_xs, ys) if x > 0 and y > 0]
        text = [f"median rel. error {np.median(errors):.0%}" if errors else "no compiled events",
                f"{within:.0%} within {EQUIVALENCE_FACTOR:g}×"]
        if ratios:
            text.append(f"median ratio to compiled ×{np.median(ratios):.2f}")
        ax.plot([], [], **marker, label=f"{name} model: {', '.join(text)}")
        ccc = [lin_ccc(model_xs, ys), lin_ccc(per_anchor(model_xs), per_anchor(ys))]
        zeros = sum(x == 0 or y == 0 for x, y in zip(model_xs, ys))
        ax.plot([], [], linestyle="none", label=f"    Lin's CCC (log): raw counts {'–' if ccc[0] is None else f'{ccc[0]:.2f}'},"
                f"\n        per anchor (bias due to circuit size removed): {'–' if ccc[1] is None else f'{ccc[1]:.2f}'}" + (f", {zeros} with a 0 left out" if zeros else ""))
    return [header]


ABSTRACT_MARKER = dict(marker="o", markersize=7, linestyle="none", color=SERIES)  # legend handles of the two models
PER_GATE_MARKER = dict(marker="o", markersize=6, linestyle="none", markerfacecolor="none", markeredgecolor=PER_GATE,
                       markeredgewidth=1.4)
BAND_SHARE = 0.8  # share of the circuits inside the band around the power-law fit
BOOTSTRAP_SAMPLES = 500  # resamples of the circuits for the confidence interval of Kendall's τ


def correlation(xs: list[float], ys: list[float], rng: np.random.Generator) -> str:
    """Legend text of the rank correlation (does the order of the circuits agree?): Kendall τ-b with a bootstrap 95% CI
    (share of circuit pairs in the same order = (1 + τ) / 2) and Spearman ρ."""
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return "no variation"
    xs, ys = np.asarray(xs), np.asarray(ys)
    tau = kendalltau(xs, ys).statistic
    resamples = (rng.integers(0, len(xs), len(xs)) for _ in range(BOOTSTRAP_SAMPLES))
    low, high = np.nanpercentile([kendalltau(xs[i], ys[i]).statistic for i in resamples], [2.5, 97.5])
    return f"Kendall τ = {tau:.2f}, 95% CI [{low:.2f}, {high:.2f}], Spearman ρ = {spearmanr(xs, ys).statistic:.2f}"


def power_law_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float, float, float, int] | None:
    """Robust (Theil-Sen) fit of log y = log c + b · log x, i.e. y = c · x^b, over the circuits with both values > 0:
    (b, 95% CI low and high of b, c, factor k around the fit with BAND_SHARE of these circuits within ×k, circuits left out
    for a zero), or None if there are too few. b = 1 means proportional, y = c · x."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2:
        return None
    log_x, log_y = np.log10(pairs).T
    b, log_c, b_low, b_high = theilslopes(log_y, log_x)
    k = 10 ** np.quantile(np.abs(log_y - (log_c + b * log_x)), BAND_SHARE)
    return b, b_low, b_high, 10 ** log_c, k, len(xs) - len(pairs)


def proportionality(fit: tuple | None) -> str:
    """Legend text of a power_law_fit: the exponent b with its CI and the band factor."""
    if fit is None:
        return "too few circuits"
    b, b_low, b_high, _, k, zeros = fit
    return (f"b = {b:.2f}, 95% CI [{b_low:.2f}, {b_high:.2f}], {BAND_SHARE:.0%} within ×{k:.2g}"
            + (f", {zeros} with a 0 left out" if zeros else ""))


def proxy_fit_and_scores(ax, xs: list[int], xs_per_gate: list[int], ys: list[int], sizes: list[int], x_lo: float,
                         x_end: float, y_lo: float) -> list[str]:
    """For a proxy (x) of a different compiled quantity (y): draw the power-law fit of the abstract model with its band and a
    proportional reference, and add the legend sections correlation (does the order of the circuits agree?) and
    proportionality (does y grow by a fixed factor with x?), both for the counts and per anchor (circuit size removed, as
    both grow with it). Returns the section headers."""
    fit = power_law_fit(xs, ys)
    if fit:
        b, _, _, c, k, _ = fit
        fit_xs = np.geomspace(x_lo, x_end, 200)  # only in the log part of the axes
        fit_ys = c * fit_xs ** b
        ax.fill_between(fit_xs, np.maximum(fit_ys / k, y_lo), np.maximum(fit_ys * k, y_lo), color=FIT, alpha=0.1, linewidth=0,
                        zorder=1, label=f"{BAND_SHARE:.0%} of the circuits (within ×{k:.2g} of the fit)")
        ax.plot(fit_xs, np.where(fit_ys >= y_lo, fit_ys, np.nan), color=FIT, alpha=0.6, linewidth=2, zorder=2,
                label=f"power-law fit y = {c:.2g} · x^{b:.2f} of the abstract model (robust, Theil-Sen)")
        ratio = np.median([y / x for x, y in zip(xs, ys) if x > 0 and y > 0])
        ax.plot(fit_xs, np.where(ratio * fit_xs >= y_lo, ratio * fit_xs, np.nan), color=REFERENCE, linewidth=3,
                zorder=2, label=f"proportional (b = 1) with the median ratio y / x = {ratio:.2g}")

    rng = np.random.default_rng(0)
    per_anchor = lambda counts: [n / max(size, 1) for n, size in zip(counts, sizes)]
    models = (("abstract", xs, ABSTRACT_MARKER), ("per-gate", xs_per_gate, PER_GATE_MARKER))
    headers = ["Correlation: is the order of the circuits the same?",
               "Proportionality: log-log fit y = c · x^b (b = 1: proportional)"]
    ax.plot([], [], linestyle="none", label=headers[0])
    for name, model_xs, marker in models:
        ax.plot([], [], **marker, label=f"{name} model, raw counts: {correlation(model_xs, ys, rng)}")
        ax.plot([], [], linestyle="none", label=f"    per anchor (bias due to circuit size removed):\n        {correlation(per_anchor(model_xs), per_anchor(ys), rng)}")
    ax.plot([], [], linestyle="none", label=headers[1])
    for name, model_xs, marker in models:
        ax.plot([], [], **marker, label=f"{name} model, raw counts: {proportionality(power_law_fit(model_xs, ys))}")
        ax.plot([], [], linestyle="none",
                label=f"    per anchor (bias due to circuit size removed):\n        {proportionality(power_law_fit(per_anchor(model_xs), per_anchor(ys)))}")
    return headers


def visualize_ti_aggregated(config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    """Scatter aggregated abstract vs compiled metrics.

    Each figure compares the aggregated numbers of a certain hardware events.
    Thus 4 figures per hardware configuration, answering 2 questions:

    A) estimation of interaction zone changes
    - number of abstract VS concrete interaction zone entries during the whole circuit
    - number of abstract VS concrete interaction zone evictions during the whole circuit
    - number of abstract VS concrete interaction zone changes (entries + evictions) during the whole circuit

    (for A, the optimal outcome is for all data points to lie on the angle bisector)

    B) correspondence of interaction zone changes and physical movement operations (shuttles + physical swaps)
    - number of abstract interaction zone changes during the whole circuit VS number of concrete movement (shuttling + physical swap) operations

    (for B, the optimal outcome is for all data points to lie on an arbitrary increasing linear function)
    """
    # 1) one row per circuit with the summed counts
    scores = []
    for rows in circuits:
        counts = Counter()  # (source, metric) -> summed count
        wall_clock_s = {}  # source -> run time in seconds
        for row in rows:
            if row["count"]:
                counts[(row["source"], row["metric"])] += int(row["count"])
            elif row["metric"] == "wall_clock_s":
                wall_clock_s[row["source"]] = float(row["value"])
        scores.append({
            "circuit": rows[0]["circuit"],
            "num_qubits": int(rows[0]["num_qubits"]),
            "anchors": sum(int(row["value"]) for row in rows if row["metric"] == "anchors"),  # circuit size
            "abstract_iz_entries": counts[("abstract", "interaction_zone_entry")],
            "abstract_iz_evictions": counts[("abstract", "interaction_zone_eviction")],
            "abstract_iz_changes": sum(counts[("abstract", metric)] for metric in TI_IZ_CHANGES),
            "compiled_iz_entries": counts[("compiled", "interaction_zone_entry")],
            "compiled_iz_evictions": counts[("compiled", "interaction_zone_eviction")],
            "compiled_iz_changes": sum(counts[("compiled", metric)] for metric in TI_IZ_CHANGES),
            "compiled_ion_moves": sum(counts[("compiled", metric)] for metric in TI_ION_MOVES),
            "per_gate_iz_entries": counts[("per_gate", "interaction_zone_entry")],
            "per_gate_iz_evictions": counts[("per_gate", "interaction_zone_eviction")],
            "per_gate_iz_changes": sum(counts[("per_gate", metric)] for metric in TI_IZ_CHANGES),
            "abstract_wall_clock_s": wall_clock_s["abstract"],
            "compiled_wall_clock_s": wall_clock_s["compiled"],
        })
    scores.sort(key=lambda c: c["num_qubits"])  # larger circuits drawn on top
    qubits = [c["num_qubits"] for c in scores]
    mean_speedup = np.mean([c["compiled_wall_clock_s"] / c["abstract_wall_clock_s"] for c in scores])  # per-circuit quotient, then mean

    # 2) the plots: (file name, title, x column, per-gate x column, y column, x label, y label, same quantity on both axes)
    plots = [
        ("iz_entries", "IZ entries", "abstract_iz_entries", "per_gate_iz_entries", "compiled_iz_entries",
         "abstract IZ entries", "compiled IZ entries", True),
        ("iz_evictions", "IZ evictions", "abstract_iz_evictions", "per_gate_iz_evictions", "compiled_iz_evictions",
         "abstract IZ evictions", "compiled IZ evictions", True),
        ("iz_changes", "IZ changes (entries + evictions)", "abstract_iz_changes", "per_gate_iz_changes", "compiled_iz_changes",
         "abstract IZ changes", "compiled IZ changes", True),
        ("iz_changes_vs_ion_moves", "IZ changes vs ion moves", "abstract_iz_changes", "per_gate_iz_changes", "compiled_ion_moves",
         "abstract IZ changes", "compiled shuttles + swaps", False),
    ]
    for name, title, x_key, per_gate_key, y_key, x_label, y_label, same_quantity in plots:
        xs = [c[x_key] for c in scores]
        xs_per_gate = [c[per_gate_key] for c in scores]
        ys = [c[y_key] for c in scores]
        fig, ax = plt.subplots(figsize=(7, 7.6) if same_quantity else (7.6, 9), facecolor=SURFACE)  # proxy: longer, wider legend

        # logarithmic axes (counts over orders of magnitude, zeros in their own strip), and the y = x line if both axes show
        # the same quantity; the dots are drawn at their axis positions (px, py), the fits and scores use the counts
        x_lo, x_end, x_zero = log_count_range([*xs, *xs_per_gate, *(ys if same_quantity else [])])
        y_lo, y_end, y_zero = (x_lo, x_end, x_zero) if same_quantity else log_count_range(ys)
        set_log_count_axis(ax, "x", x_lo, x_end, x_zero)
        set_log_count_axis(ax, "y", y_lo, y_end, y_zero)
        px, px_per_gate, py = on_log_count_axis(xs, x_zero), on_log_count_axis(xs_per_gate, x_zero), on_log_count_axis(ys, y_zero)
        if same_quantity:
            diagonal = [x_zero or x_lo, x_end]  # through the zero strip's (0, 0) as well
            ax.plot(diagonal, diagonal, color=REFERENCE, linewidth=3, zorder=2, label="exact match")

        subtitle = f"number of circuits = {len(xs)}   avg. speedup of abstract execution vs compilation: {mean_speedup:.3g}×"

        # the circuits: per-gate model hollow, abstract model colored by qubit number; one circuit after the other its white outline,
        # then the line connecting its two points (see connect_per_gate), then its fill, so the line reaches the dot over the outline
        ax.scatter(px_per_gate, py, s=RING_SIZE, facecolors="none", edgecolors=PER_GATE, linewidths=RING_WIDTH, zorder=3)
        points = ScalarMappable(Normalize(QUBIT_MIN, QUBIT_MAX), QUBIT_CMAP)
        for i, (x, y, qubit) in enumerate(zip(px, py, qubits)):
            ax.scatter([x], [y], s=56, color=SURFACE, edgecolors=SURFACE, linewidths=1.6, zorder=circuit_zorder(i))
            ax.scatter([x], [y], s=56, color=points.to_rgba(qubit), linewidths=0, zorder=circuit_zorder(i) + 2e-5)
        headers = []  # bold legend section headers
        if same_quantity:
            headers = equivalence_scores(ax, xs, xs_per_gate, ys, [c["anchors"] for c in scores], diagonal[0], x_end)
        else:  # a proxy: correlation and proportionality instead of equivalence
            headers = proxy_fit_and_scores(ax, xs, xs_per_gate, ys, [c["anchors"] for c in scores], x_lo, x_end, y_lo)
        colorbar = fig.colorbar(points, ax=ax, ticks=QUBIT_TICKS, fraction=0.05, pad=0.03)
        colorbar.set_label("qubits in circuit", color=TEXT_SECONDARY, fontsize=10)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)

        style(fig, ax, f"{config}: {title}", subtitle, x_label, y_label)
        for text in ax.get_legend().get_texts():
            if text.get_text() in headers:
                text.set_fontweight("bold")
                text.set_color(TEXT)
        connect_per_gate(fig, ax, px, px_per_gate, py)  # after the layout is fixed
        fig.savefig(f"{file_prefix}__{name}.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)

    # 3) table of the plotted numbers
    with open(f"{file_prefix}__aggregated.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(sorted(scores, key=lambda c: c["circuit"]))


def f1(predicted: set, actual: set, tolerance: int) -> float:
    """F1 of slot sets; a slot matches if the other set has a slot of the same qubit within ±tolerance anchors."""
    if not predicted or not actual:
        return float("nan")
    def hits(slots, others):
        return sum(any((q, a + d) in others for d in range(-tolerance, tolerance + 1)) for q, a in slots)
    precision, recall = hits(predicted, actual) / len(predicted), hits(actual, predicted) / len(actual)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def align(abstract: list[int], compiled: list[int], gap: float = 2.0) -> tuple[list[int], int, int]:
    """Order-preserving optimal pairing of two anchor sequences: a pair costs |shift|, an unpaired event `gap` (so
    paired events are at most 2 * gap anchors apart). Returns the shifts (abstract - compiled) and unpaired counts."""
    n, m = len(abstract), len(compiled)
    cost = np.zeros((n + 1, m + 1))
    cost[:, 0], cost[0, :] = np.arange(n + 1) * gap, np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + abs(abstract[i - 1] - compiled[j - 1]),
                             cost[i - 1, j] + gap, cost[i, j - 1] + gap)
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


def visualize_ti_distribution(config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    """Location-based metrics, i.e. where in the circuit hardware events happen.

    Four figures per hardware configuration:

    A) estimation of interaction zone changes
    - location agreement:  F1 score between the abstract model vs the compilation having an IZ change (entry or eviction)
                           in a given slot, exactly and with a tolerance of a certain number of anchors of the same qubit
                           (the optimal outcome is for all data points to lie on 1)
    - timing shift: per qubit and kind (entry / eviction), the abstract and compiled IZ changes are paired, each paired
                    event gives [shift = abstract anchor - compiled anchor] (< 0: event too early, > 0: event too late);
                    unpaired events are "missing" (too few abstract events) or "extra" (too many abstract events)
                    (the optimal outcome is for all data be concentrated on 0)

    B) correspondence of interaction zone changes and physical movement operations (shuttles + physical swaps)
    - event timing: per qubit, [shift = average anchor of its abstract IZ changes - average anchor of its compiled ion moves]
                    (shuttles + swaps), i.e. abstract events inserted earlier or later than concrete events
                    (the optimal outcome is for all data points to lie on 0)
    - event amount: per qubit, its share (number of events of qubit / number of events in general) of its
                    abstract IZ changes vs its moves over the whole circuit
                    (the optimal outcome is for all data points to lie on the angle bisector)

    """
    # 1) one score row per circuit, plus the per-event / per-qubit values pooled over all circuits
    scores = []
    shifts, missing, extra = [], 0, 0  # paired IZ change shifts, unpaired IZ changes
    qubit_shifts_per_circuit = []  # per circuit: the event timing shift of each qubit
    qubit_shares = []  # per qubit: (share of the circuit's ion moves, share of the circuit's abstract IZ changes)
    nan = float("nan")
    for rows in circuits:
        num_qubits = int(rows[0]["num_qubits"])

        # per slot: abstract / compiled IZ changes and compiled ion moves; per qubit: number of anchors
        iz_changes: dict[str, Counter[tuple[int, int]]] = {"abstract": Counter(), "compiled": Counter()}
        events: defaultdict[tuple[str, str, int], list[int]] = defaultdict(list)  # (source, kind, qubit) -> anchors
        moves: Counter[tuple[int, int]] = Counter()
        anchors = [0] * num_qubits
        for row in rows:
            if row["source"] == "circuit" and row["metric"] == "anchors":
                anchors[int(row["qubit"])] = int(row["value"])
            elif row["count"] and row["source"] != "per_gate":  # the per-gate model is only compared in the aggregated plots
                slot = (int(row["qubit"]), int(row["anchor"]))
                if row["metric"] in TI_IZ_CHANGES:
                    iz_changes[row["source"]][slot] += int(row["count"])
                    events[(row["source"], row["metric"], slot[0])].append(slot[1])
                elif row["source"] == "compiled" and row["metric"] in TI_ION_MOVES:
                    moves[slot] += int(row["count"])
        if not any(anchors):
            print(f"Skipped {config} / {rows[0]['circuit']}: no anchor rows in the raw file (re-run the runner)")
            continue

        # timing shift: pair each qubit's abstract and compiled IZ changes of the same kind
        circuit_shifts, circuit_missing, circuit_extra = [], 0, 0
        for kind in TI_IZ_CHANGES:
            for qubit in range(num_qubits):
                paired, unpaired_abstract, unpaired_compiled = align(sorted(events[("abstract", kind, qubit)]),
                                                                     sorted(events[("compiled", kind, qubit)]))
                circuit_shifts += paired
                circuit_extra += unpaired_abstract
                circuit_missing += unpaired_compiled
        shifts += circuit_shifts
        missing += circuit_missing
        extra += circuit_extra

        # event timing and amount per qubit (slots before its last anchor: (qubit, k) for k < its number of anchors)
        abstract_per_qubit = [[iz_changes["abstract"][(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
        moves_per_qubit = [[moves[(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
        qubit_shifts = [float(np.average(range(len(a)), weights=a) - np.average(range(len(m)), weights=m))
                        for a, m in zip(abstract_per_qubit, moves_per_qubit) if sum(a) and sum(m)]
        if qubit_shifts:
            qubit_shifts_per_circuit.append(qubit_shifts)
        abstract_total, moves_total = sum(map(sum, abstract_per_qubit)), sum(map(sum, moves_per_qubit))
        if abstract_total and moves_total:
            qubit_shares += [(sum(m) / moves_total, sum(a) / abstract_total, num_qubits)
                             for a, m in zip(abstract_per_qubit, moves_per_qubit) if len(a)]

        abstract_slots, compiled_slots = set(iz_changes["abstract"]), set(iz_changes["compiled"])
        scores.append({
            "circuit": rows[0]["circuit"],
            "num_qubits": num_qubits,
            **{f"f1_{'exact' if t == 0 else f'pm{t}'}": f1(abstract_slots, compiled_slots, t) for t in F1_TOLERANCES},
            "paired_iz_changes": len(circuit_shifts),
            "share_on_time": circuit_shifts.count(0) / len(circuit_shifts) if circuit_shifts else nan,
            "mean_shift": float(np.mean(circuit_shifts)) if circuit_shifts else nan,
            "missing_iz_changes": circuit_missing,
            "extra_iz_changes": circuit_extra,
            "qubits_early": float(np.mean(np.array(qubit_shifts) < -ON_TIME)) if qubit_shifts else nan,
            "qubits_on_time": float(np.mean(np.abs(qubit_shifts) <= ON_TIME)) if qubit_shifts else nan,
            "qubits_late": float(np.mean(np.array(qubit_shifts) > ON_TIME)) if qubit_shifts else nan,
            "event_shift_median": float(np.median(qubit_shifts)) if qubit_shifts else nan,
            "noise_shift_q25": float(np.percentile(qubit_shifts, 25)) if qubit_shifts else nan,
            "noise_shift_q75": float(np.percentile(qubit_shifts, 75)) if qubit_shifts else nan,
        })
    if not scores:
        return
    scores.sort(key=lambda c: (c["num_qubits"], c["circuit"]))

    # A) location agreement: F1 over the allowed anchor offset, one thin line per circuit, median bold (with its
    # values) and the middle 50% of circuits as a band
    keys = [key for key in scores[0] if key.startswith("f1_")]
    scored = [c for c in scores if not np.isnan(c["f1_exact"])]
    f1s = np.array([[c[k] for k in keys] for c in scored]).reshape(-1, len(keys))
    fig, ax = plt.subplots(figsize=(6, 7), facecolor=SURFACE)  # taller: room for the explanatory legend
    ax.axhline(1, color=REFERENCE, linewidth=4, zorder=1.6, label="abstract and concrete IZ changes in the same slots")  # above the grid
    for row in f1s:
        ax.plot(F1_TOLERANCES, row, color=SERIES, alpha=0.18, linewidth=1, zorder=2)
    ax.plot([], [], color=SERIES, alpha=0.4, linewidth=1, label="one circuit")
    if len(f1s):
        ax.fill_between(F1_TOLERANCES, np.percentile(f1s, 25, axis=0), np.percentile(f1s, 75, axis=0), color=SERIES,
                        alpha=0.15, linewidth=0, zorder=2, label="middle 50% of circuits")
        medians = np.median(f1s, axis=0)
        ax.plot(F1_TOLERANCES, medians, color=SERIES, linewidth=2.5, marker="o", zorder=3, label="median circuit")
        for t, m in zip(F1_TOLERANCES, medians):
            ax.annotate(f"{m:.2f}", (t, m), textcoords="offset points", xytext=(0, -16), ha="center",
                        color=TEXT_SECONDARY, fontsize=9)
    ax.set_xticks(F1_TOLERANCES, [f"±{t}" if t else "exact" for t in F1_TOLERANCES])
    ax.set_ylim(0, 1.04)
    style(fig, ax, f"{config}: IZ change location agreement", f"number of circuits = {len(scored)}",
          "allowed anchor offset", "F1 score of IZ change slots",
          notes=("slot = one qubit's interval between two consecutive anchors",
                 "A / C = slots with ≥ 1 abstract / compiled IZ change",
                 "precision (how many of the abstract events are real) ",
                 "     = share of A with a C slot within ±offset",
                 "recall (how many real places the model finds) ",
                 "     = share of C with an A slot within ±offset",
                 "F1 = 2 · precision · recall / (precision + recall)"))
    fig.savefig(f"{file_prefix}__location_agreement.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

    # A) timing shift: histogram of the shifts of all paired IZ changes, plus the unpaired ones
    by_shift = Counter(shifts)
    span = max([1, *(abs(s) for s in shifts)])
    values = list(range(-span, span + 1))
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=SURFACE)
    ax.bar(values, [by_shift[v] for v in values], width=0.8, color=SERIES, zorder=3, label="paired IZ changes")
    ax.bar([-span - 2, span + 2], [missing, extra], width=0.8, color=MUTED, zorder=3,
           label=f"unpaired IZ changes: {missing} missing, {extra} extra")
    ax.axvline(0, color=REFERENCE, linewidth=3, zorder=4)
    ax.set_xticks([-span - 2, *values, span + 2], ["missing", *(f"{v:+d}" if v else "0" for v in values), "extra"])
    subtitle = "no paired IZ changes"
    if shifts:
        subtitle = (f"{len(shifts)} paired: {shifts.count(0) / len(shifts):.0%} on time,"
                    f" {sum(s < 0 for s in shifts) / len(shifts):.0%} early, {sum(s > 0 for s in shifts) / len(shifts):.0%} late,"
                    f" mean shift {np.mean(shifts):+.2f} anchors")
    style(fig, ax, f"{config}: IZ change timing (abstract vs compiled)", subtitle,
          "timing shift in anchors", "IZ changes (all circuits)",
          notes=("shift = abstract - compiled anchor of a paired IZ change",
                 "< 0: events too early, > 0: too late",
                 "missing: compiled IZ change without abstract one (too few events)",
                 "extra: abstract IZ change without compiled one (too many events)"))
    fig.savefig(f"{file_prefix}__timing_shift.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

    # B) event timing: per qubit, where the weight of its abstract IZ changes lies relative to the weight center of its
    # real (compiled) ion moves; per shift bin the share of a circuit's qubits (dots) and of the average circuit (bar)
    edges = [-np.inf, -2, -1, -0.5, -ON_TIME, ON_TIME, 0.5, 1, 2, np.inf]
    bin_labels = ["< -2", "-2…-1", "-1…-½", "−½…−¼", "±¼", "¼…½", "½…1", "1…2", "> 2"]
    center = bin_labels.index("±¼")
    per_circuit = [np.histogram(s, edges)[0] / len(s) for s in qubit_shifts_per_circuit]
    positions = np.arange(len(bin_labels))
    jitter = np.random.default_rng(0)  # fixed, so the plot is the same on every run
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    ax.axvline(center, color=REFERENCE, linewidth=4, zorder=1.6,  # above the grid
               label="weight center of the qubit's real ion moves")
    if per_circuit:
        ax.bar(positions, np.mean(per_circuit, axis=0), width=0.7, color=SERIES, alpha=0.25, zorder=2,
               label="average circuit")
        for shares in per_circuit:
            drawn = shares > 0  # circuits without qubits in a bin are left out, so the tails stay readable
            ax.scatter(positions[drawn] + jitter.uniform(-0.22, 0.22, drawn.sum()), shares[drawn], s=18,
                       color=SERIES, alpha=0.8, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.scatter([], [], s=18, color=SERIES, label="circuit")
    ax.set_xticks(positions, bin_labels)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, 1.04)
    scored = [c for c in scores if not np.isnan(c["qubits_on_time"])]
    style(fig, ax, f"{config}: event timing, abstract vs compiled", f"number of circuits = {len(scored)}",
          "anchor distance between the abstract event weight compared to the concrete event weight",
          "share of the circuit's qubits",
          notes=("per-qubit comparison of weight center (avg. anchor) of its compiled ion moves",
                 "and the weight center of its abstract IZ changes",
                 "< 0: abstract events too early on a qubit, > 0: too late"))
    fig.savefig(f"{file_prefix}__event_timing.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

    # B) event amount: per qubit its share of the circuit's abstract IZ changes vs of its ion moves
    plot_event_amount(qubit_shares, f"{config}: event amount vs ion moves",
                      "qubit's share of the circuit's compiled ion moves", "qubit's share of the abstract IZ changes",
                      "above: more abstract events than its movement share, below: less", f"{file_prefix}__event_amount.png")

    # table of the per-circuit scores (empty = undefined, e.g. no IZ change slot in the circuit)
    with open(f"{file_prefix}__distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        for c in sorted(scores, key=lambda c: c["circuit"]):
            writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()})


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def visualize_na_aggregated(config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    """Scatter aggregated abstract vs compiled metrics.

    Each figure compares the aggregated numbers of a certain hardware events.
    Thus 4 figures per hardware configuration, answering 2 questions:

    A) estimation of zone changes (only the kinds the compiler has: MQT has no readout zone)
    - number of abstract VS concrete storage <-> entanglement zone changes during the whole circuit
    - number of abstract VS concrete entanglement site changes (between CZ sites) during the whole circuit
    - number of abstract VS concrete zone changes (both of the above) during the whole circuit

    (for A, the optimal outcome is for all data points to lie on the angle bisector)

    B) correspondence of zone changes and physical movement operations (atom transports: pick-up, move, drop-off)
    - number of abstract zone changes (all kinds, incl. readout) during the whole circuit VS number of concrete atom transports

    (for B, the optimal outcome is for all data points to lie on an arbitrary increasing linear function)
    """
    # 1) one row per circuit with the summed counts
    scores = []
    for rows in circuits:
        counts = Counter()  # (source, metric) -> summed count
        wall_clock_s = {}  # source -> run time in seconds
        for row in rows:
            if row["count"]:
                counts[(row["source"], row["metric"])] += float(row["count"])  # move_distance is not an integer
            elif row["metric"] == "wall_clock_s":
                wall_clock_s[row["source"]] = float(row["value"])
        scores.append({
            "circuit": rows[0]["circuit"],
            "num_qubits": int(rows[0]["num_qubits"]),
            "anchors": sum(int(row["value"]) for row in rows if row["metric"] == "anchors"),  # circuit size
            "abstract_storage_entanglement": int(counts[("abstract", "storage_entanglement")]),
            "abstract_inter_entanglement": int(counts[("abstract", "inter_entanglement")]),
            "abstract_zone_changes": int(sum(counts[("abstract", metric)] for metric in NA_COMPARABLE_ZONE_CHANGES)),
            "abstract_all_zone_changes": int(sum(counts[("abstract", metric)] for metric in NA_ZONE_CHANGES)),
            "compiled_storage_entanglement": int(counts[("compiled", "storage_entanglement")]),
            "compiled_inter_entanglement": int(counts[("compiled", "inter_entanglement")]),
            "compiled_zone_changes": int(sum(counts[("compiled", metric)] for metric in NA_COMPARABLE_ZONE_CHANGES)),
            "compiled_atom_moves": int(counts[("compiled", NA_ATOM_MOVES)]),
            "per_gate_storage_entanglement": int(counts[("per_gate", "storage_entanglement")]),
            "per_gate_inter_entanglement": int(counts[("per_gate", "inter_entanglement")]),
            "per_gate_zone_changes": int(sum(counts[("per_gate", metric)] for metric in NA_COMPARABLE_ZONE_CHANGES)),
            "per_gate_all_zone_changes": int(sum(counts[("per_gate", metric)] for metric in NA_ZONE_CHANGES)),
            "abstract_wall_clock_s": wall_clock_s["abstract"],
            "compiled_wall_clock_s": wall_clock_s["compiled"],
        })
    scores.sort(key=lambda c: c["num_qubits"])  # larger circuits drawn on top
    qubits = [c["num_qubits"] for c in scores]
    mean_speedup = np.mean([c["compiled_wall_clock_s"] / c["abstract_wall_clock_s"] for c in scores])  # per-circuit quotient, then mean

    # 2) the plots: (file name, title, x column, per-gate x column, y column, x label, y label, same quantity on both axes)
    plots = [
        ("storage_entanglement", "storage ↔ entanglement zone changes", "abstract_storage_entanglement", "per_gate_storage_entanglement",
         "compiled_storage_entanglement", "abstract storage ↔ EZ changes", "compiled storage ↔ EZ changes", True),
        ("inter_entanglement", "entanglement site changes", "abstract_inter_entanglement", "per_gate_inter_entanglement", "compiled_inter_entanglement",
         "abstract EZ site changes", "compiled EZ site changes", True),
        ("zone_changes", "zone changes (storage ↔ EZ + EZ site)", "abstract_zone_changes", "per_gate_zone_changes", "compiled_zone_changes",
         "abstract zone changes", "compiled zone changes", True),
        ("zone_changes_vs_atom_moves", "zone changes vs atom moves", "abstract_all_zone_changes", "per_gate_all_zone_changes", "compiled_atom_moves",
         "abstract zone changes (incl. readout)", "compiled atom transports", False),
    ]
    for name, title, x_key, per_gate_key, y_key, x_label, y_label, same_quantity in plots:
        xs = [c[x_key] for c in scores]
        xs_per_gate = [c[per_gate_key] for c in scores]
        ys = [c[y_key] for c in scores]
        fig, ax = plt.subplots(figsize=(7, 7.6) if same_quantity else (7.6, 9), facecolor=SURFACE)  # proxy: longer, wider legend

        # logarithmic axes (counts over orders of magnitude, zeros in their own strip), and the y = x line if both axes show
        # the same quantity; the dots are drawn at their axis positions (px, py), the fits and scores use the counts
        x_lo, x_end, x_zero = log_count_range([*xs, *xs_per_gate, *(ys if same_quantity else [])])
        y_lo, y_end, y_zero = (x_lo, x_end, x_zero) if same_quantity else log_count_range(ys)
        set_log_count_axis(ax, "x", x_lo, x_end, x_zero)
        set_log_count_axis(ax, "y", y_lo, y_end, y_zero)
        px, px_per_gate, py = on_log_count_axis(xs, x_zero), on_log_count_axis(xs_per_gate, x_zero), on_log_count_axis(ys, y_zero)
        if same_quantity:
            diagonal = [x_zero or x_lo, x_end]  # through the zero strip's (0, 0) as well
            ax.plot(diagonal, diagonal, color=REFERENCE, linewidth=3, zorder=2, label="exact match")

        subtitle = f"number of circuits = {len(xs)}   avg. speedup of abstract execution vs compilation: {mean_speedup:.3g}×"

        # the circuits: per-gate model hollow, abstract model colored by qubit number; one circuit after the other its white outline,
        # then the line connecting its two points (see connect_per_gate), then its fill, so the line reaches the dot over the outline
        ax.scatter(px_per_gate, py, s=RING_SIZE, facecolors="none", edgecolors=PER_GATE, linewidths=RING_WIDTH, zorder=3)
        points = ScalarMappable(Normalize(QUBIT_MIN, QUBIT_MAX), QUBIT_CMAP)
        for i, (x, y, qubit) in enumerate(zip(px, py, qubits)):
            ax.scatter([x], [y], s=56, color=SURFACE, edgecolors=SURFACE, linewidths=1.6, zorder=circuit_zorder(i))
            ax.scatter([x], [y], s=56, color=points.to_rgba(qubit), linewidths=0, zorder=circuit_zorder(i) + 2e-5)
        headers = []  # bold legend section headers
        if same_quantity:
            headers = equivalence_scores(ax, xs, xs_per_gate, ys, [c["anchors"] for c in scores], diagonal[0], x_end)
        else:  # a proxy: correlation and proportionality instead of equivalence
            headers = proxy_fit_and_scores(ax, xs, xs_per_gate, ys, [c["anchors"] for c in scores], x_lo, x_end, y_lo)
        colorbar = fig.colorbar(points, ax=ax, ticks=QUBIT_TICKS, fraction=0.05, pad=0.03)
        colorbar.set_label("qubits in circuit", color=TEXT_SECONDARY, fontsize=10)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)

        style(fig, ax, f"{config}: {title}", subtitle, x_label, y_label)
        for text in ax.get_legend().get_texts():
            if text.get_text() in headers:
                text.set_fontweight("bold")
                text.set_color(TEXT)
        connect_per_gate(fig, ax, px, px_per_gate, py)  # after the layout is fixed
        fig.savefig(f"{file_prefix}__{name}.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)

    # 3) table of the plotted numbers
    with open(f"{file_prefix}__aggregated.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(sorted(scores, key=lambda c: c["circuit"]))


def visualize_na_distribution(config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    """Location-based metrics, i.e. where in the circuit hardware events happen.

    Four figures per hardware configuration:

    A) estimation of zone changes (only the kinds the compiler has: storage <-> entanglement zone, between CZ sites)
    - location agreement:  F1 score between the abstract model vs the compilation having a zone change in a given slot,
                           exactly and with a tolerance of a certain number of anchors of the same qubit
                           (the optimal outcome is for all data points to lie on 1)
    - timing shift: per qubit and kind, the abstract and compiled zone changes are paired, each paired event gives
                    [shift = abstract anchor - compiled anchor] (< 0: event too early, > 0: event too late);
                    unpaired events are "missing" (too few abstract events) or "extra" (too many abstract events)
                    (the optimal outcome is for all data be concentrated on 0)

    B) correspondence of zone changes (all kinds, incl. readout) and physical movement operations (atom transports)
    - event timing: per qubit, [shift = average anchor of its abstract zone changes - average anchor of its compiled
                    atom transports], i.e. abstract events inserted earlier or later than concrete events
                    (the optimal outcome is for all data points to lie on 0)
    - event amount: per qubit, its share (number of events of qubit / number of events in general) of its
                    abstract zone changes vs its atom transports over the whole circuit
                    (the optimal outcome is for all data points to lie on the angle bisector)

    """
    # 1) one score row per circuit, plus the per-event / per-qubit values pooled over all circuits
    scores = []
    shifts, missing, extra = [], 0, 0  # paired zone change shifts, unpaired zone changes
    qubit_shifts_per_circuit = []  # per circuit: the event timing shift of each qubit
    qubit_shares = []  # per qubit: (share of the circuit's atom transports, share of the circuit's abstract zone changes)
    nan = float("nan")
    for rows in circuits:
        num_qubits = int(rows[0]["num_qubits"])

        # per slot: abstract / compiled (comparable) zone changes, all abstract zone changes and compiled atom transports;
        # per qubit: number of anchors
        zone_changes: dict[str, Counter[tuple[int, int]]] = {"abstract": Counter(), "compiled": Counter()}
        all_zone_changes: Counter[tuple[int, int]] = Counter()  # abstract, incl. readout
        events: defaultdict[tuple[str, str, int], list[int]] = defaultdict(list)  # (source, kind, qubit) -> anchors
        moves: Counter[tuple[int, int]] = Counter()
        anchors = [0] * num_qubits
        for row in rows:
            if row["source"] == "circuit" and row["metric"] == "anchors":
                anchors[int(row["qubit"])] = int(row["value"])
            elif row["count"] and row["source"] != "per_gate":  # the per-gate model is only compared in the aggregated plots
                slot = (int(row["qubit"]), int(row["anchor"]))
                if row["metric"] in NA_COMPARABLE_ZONE_CHANGES:
                    zone_changes[row["source"]][slot] += int(row["count"])
                    events[(row["source"], row["metric"], slot[0])].append(slot[1])
                if row["source"] == "abstract" and row["metric"] in NA_ZONE_CHANGES:
                    all_zone_changes[slot] += int(row["count"])
                elif row["source"] == "compiled" and row["metric"] == NA_ATOM_MOVES:
                    moves[slot] += int(row["count"])
        if not any(anchors):
            print(f"Skipped {config} / {rows[0]['circuit']}: no anchor rows in the raw file (re-run the runner)")
            continue

        # timing shift: pair each qubit's abstract and compiled zone changes of the same kind
        circuit_shifts, circuit_missing, circuit_extra = [], 0, 0
        for kind in NA_COMPARABLE_ZONE_CHANGES:
            for qubit in range(num_qubits):
                paired, unpaired_abstract, unpaired_compiled = align(sorted(events[("abstract", kind, qubit)]),
                                                                     sorted(events[("compiled", kind, qubit)]))
                circuit_shifts += paired
                circuit_extra += unpaired_abstract
                circuit_missing += unpaired_compiled
        shifts += circuit_shifts
        missing += circuit_missing
        extra += circuit_extra

        # event timing and amount per qubit (slots before its last anchor: (qubit, k) for k < its number of anchors)
        abstract_per_qubit = [[all_zone_changes[(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
        moves_per_qubit = [[moves[(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
        qubit_shifts = [float(np.average(range(len(a)), weights=a) - np.average(range(len(m)), weights=m))
                        for a, m in zip(abstract_per_qubit, moves_per_qubit) if sum(a) and sum(m)]
        if qubit_shifts:
            qubit_shifts_per_circuit.append(qubit_shifts)
        abstract_total, moves_total = sum(map(sum, abstract_per_qubit)), sum(map(sum, moves_per_qubit))
        if abstract_total and moves_total:
            qubit_shares += [(sum(m) / moves_total, sum(a) / abstract_total, num_qubits)
                             for a, m in zip(abstract_per_qubit, moves_per_qubit) if len(a)]

        abstract_slots, compiled_slots = set(zone_changes["abstract"]), set(zone_changes["compiled"])
        scores.append({
            "circuit": rows[0]["circuit"],
            "num_qubits": num_qubits,
            **{f"f1_{'exact' if t == 0 else f'pm{t}'}": f1(abstract_slots, compiled_slots, t) for t in F1_TOLERANCES},
            "paired_zone_changes": len(circuit_shifts),
            "share_on_time": circuit_shifts.count(0) / len(circuit_shifts) if circuit_shifts else nan,
            "mean_shift": float(np.mean(circuit_shifts)) if circuit_shifts else nan,
            "missing_zone_changes": circuit_missing,
            "extra_zone_changes": circuit_extra,
            "qubits_early": float(np.mean(np.array(qubit_shifts) < -ON_TIME)) if qubit_shifts else nan,
            "qubits_on_time": float(np.mean(np.abs(qubit_shifts) <= ON_TIME)) if qubit_shifts else nan,
            "qubits_late": float(np.mean(np.array(qubit_shifts) > ON_TIME)) if qubit_shifts else nan,
            "event_shift_median": float(np.median(qubit_shifts)) if qubit_shifts else nan,
            "noise_shift_q25": float(np.percentile(qubit_shifts, 25)) if qubit_shifts else nan,
            "noise_shift_q75": float(np.percentile(qubit_shifts, 75)) if qubit_shifts else nan,
        })
    if not scores:
        return
    scores.sort(key=lambda c: (c["num_qubits"], c["circuit"]))

    # A) location agreement: F1 over the allowed anchor offset, one thin line per circuit, median bold (with its
    # values) and the middle 50% of circuits as a band
    keys = [key for key in scores[0] if key.startswith("f1_")]
    scored = [c for c in scores if not np.isnan(c["f1_exact"])]
    f1s = np.array([[c[k] for k in keys] for c in scored]).reshape(-1, len(keys))
    fig, ax = plt.subplots(figsize=(6, 7), facecolor=SURFACE)  # taller: room for the explanatory legend
    ax.axhline(1, color=REFERENCE, linewidth=4, zorder=1.6, label="abstract and concrete zone changes in the same slots")  # above the grid
    for row in f1s:
        ax.plot(F1_TOLERANCES, row, color=SERIES, alpha=0.18, linewidth=1, zorder=2)
    ax.plot([], [], color=SERIES, alpha=0.4, linewidth=1, label="one circuit")
    if len(f1s):
        ax.fill_between(F1_TOLERANCES, np.percentile(f1s, 25, axis=0), np.percentile(f1s, 75, axis=0), color=SERIES,
                        alpha=0.15, linewidth=0, zorder=2, label="middle 50% of circuits")
        medians = np.median(f1s, axis=0)
        ax.plot(F1_TOLERANCES, medians, color=SERIES, linewidth=2.5, marker="o", zorder=3, label="median circuit")
        for t, m in zip(F1_TOLERANCES, medians):
            ax.annotate(f"{m:.2f}", (t, m), textcoords="offset points", xytext=(0, -16), ha="center",
                        color=TEXT_SECONDARY, fontsize=9)
    ax.set_xticks(F1_TOLERANCES, [f"±{t}" if t else "exact" for t in F1_TOLERANCES])
    ax.set_ylim(0, 1.04)
    style(fig, ax, f"{config}: zone change location agreement", f"number of circuits = {len(scored)}",
          "allowed anchor offset", "F1 score of zone change slots",
          notes=("slot = one qubit's interval between two consecutive anchors",
                 "A / C = slots with ≥ 1 abstract / compiled zone change (storage ↔ EZ, EZ site)",
                 "precision (how many of the abstract events are real) ",
                 "     = share of A with a C slot within ±offset",
                 "recall (how many real places the model finds) ",
                 "     = share of C with an A slot within ±offset",
                 "F1 = 2 · precision · recall / (precision + recall)"))
    fig.savefig(f"{file_prefix}__location_agreement.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

    # A) timing shift: histogram of the shifts of all paired zone changes, plus the unpaired ones
    by_shift = Counter(shifts)
    span = max([1, *(abs(s) for s in shifts)])
    values = list(range(-span, span + 1))
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=SURFACE)
    ax.bar(values, [by_shift[v] for v in values], width=0.8, color=SERIES, zorder=3, label="paired zone changes")
    ax.bar([-span - 2, span + 2], [missing, extra], width=0.8, color=MUTED, zorder=3,
           label=f"unpaired zone changes: {missing} missing, {extra} extra")
    ax.axvline(0, color=REFERENCE, linewidth=3, zorder=4)
    ax.set_xticks([-span - 2, *values, span + 2], ["missing", *(f"{v:+d}" if v else "0" for v in values), "extra"])
    subtitle = "no paired zone changes"
    if shifts:
        subtitle = (f"{len(shifts)} paired: {shifts.count(0) / len(shifts):.0%} on time,"
                    f" {sum(s < 0 for s in shifts) / len(shifts):.0%} early, {sum(s > 0 for s in shifts) / len(shifts):.0%} late,"
                    f" mean shift {np.mean(shifts):+.2f} anchors")
    style(fig, ax, f"{config}: zone change timing (abstract vs compiled)", subtitle,
          "timing shift in anchors", "zone changes (all circuits)",
          notes=("shift = abstract - compiled anchor of a paired zone change (of the same kind)",
                 "< 0: events too early, > 0: too late",
                 "missing: compiled zone change without abstract one (too few events)",
                 "extra: abstract zone change without compiled one (too many events)"))
    fig.savefig(f"{file_prefix}__timing_shift.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

    # B) event timing: per qubit, where the weight of its abstract zone changes lies relative to the weight center of its
    # real (compiled) atom transports; per shift bin the share of a circuit's qubits (dots) and of the average circuit (bar)
    edges = [-np.inf, -2, -1, -0.5, -ON_TIME, ON_TIME, 0.5, 1, 2, np.inf]
    bin_labels = ["< -2", "-2…-1", "-1…-½", "−½…−¼", "±¼", "¼…½", "½…1", "1…2", "> 2"]
    center = bin_labels.index("±¼")
    per_circuit = [np.histogram(s, edges)[0] / len(s) for s in qubit_shifts_per_circuit]
    positions = np.arange(len(bin_labels))
    jitter = np.random.default_rng(0)  # fixed, so the plot is the same on every run
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    ax.axvline(center, color=REFERENCE, linewidth=4, zorder=1.6,  # above the grid
               label="weight center of the qubit's real atom transports")
    if per_circuit:
        ax.bar(positions, np.mean(per_circuit, axis=0), width=0.7, color=SERIES, alpha=0.25, zorder=2,
               label="average circuit")
        for shares in per_circuit:
            drawn = shares > 0  # circuits without qubits in a bin are left out, so the tails stay readable
            ax.scatter(positions[drawn] + jitter.uniform(-0.22, 0.22, drawn.sum()), shares[drawn], s=18,
                       color=SERIES, alpha=0.8, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.scatter([], [], s=18, color=SERIES, label="circuit")
    ax.set_xticks(positions, bin_labels)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, 1.04)
    scored = [c for c in scores if not np.isnan(c["qubits_on_time"])]
    style(fig, ax, f"{config}: event timing, abstract vs compiled", f"number of circuits = {len(scored)}",
          "anchor distance between the abstract event weight compared to the concrete event weight",
          "share of the circuit's qubits",
          notes=("per-qubit comparison of weight center (avg. anchor) of its compiled atom transports",
                 "and the weight center of its abstract zone changes (incl. readout)",
                 "< 0: abstract events too early on a qubit, > 0: too late"))
    fig.savefig(f"{file_prefix}__event_timing.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

    # B) event amount: per qubit its share of the circuit's abstract zone changes vs of its atom transports
    plot_event_amount(qubit_shares, f"{config}: event amount vs atom moves",
                      "qubit's share of the circuit's compiled atom transports", "qubit's share of the abstract zone changes",
                      "above: more abstract events than its movement share, below: less", f"{file_prefix}__event_amount.png")

    # table of the per-circuit scores (empty = undefined, e.g. no zone change slot in the circuit)
    with open(f"{file_prefix}__distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        for c in sorted(scores, key=lambda c: c["circuit"]):
            writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()})


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def visualize_sc_aggregated(config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    """Scatter aggregated abstract vs compiled metrics.

    One figure per hardware configuration (there is no intermediate metric: a compiled virtual swap is a physical SWAP):

    A + B) estimation of swaps
    - number of abstract VS compiled virtual swaps during the whole circuit

    (the counts are not expected to match, only to scale similarly)
    """
    # 1) one row per circuit with the summed counts (the raw files have one row per swapped qubit, i.e. two per swap)
    scores = []
    for rows in circuits:
        counts = Counter()  # (source, metric) -> summed count
        wall_clock_s = {}  # source -> run time in seconds
        for row in rows:
            if row["count"]:
                counts[(row["source"], row["metric"])] += int(row["count"])
            elif row["metric"] == "wall_clock_s":
                wall_clock_s[row["source"]] = float(row["value"])
        scores.append({
            "circuit": rows[0]["circuit"],
            "num_qubits": int(rows[0]["num_qubits"]),
            "anchors": sum(int(row["value"]) for row in rows if row["metric"] == "anchors"),  # circuit size
            "abstract_swaps": counts[("abstract", SC_SWAPS)] // 2,
            "compiled_swaps": counts[("compiled", SC_SWAPS)] // 2,
            "per_gate_swaps": counts[("per_gate", SC_SWAPS)] // 2,
            "abstract_wall_clock_s": wall_clock_s["abstract"],
            "compiled_wall_clock_s": wall_clock_s["compiled"],
        })
    scores.sort(key=lambda c: c["num_qubits"])  # larger circuits drawn on top
    qubits = [c["num_qubits"] for c in scores]
    mean_speedup = np.mean([c["compiled_wall_clock_s"] / c["abstract_wall_clock_s"] for c in scores])  # per-circuit quotient, then mean

    # 2) the plots: (file name, title, x column, per-gate x column, y column, x label, y label, same quantity on both axes)
    plots = [
        ("swaps", "virtual swaps", "abstract_swaps", "per_gate_swaps", "compiled_swaps",
         "abstract virtual swaps", "compiled swaps", False),
    ]
    for name, title, x_key, per_gate_key, y_key, x_label, y_label, same_quantity in plots:
        xs = [c[x_key] for c in scores]
        xs_per_gate = [c[per_gate_key] for c in scores]
        ys = [c[y_key] for c in scores]
        fig, ax = plt.subplots(figsize=(7, 7.6) if same_quantity else (7.6, 9), facecolor=SURFACE)  # proxy: longer, wider legend

        # logarithmic axes (counts over orders of magnitude, zeros in their own strip), and the y = x line if both axes show
        # the same quantity; the dots are drawn at their axis positions (px, py), the fits and scores use the counts
        x_lo, x_end, x_zero = log_count_range([*xs, *xs_per_gate, *(ys if same_quantity else [])])
        y_lo, y_end, y_zero = (x_lo, x_end, x_zero) if same_quantity else log_count_range(ys)
        set_log_count_axis(ax, "x", x_lo, x_end, x_zero)
        set_log_count_axis(ax, "y", y_lo, y_end, y_zero)
        px, px_per_gate, py = on_log_count_axis(xs, x_zero), on_log_count_axis(xs_per_gate, x_zero), on_log_count_axis(ys, y_zero)
        if same_quantity:
            diagonal = [x_zero or x_lo, x_end]  # through the zero strip's (0, 0) as well
            ax.plot(diagonal, diagonal, color=REFERENCE, linewidth=3, zorder=2, label="exact match")

        subtitle = f"number of circuits = {len(xs)}   avg. speedup of abstract execution vs compilation: {mean_speedup:.3g}×"

        # the circuits: per-gate model hollow, abstract model colored by qubit number; one circuit after the other its white outline,
        # then the line connecting its two points (see connect_per_gate), then its fill, so the line reaches the dot over the outline
        ax.scatter(px_per_gate, py, s=RING_SIZE, facecolors="none", edgecolors=PER_GATE, linewidths=RING_WIDTH, zorder=3)
        points = ScalarMappable(Normalize(QUBIT_MIN, QUBIT_MAX), QUBIT_CMAP)
        for i, (x, y, qubit) in enumerate(zip(px, py, qubits)):
            ax.scatter([x], [y], s=56, color=SURFACE, edgecolors=SURFACE, linewidths=1.6, zorder=circuit_zorder(i))
            ax.scatter([x], [y], s=56, color=points.to_rgba(qubit), linewidths=0, zorder=circuit_zorder(i) + 2e-5)
        headers = []  # bold legend section headers
        if same_quantity:
            headers = equivalence_scores(ax, xs, xs_per_gate, ys, [c["anchors"] for c in scores], diagonal[0], x_end)
        else:  # a proxy: correlation and proportionality instead of equivalence
            headers = proxy_fit_and_scores(ax, xs, xs_per_gate, ys, [c["anchors"] for c in scores], x_lo, x_end, y_lo)
        colorbar = fig.colorbar(points, ax=ax, ticks=QUBIT_TICKS, fraction=0.05, pad=0.03)
        colorbar.set_label("qubits in circuit", color=TEXT_SECONDARY, fontsize=10)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=MUTED, labelcolor=TEXT_SECONDARY, labelsize=9)

        style(fig, ax, f"{config}: {title}", subtitle, x_label, y_label)
        for text in ax.get_legend().get_texts():
            if text.get_text() in headers:
                text.set_fontweight("bold")
                text.set_color(TEXT)
        connect_per_gate(fig, ax, px, px_per_gate, py)  # after the layout is fixed
        fig.savefig(f"{file_prefix}__{name}.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)

    # 3) table of the plotted numbers
    with open(f"{file_prefix}__aggregated.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(sorted(scores, key=lambda c: c["circuit"]))


def visualize_sc_distribution(config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    """Location-based metrics, i.e. where in the circuit hardware events happen.

    Two figures per hardware configuration, the same as part B for TI (the IZ change vs ion move perspective): the compiled
    swaps are far more than the abstract ones, so they are not matched 1 to 1, only their per-qubit distribution is compared;
    a swap counts for both of its qubits.

    - event timing: per qubit, [shift = average anchor of its abstract swaps - average anchor of its compiled swaps],
                    i.e. abstract events inserted earlier or later than concrete events
                    (the optimal outcome is for all data points to lie on 0)
    - event amount: per qubit, its share (number of events of qubit / number of events in general) of its
                    abstract swaps vs its compiled swaps over the whole circuit
                    (the optimal outcome is for all data points to lie on the angle bisector)

    """
    # 1) one score row per circuit, plus the per-event / per-qubit values pooled over all circuits
    scores = []
    qubit_shifts_per_circuit = []  # per circuit: the event timing shift of each qubit
    qubit_shares = []  # per qubit: (share of the circuit's compiled swaps, share of the circuit's abstract swaps)
    nan = float("nan")
    for rows in circuits:
        num_qubits = int(rows[0]["num_qubits"])

        # per slot: abstract / compiled virtual swaps and compiled swaps; per qubit: number of anchors
        swaps: dict[str, Counter[tuple[int, int]]] = {"abstract": Counter(), "compiled": Counter()}
        moves: Counter[tuple[int, int]] = Counter()
        anchors = [0] * num_qubits
        for row in rows:
            if row["source"] == "circuit" and row["metric"] == "anchors":
                anchors[int(row["qubit"])] = int(row["value"])
            elif row["count"] and row["source"] != "per_gate":  # the per-gate model is only compared in the aggregated plots
                slot = (int(row["qubit"]), int(row["anchor"]))
                if row["metric"] == SC_SWAPS:
                    swaps[row["source"]][slot] += int(row["count"])
                elif row["source"] == "compiled" and row["metric"] == SC_COMPILED_SWAPS:
                    moves[slot] += int(row["count"])
        if not any(anchors):
            print(f"Skipped {config} / {rows[0]['circuit']}: no anchor rows in the raw file (re-run the runner)")
            continue

        # event timing and amount per qubit (slots before its last anchor: (qubit, k) for k < its number of anchors)
        abstract_per_qubit = [[swaps["abstract"][(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
        moves_per_qubit = [[moves[(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
        qubit_shifts = [float(np.average(range(len(a)), weights=a) - np.average(range(len(m)), weights=m))
                        for a, m in zip(abstract_per_qubit, moves_per_qubit) if sum(a) and sum(m)]
        if qubit_shifts:
            qubit_shifts_per_circuit.append(qubit_shifts)
        abstract_total, moves_total = sum(map(sum, abstract_per_qubit)), sum(map(sum, moves_per_qubit))
        if abstract_total and moves_total:
            qubit_shares += [(sum(m) / moves_total, sum(a) / abstract_total, num_qubits)
                             for a, m in zip(abstract_per_qubit, moves_per_qubit) if len(a)]

        scores.append({
            "circuit": rows[0]["circuit"],
            "num_qubits": num_qubits,
            "abstract_swaps": sum(swaps["abstract"].values()) // 2,
            "compiled_swaps": sum(swaps["compiled"].values()) // 2,
            "qubits_early": float(np.mean(np.array(qubit_shifts) < -ON_TIME)) if qubit_shifts else nan,
            "qubits_on_time": float(np.mean(np.abs(qubit_shifts) <= ON_TIME)) if qubit_shifts else nan,
            "qubits_late": float(np.mean(np.array(qubit_shifts) > ON_TIME)) if qubit_shifts else nan,
            "event_shift_median": float(np.median(qubit_shifts)) if qubit_shifts else nan,
            "noise_shift_q25": float(np.percentile(qubit_shifts, 25)) if qubit_shifts else nan,
            "noise_shift_q75": float(np.percentile(qubit_shifts, 75)) if qubit_shifts else nan,
        })
    if not scores:
        return
    scores.sort(key=lambda c: (c["num_qubits"], c["circuit"]))

    # event timing: per qubit, where the weight of its abstract swaps lies relative to the weight center of its
    # compiled swaps; per shift bin the share of a circuit's qubits (dots) and of the average circuit (bar)
    edges = [-np.inf, -2, -1, -0.5, -ON_TIME, ON_TIME, 0.5, 1, 2, np.inf]
    bin_labels = ["< -2", "-2…-1", "-1…-½", "−½…−¼", "±¼", "¼…½", "½…1", "1…2", "> 2"]
    center = bin_labels.index("±¼")
    per_circuit = [np.histogram(s, edges)[0] / len(s) for s in qubit_shifts_per_circuit]
    positions = np.arange(len(bin_labels))
    jitter = np.random.default_rng(0)  # fixed, so the plot is the same on every run
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE)
    ax.axvline(center, color=REFERENCE, linewidth=4, zorder=1.6,  # above the grid
               label="weight center of the qubit's compiled swaps")
    if per_circuit:
        ax.bar(positions, np.mean(per_circuit, axis=0), width=0.7, color=SERIES, alpha=0.25, zorder=2,
               label="average circuit")
        for shares in per_circuit:
            drawn = shares > 0  # circuits without qubits in a bin are left out, so the tails stay readable
            ax.scatter(positions[drawn] + jitter.uniform(-0.22, 0.22, drawn.sum()), shares[drawn], s=18,
                       color=SERIES, alpha=0.8, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    ax.scatter([], [], s=18, color=SERIES, label="circuit")
    ax.set_xticks(positions, bin_labels)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_ylim(0, 1.04)
    scored = [c for c in scores if not np.isnan(c["qubits_on_time"])]
    style(fig, ax, f"{config}: event timing, abstract vs compiled", f"number of circuits = {len(scored)}",
          "anchor distance between the abstract event weight compared to the concrete event weight",
          "share of the circuit's qubits",
          notes=("per-qubit comparison of weight center (avg. anchor) of its compiled swaps",
                 "and the weight center of its abstract swaps",
                 "< 0: abstract events too early on a qubit, > 0: too late"))
    fig.savefig(f"{file_prefix}__event_timing.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

    # event amount: per qubit its share of the circuit's abstract swaps vs of its compiled swaps
    plot_event_amount(qubit_shares, f"{config}: event amount vs compiled swaps",
                      "qubit's share of the circuit's compiled swaps", "qubit's share of the abstract swaps",
                      "above: more abstract events than its compiled share, below: less", f"{file_prefix}__event_amount.png",
                      zero_events="swaps")

    # table of the per-circuit scores (empty = undefined, e.g. no swap slot in the circuit)
    with open(f"{file_prefix}__distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        for c in sorted(scores, key=lambda c: c["circuit"]):
            writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()})
