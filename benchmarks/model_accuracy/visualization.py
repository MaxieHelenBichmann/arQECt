"""Extract and aggregate relevant metrics comparing the abstract execution and concrete compilation, and visualize them.

I want to answer two questions:
A) How well does the abstract hardware model approximate certain concrete hardware events compared to the actually compiled circuit?
B) How well do the abstract hardware events function as a proxy for the relevant concrete hardware events in the compiled circuit?

And "well" is defined as resulting in a suitable (either similar (A) or having a similar scaling behavior (B)) number of hardware events
aggregated over the whole circuit (important for structural cost computation), as well as having a suitable distribution of hardware events 
throughout the circuit (important for noise injection).
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only
import matplotlib.pyplot as plt
import numpy as np
import stim
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr, theilslopes


def visualize(platform: list[str], results_dir: Path, max_qubits: int | None = None) -> None:
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if "TI" in platform:
        for old_file in out_dir.glob("ti_*"):  # remove previous TI outputs (also of configs that no longer have results)
            old_file.unlink()
        visualize_ti_aggregated(results_dir, max_qubits)
        visualize_ti_distribution(results_dir, max_qubits)
    if "NA" in platform:
        pass
    if "SC" in platform:
        pass
    
def visualize_ti_aggregated(results_dir: Path, max_qubits: int | None = None) -> None:
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
    # colors: surface, text, grid / axes, fit line, and a blue ramp for the qubit number of a circuit
    surface, text, text_secondary, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    grid, baseline, fit, red = "#e1e0d9", "#c3c2b7", "#0d366b", "#e34948"
    qubit_cmap = LinearSegmentedColormap.from_list("qubits", ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                                                              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
    qubit_min, qubit_max = 4, 20  # fixed, so the same color means the same qubit number in every plot

    # 1) one row per circuit with the summed counts, grouped by hardware config
    runs = defaultdict(list)  # config -> list of circuit rows
    for file in sorted((results_dir / "raw" / "ti").glob("*.csv")):
        with open(file, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or (max_qubits is not None and int(rows[0]["num_qubits"]) > max_qubits):
            continue
        counts = Counter()  # "source metric" -> summed count
        wall_clock_s = {}  # source -> run time in seconds
        for row in rows:
            if row["count"]:
                counts[f"{row['source']} {row['metric']}"] += int(row["count"])
            elif row["metric"] == "wall_clock_s":
                wall_clock_s[row["source"]] = float(row["value"])
        runs[rows[0]["config"]].append({
            "circuit": rows[0]["circuit"],
            "num_qubits": int(rows[0]["num_qubits"]),
            "abstract_iz_entries": counts["abstract interaction_zone_entry"],
            "abstract_iz_evictions": counts["abstract interaction_zone_eviction"],
            "abstract_iz_changes": counts["abstract interaction_zone_entry"] + counts["abstract interaction_zone_eviction"],
            "compiled_iz_entries": counts["compiled interaction_zone_entry"],
            "compiled_iz_evictions": counts["compiled interaction_zone_eviction"],
            "compiled_iz_changes": counts["compiled interaction_zone_entry"] + counts["compiled interaction_zone_eviction"],
            "compiled_ion_moves": counts["compiled shuttles"] + counts["compiled swaps"],  # swaps are counted per ion
            "abstract_wall_clock_s": wall_clock_s["abstract"],
            "compiled_wall_clock_s": wall_clock_s["compiled"],
        })

    # 2) the plots: (file name, title, x column, y column, x label, y label, same quantity on both axes)
    plots = [
        ("iz_entries", "IZ entries", "abstract_iz_entries", "compiled_iz_entries",
         "abstract IZ entries", "compiled IZ entries", True),
        ("iz_evictions", "IZ evictions", "abstract_iz_evictions", "compiled_iz_evictions",
         "abstract IZ evictions", "compiled IZ evictions", True),
        ("iz_changes", "IZ changes (entries + evictions)", "abstract_iz_changes", "compiled_iz_changes",
         "abstract IZ changes", "compiled IZ changes", True),
        ("iz_changes_vs_ion_moves", "IZ changes vs ion moves", "abstract_iz_changes", "compiled_ion_moves",
         "abstract IZ changes", "compiled shuttles + swaps", False),
    ]

    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for config, circuits in sorted(runs.items()):
        slug = re.sub(r"[^a-z0-9]+", "_", config.lower()).strip("_")
        circuits.sort(key=lambda c: c["num_qubits"])  # larger circuits drawn on top
        qubits = [c["num_qubits"] for c in circuits]
        mean_compile_s = np.mean([c["compiled_wall_clock_s"] for c in circuits])
        mean_abstract_s = np.mean([c["abstract_wall_clock_s"] for c in circuits])

        for name, title, x_key, y_key, x_label, y_label, same_quantity in plots:
            xs = [c[x_key] for c in circuits]
            ys = [c[y_key] for c in circuits]

            # axes styling
            fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=surface)
            ax.set_facecolor(surface)
            ax.grid(color=grid, linewidth=0.8, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(baseline)
            ax.tick_params(colors=muted, labelcolor=text_secondary, labelsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # counts: integer ticks only
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            # axis ranges, and the y = x line if both axes show the same quantity
            x_end = max([*xs, 1]) * 1.08
            y_end = max([*ys, 1]) * 1.08
            if same_quantity:
                x_end = y_end = max(x_end, y_end)
                ax.plot([0, x_end], [0, x_end], color=red, linewidth=3, zorder=2, label="exact match")
            ax.set_xlim(0, x_end)
            ax.set_ylim(0, y_end)

            # linear fits: robust Theil-Sen (with 95% CI of the slope) and least squares (with R²)
            subtitle = f"number of circuits = {len(xs)}"
            if len(set(xs)) >= 2 and len(set(ys)) >= 2:
                slope, intercept, slope_low, slope_high = theilslopes(ys, xs)
                ax.plot([0, x_end], [intercept, slope * x_end + intercept], color=fit, linewidth=2, zorder=2,
                        label=f"robust fit (Theil-Sen), slope 95% CI {slope_low:.2f}–{slope_high:.2f}")
                slope, intercept = np.polyfit(xs, ys, 1)
                r_squared = np.corrcoef(xs, ys)[0, 1] ** 2
                ax.plot([0, x_end], [intercept, slope * x_end + intercept], color=fit, linewidth=1.2,
                        linestyle="--", zorder=2, label=f"least-squares fit, R² = {r_squared:.2f}")
                if len(xs) >= 3:
                    subtitle += f"   Pearson r = {pearsonr(xs, ys).statistic:.2f}"
                    subtitle += f"   Spearman ρ = {spearmanr(xs, ys).statistic:.2f}"
            ax.plot([], [], linestyle="none",  # text-only legend entry
                    label=f"avg. run time: compilation {mean_compile_s:.3g} s vs abstract execution {mean_abstract_s:.3g} s")

            # the circuits, colored by qubit number
            points = ax.scatter(xs, ys, s=56, c=qubits, cmap=qubit_cmap, vmin=qubit_min, vmax=qubit_max,
                                edgecolors=surface, linewidths=1.5, zorder=3)
            colorbar = fig.colorbar(points, ax=ax, ticks=range(qubit_min, qubit_max + 1, 4), fraction=0.05, pad=0.03)
            colorbar.set_label("qubits in circuit", color=text_secondary, fontsize=10)
            colorbar.outline.set_visible(False)
            colorbar.ax.tick_params(colors=muted, labelcolor=text_secondary, labelsize=9)

            # title, subtitle, axis labels, legend below the plot
            ax.set_title(f"{config}: {title}", loc="left", color=text, fontsize=12, fontweight="bold", pad=22)
            ax.text(0, 1.02, subtitle, transform=ax.transAxes, va="bottom", ha="left", color=text_secondary, fontsize=9)
            ax.set_xlabel(x_label, color=text_secondary, fontsize=10)
            ax.set_ylabel(y_label, color=text_secondary, fontsize=10)
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), frameon=False, fontsize=9,
                      labelcolor=text_secondary, handlelength=2.4)
            fig.tight_layout()
            fig.savefig(out_dir / f"ti_{slug}__{name}.png", dpi=200, facecolor=surface)
            plt.close(fig)

        # 3) table of the plotted numbers
        with open(out_dir / f"ti_{slug}__aggregated.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(circuits[0]))
            writer.writeheader()
            writer.writerows(sorted(circuits, key=lambda c: c["circuit"]))


def visualize_ti_distribution(results_dir: Path, max_qubits: int | None = None) -> None:
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
    surface, text, text_secondary, muted = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    grid, baseline, series, red = "#e1e0d9", "#c3c2b7", "#2a78d6", "#e34948"
    circuit_dir = Path(__file__).resolve().parents[2] / "data" / "model_accuracy" / "circuits"

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

    # 1) one score row per circuit, grouped by hardware config, plus the pooled per-event / per-qubit values
    runs = defaultdict(list)
    pooled: defaultdict[str, dict] = defaultdict(lambda: {"shifts": [], "missing": 0, "extra": 0, "qubit_shares": [],
                                                          "qubit_shifts_per_circuit": []})
    for file in sorted((results_dir / "raw" / "ti").glob("*.csv")):
        with open(file, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or (max_qubits is not None and int(rows[0]["num_qubits"]) > max_qubits):
            continue
        circuit_file = circuit_dir / f"{rows[0]['circuit']}.stim"
        if not circuit_file.exists():
            print(f"Skipped {file.name}: circuit file {circuit_file.name} not found")
            continue

        # per slot: abstract / compiled IZ changes and compiled ion moves
        iz_changes: dict[str, Counter[tuple[int, int]]] = {"abstract": Counter(), "compiled": Counter()}
        events: defaultdict[tuple[str, str, int], list[int]] = defaultdict(list)  # (source, kind, qubit) -> anchors
        moves: Counter[tuple[int, int]] = Counter()
        for row in rows:
            if not row["count"]:
                continue
            slot = (int(row["qubit"]), int(row["anchor"]))
            if row["metric"] in ("interaction_zone_entry", "interaction_zone_eviction"):
                iz_changes[row["source"]][slot] += int(row["count"])
                events[(row["source"], row["metric"], slot[0])].append(slot[1])
            elif row["source"] == "compiled" and row["metric"] in ("shuttles", "swaps"):
                moves[slot] += int(row["count"])

        # all slots before an anchor: (qubit, k) for k < number of anchors of that qubit
        circuit = stim.Circuit.from_file(circuit_file)
        anchors = Counter(t.value for op in circuit for group in op.target_groups() for t in group
                          if stim.gate_data(op.name).is_two_qubit_gate or stim.gate_data(op.name).produces_measurements
                          or stim.gate_data(op.name).is_reset)
        config = rows[0]["config"]

        # timing shift: pair each qubit's abstract and compiled IZ changes of the same kind
        shifts, missing, extra = [], 0, 0
        for kind in ("interaction_zone_entry", "interaction_zone_eviction"):
            for qubit in range(circuit.num_qubits):
                paired, unpaired_abstract, unpaired_compiled = align(sorted(events[("abstract", kind, qubit)]),
                                                                     sorted(events[("compiled", kind, qubit)]))
                shifts += paired
                extra += unpaired_abstract
                missing += unpaired_compiled
        pooled[config]["shifts"] += shifts
        pooled[config]["missing"] += missing
        pooled[config]["extra"] += extra

        # event timing and amount per qubit (slots before its last anchor)
        abstract_per_qubit = [[iz_changes["abstract"][(q, k)] for k in range(anchors[q])] for q in range(circuit.num_qubits)]
        moves_per_qubit = [[moves[(q, k)] for k in range(anchors[q])] for q in range(circuit.num_qubits)]
        noise_shifts = [float(np.average(range(len(a)), weights=a) - np.average(range(len(m)), weights=m))
                        for a, m in zip(abstract_per_qubit, moves_per_qubit) if sum(a) and sum(m)]
        if noise_shifts:
            pooled[config]["qubit_shifts_per_circuit"].append(noise_shifts)
        abstract_total, moves_total = sum(map(sum, abstract_per_qubit)), sum(map(sum, moves_per_qubit))
        if abstract_total and moves_total:
            pooled[config]["qubit_shares"] += [(sum(m) / moves_total, sum(a) / abstract_total, circuit.num_qubits)
                                               for a, m in zip(abstract_per_qubit, moves_per_qubit) if len(a)]

        abstract_slots, compiled_slots = set(iz_changes["abstract"]), set(iz_changes["compiled"])
        nan = float("nan")
        runs[config].append({
            "circuit": rows[0]["circuit"],
            "num_qubits": circuit.num_qubits,
            "f1_exact": f1(abstract_slots, compiled_slots, 0),
            "f1_pm1": f1(abstract_slots, compiled_slots, 1),
            "f1_pm2": f1(abstract_slots, compiled_slots, 2),
            "f1_pm3": f1(abstract_slots, compiled_slots, 3),
            "paired_iz_changes": len(shifts),
            "share_on_time": shifts.count(0) / len(shifts) if shifts else nan,
            "mean_shift": float(np.mean(shifts)) if shifts else nan,
            "missing_iz_changes": missing,
            "extra_iz_changes": extra,
            "qubits_early": float(np.mean(np.array(noise_shifts) < -0.25)) if noise_shifts else nan,
            "qubits_on_time": float(np.mean(np.abs(noise_shifts) <= 0.25)) if noise_shifts else nan,
            "qubits_late": float(np.mean(np.array(noise_shifts) > 0.25)) if noise_shifts else nan,
            "event_shift_median": float(np.median(noise_shifts)) if noise_shifts else nan,
            "noise_shift_q25": float(np.percentile(noise_shifts, 25)) if noise_shifts else nan,
            "noise_shift_q75": float(np.percentile(noise_shifts, 75)) if noise_shifts else nan,
        })

    def style(fig, ax, title: str, subtitle: str, x_label: str, y_label: str) -> None:
        ax.set_facecolor(surface)
        ax.grid(color=grid, linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(baseline)
        ax.tick_params(colors=muted, labelcolor=text_secondary, labelsize=9)
        ax.set_title(title, loc="left", color=text, fontsize=12, fontweight="bold", pad=22)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, va="bottom", ha="left", color=text_secondary, fontsize=9)
        ax.set_xlabel(x_label, color=text_secondary, fontsize=10)
        ax.set_ylabel(y_label, color=text_secondary, fontsize=10)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), frameon=False, fontsize=9, labelcolor=text_secondary,
                  handlelength=2.4)
        fig.tight_layout()

    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for config, circuits in sorted(runs.items()):
        slug = re.sub(r"[^a-z0-9]+", "_", config.lower()).strip("_")
        circuits.sort(key=lambda c: (c["num_qubits"], c["circuit"]))

        # A) location agreement: F1 over the allowed anchor offset, one thin line per circuit, median bold (with its
        # values) and the middle 50% of circuits as a band
        tolerances, keys = [0, 1, 2, 3], ["f1_exact", "f1_pm1", "f1_pm2", "f1_pm3"]
        scored = [c for c in circuits if not np.isnan(c["f1_exact"])]
        f1s = np.array([[c[k] for k in keys] for c in scored]).reshape(-1, len(keys))
        fig, ax = plt.subplots(figsize=(6, 7), facecolor=surface)  # taller: room for the explanatory legend
        ax.axhline(1, color=red, linewidth=4, zorder=1.6, label="abstract and concrete IZ changes in the same slots")  # above the grid
        for row in f1s:
            ax.plot(tolerances, row, color=series, alpha=0.18, linewidth=1, zorder=2)
        ax.plot([], [], color=series, alpha=0.4, linewidth=1, label="one circuit")
        if len(f1s):
            ax.fill_between(tolerances, np.percentile(f1s, 25, axis=0), np.percentile(f1s, 75, axis=0), color=series,
                            alpha=0.15, linewidth=0, zorder=2, label="middle 50% of circuits")
            medians = np.median(f1s, axis=0)
            ax.plot(tolerances, medians, color=series, linewidth=2.5, marker="o", zorder=3, label="median circuit")
            for t, m in zip(tolerances, medians):
                ax.annotate(f"{m:.2f}", (t, m), textcoords="offset points", xytext=(0, -16), ha="center",
                            color=text_secondary, fontsize=9)
        for line in ("slot = one qubit's interval between two consecutive anchors",
                     "A / C = slots with ≥ 1 abstract / compiled IZ change",
                     "precision (how many of the abstract events are real) ",
                     "     = share of A with a C slot within ±offset",
                     "recall (how many real places the model finds) ",
                     "     = share of C with an A slot within ±offset",
                     "F1 = 2 · precision · recall / (precision + recall)"):
            ax.plot([], [], linestyle="none", label=line)  # text-only legend entry
        ax.set_xticks(tolerances, ["exact", "±1", "±2", "±3"])
        ax.set_ylim(0, 1.04)
        subtitle = f"number of circuits = {len(scored)}"
        style(fig, ax, f"{config}: IZ change location agreement", subtitle,
              "allowed anchor offset", "F1 score of IZ change slots")
        fig.savefig(out_dir / f"ti_{slug}__location_agreement.png", dpi=200, facecolor=surface)
        plt.close(fig)

        # A) timing shift: histogram of the shifts of all paired IZ changes, plus the unpaired ones
        shifts, missing, extra = pooled[config]["shifts"], pooled[config]["missing"], pooled[config]["extra"]
        by_shift = Counter(shifts)
        span = max([1, *(abs(s) for s in shifts)])
        values = list(range(-span, span + 1))
        fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=surface)
        ax.bar(values, [by_shift[v] for v in values], width=0.8, color=series, zorder=3, label="paired IZ changes")
        ax.bar([-span - 2, span + 2], [missing, extra], width=0.8, color=muted, zorder=3,
               label=f"unpaired IZ changes: {missing} missing, {extra} extra")
        ax.axvline(0, color=red, linewidth=3, zorder=4)
        ax.set_xticks([-span - 2, *values, span + 2], ["missing", *(f"{v:+d}" if v else "0" for v in values), "extra"])
        for line in ("shift = abstract - compiled anchor of a paired IZ change",
                     "< 0: events too early, > 0: too late",
                     "missing: compiled IZ change without abstract one (too few events)",
                     "extra: abstract IZ change without compiled one (too many events)"):
            ax.plot([], [], linestyle="none", label=line)
        subtitle = "no paired IZ changes"
        if shifts:
            subtitle = (f"{len(shifts)} paired: {shifts.count(0) / len(shifts):.0%} on time,"
                        f" {sum(s < 0 for s in shifts) / len(shifts):.0%} early, {sum(s > 0 for s in shifts) / len(shifts):.0%} late,"
                        f" mean shift {np.mean(shifts):+.2f} anchors")
        style(fig, ax, f"{config}: IZ change timing (abstract vs compiled)", subtitle,
              "timing shift in anchors", "IZ changes (all circuits)")
        fig.savefig(out_dir / f"ti_{slug}__timing_shift.png", dpi=200, facecolor=surface)
        plt.close(fig)

        # B) event timing: per qubit, where the weight of its abstract IZ changes lies relative to the weight center of its
        # real (compiled) ion moves; per shift bin the share of a circuit's qubits (dots) and of the average circuit (bar)
        edges = [-np.inf, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, np.inf]
        bin_labels = ["< -2", "-2…-1", "-1…-½", "−½…−¼", "±¼", "¼…½", "½…1", "1…2", "> 2"]
        center = bin_labels.index("±¼")
        per_circuit = [np.histogram(s, edges)[0] / len(s) for s in pooled[config]["qubit_shifts_per_circuit"]]
        positions = np.arange(len(bin_labels))
        jitter = np.random.default_rng(0)  # fixed, so the plot is the same on every run
        fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=surface)
        ax.axvline(center, color=red, linewidth=4, zorder=1.6,  # above the grid
                   label="weight center of the qubit's real ion moves")
        if per_circuit:
            ax.bar(positions, np.mean(per_circuit, axis=0), width=0.7, color=series, alpha=0.25, zorder=2,
                   label="average circuit")
            for shares in per_circuit:
                drawn = shares > 0  # circuits without qubits in a bin are left out, so the tails stay readable
                ax.scatter(positions[drawn] + jitter.uniform(-0.22, 0.22, drawn.sum()), shares[drawn], s=18,
                           color=series, alpha=0.8, edgecolors=surface, linewidths=0.6, zorder=3)
        ax.scatter([], [], s=18, color=series, label="circuit")
        for line in ("per-qubit comparison of weight center (avg. anchor) of its compiled ion moves",
                     "and the weight center of its abstract IZ changes",
                     "< 0: abstract events too early on a qubit, > 0: too late"):
            ax.plot([], [], linestyle="none", label=line)
        ax.set_xticks(positions, bin_labels)
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.set_ylim(0, 1.04)
        scored = [c for c in circuits if not np.isnan(c["qubits_on_time"])]
        subtitle = f"number of circuits = {len(scored)}"
        style(fig, ax, f"{config}: event timing, abstract vs compiled", subtitle,
              "anchor distance between the abstract event weight compared to the concrete event weight", "share of the circuit's qubits")
        fig.savefig(out_dir / f"ti_{slug}__event_timing.png", dpi=200, facecolor=surface)
        plt.close(fig)

        # B) event amount: per qubit its share of the circuit's abstract IZ changes vs of its ion moves
        shares = pooled[config]["qubit_shares"]
        fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=surface)
        end = max([0.1, *(max(m, a) for m, a, _ in shares)]) * 1.08
        ax.plot([0, end], [0, end], color=red, linewidth=3, zorder=2, label="exact match")
        ax.scatter([m for m, _, _ in shares], [a for _, a, _ in shares], s=40, color=series, alpha=0.75,
                   edgecolors=surface, linewidths=1, zorder=3)
        for line in ("above: more abstract events than its movement share, below: less",
                    "one data point = one qubit, shares within its circuit"):
            ax.plot([], [], linestyle="none", label=line)
        ax.set_xlim(0, end)
        ax.set_ylim(0, end)
        subtitle = f"number of qubits = {len(shares)}"
        if shares:
            subtitle += (f"   > 1.5× its movement share: {sum(a > 1.5 * m for m, a, _ in shares) / len(shares):.0%},"
                         f" < 1/1.5×: {sum(1.5 * a < m for m, a, _ in shares) / len(shares):.0%}")
        style(fig, ax, f"{config}: event amount vs ion moves", subtitle,
              "qubit's share of the circuit's compiled ion moves", "qubit's share of the abstract IZ changes")
        fig.savefig(out_dir / f"ti_{slug}__event_amount.png", dpi=200, facecolor=surface)
        plt.close(fig)

        # table of the per-circuit scores (empty = undefined, e.g. no IZ change slot in the circuit)
        with open(out_dir / f"ti_{slug}__distribution.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(circuits[0]))
            writer.writeheader()
            for c in sorted(circuits, key=lambda c: c["circuit"]):
                writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()})