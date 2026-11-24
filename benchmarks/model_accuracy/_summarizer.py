"""Summarize abstract hardware events, compiled events and physical movement into the intermediate csvs of the plots.

Aggregated event counts assess agreement or proxy scaling (correlation + proportionality).
Location scores compare occupied slots; timing alignment also distinguishes event kinds.
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr, theilslopes

BAND_SHARE = 0.8
BOOTSTRAP_SAMPLES = 500
F1_TOLERANCES = (0, 1, 2, 3)
ON_TIME = 0.25
ALIGNMENT_GAP = 2.0

TI_IZ_CHANGES = ("interaction_zone_entry", "interaction_zone_eviction")
TI_ION_MOVES = ("shuttles", "swaps")  # swaps count once per participating ion
NA_ZONE_CHANGES = ("storage_entanglement", "storage_readout", "entanglement_readout", "inter_entanglement")


def summarize(platform: list[str], results_dir: Path, max_qubits: int) -> None:
    for hardware in ("TI", "NA", "SC"):
        if hardware not in platform:
            continue
        prefix = hardware.lower()
        out_dir = results_dir / "plots" / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        for old_file in out_dir.glob(f"{prefix}_*__*"):
            if old_file.is_file() and old_file.suffix in (".png", ".csv"):
                old_file.unlink()

        runs = defaultdict(list)
        for file in sorted((results_dir / "raw" / prefix).glob("*.csv")):
            with file.open(newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and int(rows[0]["num_qubits"]) <= max_qubits:
                runs[rows[0]["config"]].append(rows)
        for config, circuits in sorted(runs.items()):
            name = re.sub(r"[^a-z0-9]+", "_", config.lower()).strip("_")
            file_prefix = f"{out_dir}/{prefix}_{name}"
            summarize_aggregated(hardware, config, circuits, file_prefix)
            summarize_distribution(hardware, config, circuits, file_prefix)


def compiler_groups(items: list, compiler_of) -> list[tuple[str, int, list]]:
    """Split items by compiler: (compiler, index for its color / marker, items), with QMAP first."""
    compilers = sorted({compiler_of(item) for item in items}, key=lambda compiler: (compiler != "QMAP", compiler))
    return [(compiler, g, [item for item in items if compiler_of(item) == compiler]) for g, compiler in enumerate(compilers)]


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


def summarize_aggregated(platform: str, config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    if not circuits:
        return
    if platform == "TI":
        metrics = {
            "iz_entries": ("interaction_zone_entry",),
            "iz_evictions": ("interaction_zone_eviction",),
            "iz_changes": TI_IZ_CHANGES,
        }
        compiled_metrics = {**metrics, "ion_moves": TI_ION_MOVES}
    elif platform == "NA":
        metrics = {"zone_changes": NA_ZONE_CHANGES}
        compiled_metrics = {**metrics, "atom_moves": ("transports",)}
    else:
        metrics = compiled_metrics = {"swaps": ("virtual_swap",)}

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
            "config": config,
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


def summarize_distribution(platform: str, config: str, circuits: list[list[dict]], file_prefix: str) -> None:
    comparable: tuple[str, ...]
    proxy: tuple[str, ...]
    movement: tuple[str, ...]
    if platform == "TI":
        comparable = proxy = TI_IZ_CHANGES
        movement = TI_ION_MOVES
        event_key = "iz_changes"
    elif platform == "NA":
        comparable = proxy = NA_ZONE_CHANGES
        movement = ("transports",)
        event_key = "zone_changes"
    else:
        comparable = proxy = ("virtual_swap",)
        movement = ("swaps",)
        event_key = "swaps"

    scores, shift_rows, qubit_rows = [], [], []
    nan = float("nan")
    for compiler, _, compiler_circuits in compiler_groups(circuits, lambda rows: rows[0]["compiler"]):
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
                shift_rows.extend({"compiler": compiler, "circuit": rows[0]["circuit"], "shift": s} for s in circuit_shifts)

            # Preserve the noise-comparison window: slots before each qubit's last anchor.
            abstract_per_qubit = [[abstract_events[(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
            moves_per_qubit = [[moves[(q, k)] for k in range(anchors[q])] for q in range(num_qubits)]
            abstract_total, moves_total = sum(map(sum, abstract_per_qubit)), sum(map(sum, moves_per_qubit))
            circuit_qubit_rows = [
                {
                    "compiler": compiler,
                    "circuit": rows[0]["circuit"],
                    "num_qubits": num_qubits,
                    "qubit": q,
                    "timing_shift": float(np.average(range(len(a)), weights=a) - np.average(range(len(m)), weights=m))
                    if sum(a) and sum(m)
                    else nan,
                    "moves_share": sum(m) / moves_total if abstract_total and moves_total else nan,
                    "abstract_share": sum(a) / abstract_total if abstract_total and moves_total else nan,
                }
                for q, (a, m) in enumerate(zip(abstract_per_qubit, moves_per_qubit))
                if a
            ]
            qubit_rows.extend(circuit_qubit_rows)
            qubit_shifts = [r["timing_shift"] for r in circuit_qubit_rows if not np.isnan(r["timing_shift"])]

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
    if not scores:
        return

    # fieldnames given, since qubit and shift tables may be empty
    tables = [
        ("distribution", list(scores[0]), sorted(scores, key=lambda c: (c["circuit"], c["compiler"]))),
        ("qubits", ["compiler", "circuit", "num_qubits", "qubit", "timing_shift", "moves_share", "abstract_share"], qubit_rows),
    ]
    if platform != "SC":
        tables.append(("timing_shifts", ["compiler", "circuit", "shift"], shift_rows))
    for name, fieldnames, table in tables:
        with open(f"{file_prefix}__{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in table:
                writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in c.items()})
