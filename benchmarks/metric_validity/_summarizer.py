"""Summarize the concordance, proportionality and decision regret of the static metrics on circuit pairs into intermediate csvs of the plots.
Only the runtime cost aspect is validated (hardware events of the compiled circuit), not the compilation cost (compiler work / freedom).

Concordance C: right order of the costs.
Proportionality P: larger metric gap = larger cost difference.

Paired by comparable task (same task on the same quantum error-correction code), or matched workload (equal qubits, similar number of 2q gates).
"""

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


# targets: raw (aggregated, unnormalized) values associated with runtime cost of a compiled circuit
TARGETS = {
    "routing": {"TI": ["shuttles"], "NA": ["transports"], "SC": ["swaps"]}, # qubit movements
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
        "statically_servable_interaction_fraction": (-1, "routing"),
        "capacity_constrained_qubit_partition": (1, "routing"),
        "non_bipartite_interaction_fraction": (1, "routing"),
    },
    "Qubit timelines": {
        "average_interaction_reuse_distance": (1, "routing"),
        "capacity_constrained_reuse": (-1, "routing"),
        "partner_switches_per_qubit": (1, "routing"),
        "capacity_constrained_partner_switches_per_qubit": (1, "routing"),
        "activity_imbalance": (1, "schedule"),
    },
    "Workload baselines": {
        "N_2q": (1, "routing"),
    },
}


def summarize(results_dir: Path) -> None:
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
    write_csv(results_dir / "summary.csv", summary)


def evaluate(circuits: dict, metric: str, direction: int, target: str, design: str, multiplicity: dict) -> dict:
    """Concordance C, proportionality P (both with family-bootstrap CI), decisiveness and relative regret of one metric on one pair set."""
    names = [n for n, c in circuits.items() if c[metric] is not None and c.get(target) is not None]
    if metric in ("average_interaction_reuse_distance", "capacity_constrained_reuse"):  # 0 means "no partner reuse", not the best value
        names = [n for n in names if circuits[n]["average_interaction_reuse_distance"] > 0]
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
