"""Summarize the agreement between the Monte-Carlo-Sampling-based (MC) and the Detector-Error-Model-based (DEM) quality metrics per circuit function.

MC: point estimate with a statistical 95% Wilson score confidence interval
DEM: point estimate (renormalized over the enumerated fault sets) with hard bounds (unknown outcome of the truncated fault sets)
"""

import csv
from pathlib import Path

import numpy as np

# compared metrics per circuit function
METRICS = {
    "StatePrep": ["logical_error_rate", "acceptance_rate"],
    "Encoding": ["logical_error_rate"],
    "Syndrome": ["logical_error_rate", "syndrome_error_rate"],
}


def summarize(results_dir: Path) -> None:
    """Per circuit function and metric: share of agreeing runs, deviation of the point estimates and truncation of the DEM, read from the raw results."""
    runs = load(results_dir)

    summary = []
    for function, metrics in METRICS.items():
        for metric in metrics:
            rows = [r for r in runs if r["function"] == function and r[f"mc_{metric}"] != "" and r[f"dem_{metric}"] != ""] # LER undefined if no shot accepted
            if not rows:
                continue
            column = lambda key: np.array([float(r[key]) for r in rows])
            mc, mc_lo, mc_hi = column(f"mc_{metric}"), column(f"mc_{metric}_ci_lower"), column(f"mc_{metric}_ci_upper")
            dem, dem_lo, dem_hi = column(f"dem_{metric}"), column(f"dem_{metric}_lower"), column(f"dem_{metric}_upper")
            tail = column("dem_truncation_tail")

            summary.append({
                "function": function,
                "metric": metric,
                "n_runs": len(rows),
                "n_mc_zero": int((mc == 0).sum()), # no failure sampled --> MC only gives an upper bound
                "dem_in_mc_ci": np.mean((dem >= mc_lo - 1e-12) & (dem <= mc_hi + 1e-12)), # tolerance: float rounding of the Wilson bounds at rates 0 and 1
                "intervals_overlap": np.mean((dem_lo <= mc_hi + 1e-12) & (dem_hi >= mc_lo - 1e-12)),
                "median_rel_deviation": np.median(np.abs(mc - dem)[dem > 0] / dem[dem > 0]), # |MC - DEM| / DEM
                "median_truncation_tail": np.median(tail),
                "max_truncation_tail": np.max(tail),
            })
            print(f"[{function}] {metric}: DEM in MC CI for {summary[-1]['dem_in_mc_ci']:.0%} of {len(rows)} runs")

    with (results_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def load(results_dir: Path) -> list[dict]:
    """All raw results (one row per circuit and hardware configuration), values as strings ("" if not available)."""
    runs = []
    for file in sorted((results_dir / "raw").glob("*/*.csv")):
        with file.open(newline="") as f:
            runs += list(csv.DictReader(f))
    return runs
