"""Validate the quality metrics of QEC circuits extracted via Monte-Carlo-Sampling against a Detector-Error-Model-based analysis."""

import csv
import shutil
import sys
from pathlib import Path

import stim
import yaml

from src.circuits.circuit import CircuitFunction

from benchmarks.quality_validity._visualizer import visualize
from benchmarks.quality_validity._collector import collect
from benchmarks.quality_validity._summarizer import summarize

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "configs"
_CIRCUIT_DIR = Path(__file__).resolve().parents[2] / "data" / "circuits"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "quality_validity"


def run(functions: list[CircuitFunction], max_qubits: int) -> None:

    for path in [*(_RESULTS_DIR / "raw" / function.lower() for function in functions), _RESULTS_DIR / "plots", _RESULTS_DIR / "summary.csv"]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            continue
        print(f"Deleted {path}...")

    collect(functions, _CIRCUIT_DIR, _CONFIG_DIR, _RESULTS_DIR, max_qubits)
    summarize(_RESULTS_DIR)
    visualize(_RESULTS_DIR)


def save_raw_results(file: Path, metadata: dict, metrics: dict, compiled: dict) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*metadata, "source", "metric", "value"])
        for source, values in {"structural": metrics, "compiled": compiled}.items():
            for metric, value in values.items():
                writer.writerow([*metadata.values(), source, metric, "" if value is None else value])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the static efficiency metrics against compiled resource costs.")
    parser.add_argument("--function", nargs="+", choices=["SP", "EC", "SE"], default=["SP", "EC", "SE"],
                        help="only run analyses of these functions (default: all)")
    parser.add_argument("--max-qubits", type=int, default=sys.maxsize, help="skip circuits with more qubits (default: no limit)")
    args = parser.parse_args()

    run(functions=args.functions, max_qubits=args.max_qubits)
