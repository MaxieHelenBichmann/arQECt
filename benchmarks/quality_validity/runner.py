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


def run(functions: list[CircuitFunction], max_qubits: int, max_shots: int, max_weight: int, clean: bool) -> None:

    if clean:
        for path in [*(_RESULTS_DIR / "raw" / function.lower() for function in functions), _RESULTS_DIR / "plots", _RESULTS_DIR / "summary.csv"]:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            else:
                continue
            print(f"Deleted {path}...")

    collect(functions, _CIRCUIT_DIR, _CONFIG_DIR, _RESULTS_DIR, max_qubits, max_shots, max_weight)
    summarize(_RESULTS_DIR)
    visualize(_RESULTS_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the quality metrics of QEC circuits.")
    parser.add_argument("--function", nargs="+", choices=["StatePrep", "Encoding", "Syndrome"], default=["StatePrep", "Encoding", "Syndrome"],
                        help="only run analyses of these functions (default: all)")
    parser.add_argument("--max-qubits", type=int, default=sys.maxsize, help="skip circuits with more qubits (default: no limit)")
    parser.add_argument("--max-shots", type=int, default=1_000_000, help="number of shots used for sampling")
    parser.add_argument("--max-weight", type=int, default=3, help="maximum weight used for truncation of faults in the DEM")
    parser.add_argument("--clean", action="store_true",
                            help="delete all results (raw, intermediate csvs, plots) before running")
    args = parser.parse_args()

    run(functions=args.function, max_qubits=args.max_qubits, max_shots=args.max_shots, max_weight=args.max_weight, clean=args.clean)
