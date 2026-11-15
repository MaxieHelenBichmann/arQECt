"""Validate the static efficiency metrics against the runtime resource costs (not compilation costs) of actual compilations."""

import csv
import shutil
import sys
from pathlib import Path

import stim
import yaml

from src.models.hardware import PlatformTag

from benchmarks.common_reference.compilation import _RUN_COMPILER
from benchmarks.common_reference.structural_analysis import _RUN_STRUCTURAL_ANALYSIS
from benchmarks.common_reference.architecture_preparation import _PREPARE_HARDWARE
from benchmarks.static_metric_validation._visualizer import visualize
from benchmarks.static_metric_validation._collector import collect
from benchmarks.static_metric_validation._summarizer import summarize

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "configs"
_CIRCUIT_DIR = Path(__file__).resolve().parents[2] / "data" / "circuits"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "static_metric_validation"


def run(platforms: list[PlatformTag], max_qubits: int, max_compile_time: float, clean: bool) -> None:

    if clean:
        for path in [*(_RESULTS_DIR / "raw" / platform.lower() for platform in platforms), _RESULTS_DIR / "plots", _RESULTS_DIR / "summary.csv"]:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            else:
                continue
            print(f"Deleted {path}...")

    collect(platforms, _CIRCUIT_DIR, _CONFIG_DIR, _RESULTS_DIR, max_qubits, max_compile_time)
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
    parser.add_argument("--platform", nargs="+", choices=["TI", "NA", "SC"], default=["TI", "NA", "SC"],
                        help="only run configs of these platforms (default: all)")
    parser.add_argument("--max-qubits", type=int, default=sys.maxsize, help="skip circuits with more qubits (default: no limit)")
    parser.add_argument("--max-compile-time", type=float, default=600.0, help="time limit per compilation in seconds (default: 600)")
    parser.add_argument("--clean", action="store_true",
                        help="delete all results (raw results of the run platforms, summary, plots) before running")
    args = parser.parse_args()

    run(platforms=args.platform, max_qubits=args.max_qubits, max_compile_time=args.max_compile_time, clean=args.clean)
