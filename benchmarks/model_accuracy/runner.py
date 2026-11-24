
"""Run the model accuracy experiments by comparing hardware events of different circuits on the abstract executions and actual compilations."""

import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import stim
import yaml
from collections.abc import Callable

from src.models.hardware import PlatformTag

from benchmarks.model_accuracy._summarizer import summarize
from benchmarks.model_accuracy._visualizer import visualize
from benchmarks.model_accuracy._collector import collect

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "data" / "configs"
_CIRCUIT_DIR = Path(__file__).resolve().parents[2] / "data" / "circuits"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "model_accuracy"


def run(platforms: list[PlatformTag], max_qubits: int, max_compile_time: float, clean: bool) -> None:

    if clean:
        for platform in platforms:
            for kind in ("raw", "plots"):
                platform_dir = _RESULTS_DIR / kind / platform.lower()
                if platform_dir.exists():
                    shutil.rmtree(platform_dir)
                    print(f"Emptied {platform_dir}...")

    collect(platforms, _CIRCUIT_DIR, _CONFIG_DIR, _RESULTS_DIR, max_qubits, max_compile_time)
    summarize(platforms, _RESULTS_DIR, max_qubits)
    visualize(platforms, _RESULTS_DIR)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare abstract executions and compilations of circuits on hardware configs.")
    parser.add_argument("--platform", nargs="+", choices=["TI", "NA", "SC"], default=["TI", "NA", "SC"],
                        help="only run configs of these platforms (default: all)")
    parser.add_argument("--max-qubits", type=int, default=sys.maxsize, help="skip circuits with more qubits (default: no limit)")
    parser.add_argument("--max-compile-time", type=float, default=600.0, help="time limit per compilation in seconds (default: 600)")
    parser.add_argument("--clean", action="store_true",
                        help="delete all results (raw, intermediate csvs, plots) of the run platforms before running")
    args = parser.parse_args()

    run(platforms=args.platform, max_qubits=args.max_qubits, max_compile_time=args.max_compile_time, clean=args.clean)

