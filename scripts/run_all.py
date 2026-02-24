#!/usr/bin/env python3
"""Run benchmarks across all suites and print summary tables.

Usage:
    python scripts/run_all.py                                    # all Phase 1 benchmarks
    python scripts/run_all.py --bench beir --mode hybrid keyword # BEIR with multiple modes
    python scripts/run_all.py --bench ycsb --workload a b c      # specific YCSB workloads
    python scripts/run_all.py --bench beir ann ycsb              # multiple suites
    python scripts/run_all.py --latex                            # LaTeX table output
    python scripts/run_all.py --clean                            # remove old results first
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BEIR_DATASETS = [
    "nfcorpus", "scifact", "arguana", "scidocs", "trec-covid",
    "fiqa", "quora", "webis-touche2020", "cqadupstack",
    "fever", "climate-fever", "nq", "hotpotqa", "dbpedia-entity", "msmarco",
]

YCSB_WORKLOADS = ["a", "b", "c", "d", "e", "f"]

ANN_DATASETS = ["sift-128-euclidean", "glove-100-angular", "glove-25-angular"]

GRAPH_ALGORITHMS = ["bfs", "wcc", "pagerank"]


def run_command(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  FAILED: {label} (exit code {result.returncode})")
        return False
    return True


def run_beir(modes: list[str], datasets: list[str]) -> list[str]:
    failures = []
    for dataset in datasets:
        for mode in modes:
            label = f"BEIR: {dataset} / {mode}"
            if not run_command(
                [sys.executable, str(ROOT / "run.py"), "beir",
                 "--dataset", dataset, "--mode", mode],
                label,
            ):
                failures.append(label)
    return failures


def run_ycsb(workloads: list[str], records: int, ops: int) -> list[str]:
    failures = []
    for wl in workloads:
        label = f"YCSB: workload {wl} ({records} records, {ops} ops)"
        if not run_command(
            [sys.executable, str(ROOT / "run.py"), "ycsb",
             "--workload", wl, "--records", str(records), "--ops", str(ops)],
            label,
        ):
            failures.append(label)
    return failures


def run_ann(datasets: list[str]) -> list[str]:
    failures = []
    for ds in datasets:
        # Download first
        dl_result = subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "download",
             "--bench", "ann", "--dataset", ds],
            cwd=str(ROOT),
        )
        if dl_result.returncode != 0:
            failures.append(f"ANN download: {ds}")
            continue

        label = f"ANN: {ds}"
        if not run_command(
            [sys.executable, str(ROOT / "run.py"), "ann", "--dataset", ds],
            label,
        ):
            failures.append(label)
    return failures


def run_graphalytics(algorithms: list[str], dataset: str, runs: int) -> list[str]:
    failures = []
    for algo in algorithms:
        label = f"Graphalytics: {algo} on {dataset}"
        if not run_command(
            [sys.executable, str(ROOT / "run.py"), "graphalytics",
             "--algorithm", algo, "--dataset", dataset, "--runs", str(runs)],
            label,
        ):
            failures.append(label)
    return failures


def generate_report(fmt: str = "markdown") -> None:
    print(f"\n{'='*60}")
    print(f"  Generating {fmt} report")
    print(f"{'='*60}\n")
    subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "report", "--format", fmt],
        cwd=str(ROOT),
    )


def clean_results() -> None:
    results_dir = ROOT / "results"
    if results_dir.exists():
        count = sum(1 for _ in results_dir.glob("*.json"))
        shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleaned {count} old result files from {results_dir}")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)


def get_version() -> str:
    try:
        import stratadb
        return getattr(stratadb, "__version__", "unknown")
    except ImportError:
        return "not installed"


def main():
    parser = argparse.ArgumentParser(description="Run all strata-eval benchmarks")
    parser.add_argument(
        "--bench", nargs="+",
        default=["beir", "ycsb", "ann", "graphalytics"],
        choices=["beir", "ycsb", "ann", "graphalytics"],
        help="Benchmark suites to run (default: all Phase 1)",
    )
    parser.add_argument(
        "--mode", nargs="+", default=["hybrid"],
        choices=["keyword", "hybrid", "hybrid-llm"],
        help="BEIR search modes (default: hybrid)",
    )
    parser.add_argument(
        "--dataset", nargs="+", default=None,
        help="BEIR datasets to evaluate (default: all 15)",
    )
    parser.add_argument(
        "--workload", nargs="+", default=None,
        help="YCSB workloads to run (default: all a-f)",
    )
    parser.add_argument(
        "--records", type=int, default=100_000,
        help="YCSB record count (default: 100000)",
    )
    parser.add_argument(
        "--ops", type=int, default=100_000,
        help="YCSB operation count (default: 100000)",
    )
    parser.add_argument(
        "--ann-dataset", nargs="+", default=None,
        help="ANN datasets (default: all 3)",
    )
    parser.add_argument(
        "--graph-algorithm", nargs="+", default=None,
        help="Graphalytics algorithms (default: bfs, wcc, pagerank)",
    )
    parser.add_argument(
        "--graph-dataset", default="example-directed",
        help="Graphalytics dataset (default: example-directed)",
    )
    parser.add_argument(
        "--graph-runs", type=int, default=10,
        help="Graphalytics runs per algorithm (default: 10)",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Generate LaTeX tables",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove old result files before running",
    )
    args = parser.parse_args()

    print(f"strata-eval batch runner — stratadb {get_version()}")

    if args.clean:
        clean_results()

    all_failures: list[str] = []

    if "beir" in args.bench:
        all_failures.extend(run_beir(args.mode, args.dataset or BEIR_DATASETS))

    if "ycsb" in args.bench:
        all_failures.extend(run_ycsb(args.workload or YCSB_WORKLOADS, args.records, args.ops))

    if "ann" in args.bench:
        all_failures.extend(run_ann(args.ann_dataset or ANN_DATASETS))

    if "graphalytics" in args.bench:
        all_failures.extend(run_graphalytics(
            args.graph_algorithm or GRAPH_ALGORITHMS,
            args.graph_dataset,
            args.graph_runs,
        ))

    # Generate reports
    generate_report("markdown")
    if args.latex:
        generate_report("latex")

    # Summary
    print(f"\n{'='*60}")
    if all_failures:
        print(f"  DONE — {len(all_failures)} failure(s):")
        for f in all_failures:
            print(f"    - {f}")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"  DONE — all benchmarks passed")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
