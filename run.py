#!/usr/bin/env python3
"""Unified CLI with subcommands for all benchmark suites.

Usage:
    python run.py beir --dataset nfcorpus --mode hybrid
    python run.py ycsb --workload a --records 100000
    python run.py ann --dataset sift-128-euclidean
    python run.py graphalytics --algorithm bfs --dataset example-directed
    python run.py download --bench ann --dataset sift-128-euclidean
    python run.py report --format latex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks import get_benchmarks
from benchmarks.base import BaseBenchmark
from lib import download as dl_mod
from lib import report as report_mod
from lib.recorder import ResultRecorder

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> None:
    # Backward compat: if first arg looks like old-style BEIR invocation
    # (starts with --dataset), inject "beir" subcommand.
    args = argv if argv is not None else sys.argv[1:]
    if args and args[0] == "--dataset":
        args = ["beir"] + args

    parser = argparse.ArgumentParser(
        prog="strata-eval",
        description="Comprehensive benchmarks for StrataDB",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(ROOT / "results"),
        help="Directory for result JSON files (default: results/)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Register each benchmark as a subcommand
    benchmarks = get_benchmarks()
    bench_instances: dict[str, BaseBenchmark] = {}
    for name, cls in sorted(benchmarks.items()):
        sub = subparsers.add_parser(name, help=f"Run {name} benchmarks")
        instance = cls()
        instance.register_args(sub)
        bench_instances[name] = instance

    # Download subcommand
    dl_parser = subparsers.add_parser("download", help="Download benchmark datasets")
    dl_mod.register_args(dl_parser)

    # Report subcommand
    report_parser = subparsers.add_parser("report", help="Generate benchmark reports")
    report_mod.register_args(report_parser)

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return

    if parsed.command == "download":
        dl_mod.run_download(parsed)
        return

    if parsed.command == "report":
        # Wire --output-dir through to report if --results-dir not explicitly set
        if not hasattr(parsed, "results_dir") or parsed.results_dir == "results":
            parsed.results_dir = parsed.output_dir
        report_mod.run_report(parsed)
        return

    # Run a benchmark
    bench = bench_instances.get(parsed.command)
    if bench is None:
        parser.print_help()
        return

    if not bench.validate(parsed):
        print(f"Validation failed for {parsed.command}. Check prerequisites.")
        sys.exit(1)

    try:
        results = bench.run(parsed)
    except NotImplementedError as e:
        print(f"\n{parsed.command}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR running {parsed.command}: {e}", file=sys.stderr)
        sys.exit(1)

    if results:
        recorder = ResultRecorder(category=parsed.command)
        for r in results:
            recorder.record(r)
        recorder.save(parsed.output_dir)


if __name__ == "__main__":
    main()
