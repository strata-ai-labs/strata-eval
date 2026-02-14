#!/usr/bin/env python3
"""Run all 15 BEIR benchmarks and print a summary table.

Usage:
    python scripts/run_all.py                          # all 15 datasets, hybrid mode
    python scripts/run_all.py --mode keyword            # keyword-only
    python scripts/run_all.py --mode hybrid keyword     # both modes per dataset
    python scripts/run_all.py --mode hybrid --latex      # with LaTeX table output
    python scripts/run_all.py --clean --mode hybrid      # remove old results first
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

DATASETS = [
    "nfcorpus", "scifact", "arguana", "scidocs", "trec-covid",
    "fiqa", "quora", "webis-touche2020", "cqadupstack",
    "fever", "climate-fever", "nq", "hotpotqa", "dbpedia-entity", "msmarco",
]
ROOT = Path(__file__).resolve().parent.parent


def run_benchmark(dataset: str, mode: str) -> dict | None:
    """Run a single benchmark and return the parsed result JSON."""
    print(f"\n{'='*60}")
    print(f"  Running: {dataset} / {mode}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "-m", "strata_eval",
        "--dataset", dataset,
        "--mode", mode,
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    if result.returncode != 0:
        print(f"  FAILED: {dataset} / {mode} (exit code {result.returncode})")
        return None

    # Find the most recent result file for this dataset/mode
    results_dir = ROOT / "results"
    pattern = f"{dataset}_{mode}_*.json"
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"  WARNING: No result file found for {dataset}/{mode}")
        return None

    with open(files[-1]) as f:
        return json.load(f)


def _fmt_corpus(n: int) -> str:
    """Format corpus size as human-readable string (e.g. 3.6K, 5.42M)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_summary_table(results: list[dict]) -> None:
    """Print a markdown-style summary table with MRR@10 and QPS columns."""
    print(f"\n{'='*110}")
    print(f"  BENCHMARK SUMMARY — stratadb {get_version()}")
    print(f"{'='*110}\n")

    header = (
        f"  {'Dataset':<18} {'Mode':<10} {'NDCG@10':>8} {'MRR@10':>8} "
        f"{'R@100':>8} {'BM25 N@10':>10} {'vs BM25':>8} {'QPS':>8} {'Docs':>8}"
    )
    print(header)
    print(
        f"  {'─'*18} {'─'*10} {'─'*8} {'─'*8} "
        f"{'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8}"
    )

    ndcg_sum, mrr_sum, map_sum, recall_sum, count = 0.0, 0.0, 0.0, 0.0, 0

    for r in results:
        ds = r["dataset"]
        mode = r["mode"]
        ndcg10 = r["metrics"]["ndcg"].get("NDCG@10", 0)
        mrr10 = r["metrics"].get("mrr", {}).get("MRR@10", 0)
        recall100 = r["metrics"]["recall"].get("Recall@100", 0)
        map10 = r["metrics"]["map"].get("MAP@10", 0)
        corpus = r["corpus_size"]

        timing = r.get("timing", {})
        qps = timing.get("queries_per_second", 0)

        baselines = r.get("pyserini_baselines", {})
        bm25_ndcg = baselines.get("bm25_flat", {}).get("NDCG@10", 0) if baselines else 0
        delta = ndcg10 - bm25_ndcg if bm25_ndcg else 0
        sign = "+" if delta >= 0 else ""

        bm25_str = f"{bm25_ndcg:.4f}" if bm25_ndcg else "—"
        delta_str = f"{sign}{delta:.4f}" if bm25_ndcg else "—"
        qps_str = f"{qps:.1f}" if qps else "—"

        print(
            f"  {ds:<18} {mode:<10} {ndcg10:>8.4f} {mrr10:>8.4f} "
            f"{recall100:>8.4f} {bm25_str:>10} {delta_str:>8} {qps_str:>8} {corpus:>8}"
        )

        ndcg_sum += ndcg10
        mrr_sum += mrr10
        map_sum += map10
        recall_sum += recall100
        count += 1

    if count > 0:
        print(
            f"  {'─'*18} {'─'*10} {'─'*8} {'─'*8} "
            f"{'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8}"
        )
        print(
            f"  {'Average':<18} {'':10} {ndcg_sum/count:>8.4f} {mrr_sum/count:>8.4f} "
            f"{recall_sum/count:>8.4f} {'':>10} {'':>8} {'':>8} {'':>8}"
        )

    print(f"\n{'='*110}\n")


def print_latex_table(results: list[dict]) -> None:
    """Print a LaTeX table suitable for copy-paste into a paper."""
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{BEIR Benchmark Results (Strata Hybrid Search)}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Dataset & N & NDCG@10 & MRR@10 & MAP@10 & R@100 & QPS \\")
    print(r"\midrule")

    ndcg_sum, mrr_sum, map_sum, recall_sum, count = 0.0, 0.0, 0.0, 0.0, 0

    for r in results:
        ds = r["dataset"]
        corpus = _fmt_corpus(r["corpus_size"])
        ndcg10 = r["metrics"]["ndcg"].get("NDCG@10", 0)
        mrr10 = r["metrics"].get("mrr", {}).get("MRR@10", 0)
        map10 = r["metrics"]["map"].get("MAP@10", 0)
        recall100 = r["metrics"]["recall"].get("Recall@100", 0)
        timing = r.get("timing", {})
        qps = timing.get("queries_per_second", 0)
        qps_str = f"{qps:.1f}" if qps else "---"

        # Bold the NDCG@10 score
        ndcg_str = rf"\textbf{{{ndcg10:.3f}}}"

        # Pretty-print dataset name
        ds_display = ds.replace("-", " ").replace("_", " ").title()
        if ds == "nfcorpus":
            ds_display = "NFCorpus"
        elif ds == "cqadupstack":
            ds_display = "CQADupStack"
        elif ds == "dbpedia-entity":
            ds_display = "DBpedia"
        elif ds == "msmarco":
            ds_display = "MS MARCO"
        elif ds == "nq":
            ds_display = "Natural Questions"
        elif ds == "hotpotqa":
            ds_display = "HotpotQA"

        print(
            f"{ds_display} & {corpus} & {ndcg_str} & {mrr10:.3f} & "
            f"{map10:.3f} & {recall100:.3f} & {qps_str} \\\\"
        )

        ndcg_sum += ndcg10
        mrr_sum += mrr10
        map_sum += map10
        recall_sum += recall100
        count += 1

    if count > 0:
        print(r"\midrule")
        avg_ndcg = rf"\textbf{{{ndcg_sum/count:.3f}}}"
        print(
            rf"\textit{{Average}} & & {avg_ndcg} & {mrr_sum/count:.3f} & "
            f"{map_sum/count:.3f} & {recall_sum/count:.3f} & \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def clean_results(results_dir: Path) -> None:
    """Remove all result files from the results directory."""
    if results_dir.exists():
        count = sum(1 for _ in results_dir.glob("*.json"))
        shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cleaned {count} old result files from {results_dir}")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created results directory: {results_dir}")


def get_version() -> str:
    """Get the installed stratadb version."""
    try:
        import stratadb
        return getattr(stratadb, "__version__", "unknown")
    except ImportError:
        return "not installed"


def main():
    parser = argparse.ArgumentParser(description="Run all BEIR benchmarks")
    parser.add_argument(
        "--mode", nargs="+", default=["hybrid"],
        choices=["keyword", "hybrid", "hybrid-llm"],
        help="Search modes to evaluate (default: hybrid)",
    )
    parser.add_argument(
        "--dataset", nargs="+", default=DATASETS,
        choices=DATASETS,
        help="Datasets to evaluate (default: all 15)",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Output a LaTeX table suitable for paper inclusion",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove all old result files before running",
    )
    args = parser.parse_args()

    if args.clean:
        clean_results(ROOT / "results")

    all_results = []
    for dataset in args.dataset:
        for mode in args.mode:
            result = run_benchmark(dataset, mode)
            if result:
                all_results.append(result)

    if all_results:
        print_summary_table(all_results)
        if args.latex:
            print_latex_table(all_results)
    else:
        print("\nNo results collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
