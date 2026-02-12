"""CLI entry point: python -m beir.run --dataset nfcorpus --mode hybrid"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from .config import DATASETS, K_VALUES, MODES
from .retriever import StrataRetriever

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def download_dataset(name: str, data_dir: Path) -> str:
    """Download a BEIR dataset and return the extracted path."""
    url = DATASETS[name]["url"]
    out_dir = str(data_dir)
    data_path = util.download_and_unzip(url, out_dir)
    return data_path


def save_results(report: dict, output_dir: Path) -> Path:
    """Write the evaluation report to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{report['dataset']}_{report['mode']}_{report['timestamp']}.json"
    # Sanitize colons from ISO timestamp for filenames
    filename = filename.replace(":", "-")
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {path}")
    return path


def print_summary(report: dict) -> None:
    """Print a human-readable summary of evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {report['dataset']}  |  Mode: {report['mode']}")
    print(f"  Corpus: {report['corpus_size']} docs  |  Queries: {report['num_queries']}")
    print(f"{'='*60}")

    metrics = report["metrics"]
    for metric_name in ("ndcg", "map", "recall", "precision"):
        values = metrics.get(metric_name, {})
        parts = [f"{k}: {v:.4f}" for k, v in sorted(values.items())]
        print(f"  {metric_name:>10}  {', '.join(parts)}")

    print(f"{'='*60}\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Strata retrieval quality on BEIR benchmarks",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASETS.keys()),
        help="BEIR dataset to evaluate on",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=MODES,
        help="Search mode (default: hybrid)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=K_VALUES,
        help=f"Cutoff depths for evaluation (default: {K_VALUES})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "results"),
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(ROOT / "datasets"),
        help="Directory for downloaded BEIR datasets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Download + load dataset
    print(f"Loading BEIR dataset: {args.dataset}")
    data_path = download_dataset(args.dataset, Path(args.data_dir))
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # 2. Index corpus into Strata (fresh temp dir per run)
    with tempfile.TemporaryDirectory() as tmpdir:
        retriever = StrataRetriever(tmpdir, mode=args.mode)
        retriever.index(corpus)

        # 3. Run retrieval
        max_k = max(args.k)
        results = retriever.retrieve(queries, k=max_k)

    # 4. Evaluate using pytrec_eval via BEIR
    ndcg, map_score, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, k_values=args.k,
    )

    # 5. Build + save report
    report = {
        "dataset": args.dataset,
        "mode": args.mode,
        "corpus_size": len(corpus),
        "num_queries": len(queries),
        "k_values": args.k,
        "metrics": {
            "ndcg": ndcg,
            "map": map_score,
            "recall": recall,
            "precision": precision,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_results(report, Path(args.output_dir))
    print_summary(report)


if __name__ == "__main__":
    main()
