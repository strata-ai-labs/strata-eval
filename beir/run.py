"""CLI entry point: python -m beir.run --dataset nfcorpus --mode hybrid"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from .config import DATASETS, K_VALUES, MODES, PYSERINI_BASELINES
from .retriever import StrataSearch

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def download_dataset(name: str, data_dir: Path) -> str:
    """Download a BEIR dataset and return the extracted path."""
    url = DATASETS[name]["url"]
    return util.download_and_unzip(url, str(data_dir))


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
    dataset = report["dataset"]

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}  |  Mode: {report['mode']}")
    print(f"  Corpus: {report['corpus_size']} docs  |  Queries: {report['num_queries']}")
    print(f"{'='*60}")

    metrics = report["metrics"]
    for metric_name in ("ndcg", "map", "recall", "precision"):
        values = metrics.get(metric_name, {})
        parts = [f"{k}: {v:.4f}" for k, v in sorted(values.items())]
        print(f"  {metric_name:>10}  {', '.join(parts)}")

    # Compare against Pyserini BM25 baselines
    baselines = PYSERINI_BASELINES.get(dataset)
    if baselines:
        ndcg10 = metrics.get("ndcg", {}).get("NDCG@10")
        recall100 = metrics.get("recall", {}).get("Recall@100")
        bm25_flat = baselines["bm25_flat"]
        bm25_mf = baselines["bm25_mf"]

        print(f"\n  {'--- Pyserini BM25 Baselines (Lucene) ---':^50}")
        print(f"  {'':>18} {'BM25 flat':>12} {'BM25 mf':>12} {'Strata':>12} {'vs flat':>10}")
        if ndcg10 is not None:
            delta = ndcg10 - bm25_flat["NDCG@10"]
            sign = "+" if delta >= 0 else ""
            print(f"  {'NDCG@10':>18} {bm25_flat['NDCG@10']:>12.4f} {bm25_mf['NDCG@10']:>12.4f} {ndcg10:>12.4f} {sign}{delta:>9.4f}")
        if recall100 is not None:
            delta = recall100 - bm25_flat["Recall@100"]
            sign = "+" if delta >= 0 else ""
            print(f"  {'Recall@100':>18} {bm25_flat['Recall@100']:>12.4f} {bm25_mf['Recall@100']:>12.4f} {recall100:>12.4f} {sign}{delta:>9.4f}")

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

    # 2. Create Strata search model and BEIR retriever
    model = StrataSearch(mode=args.mode)
    retriever = EvaluateRetrieval(model, k_values=args.k)

    # 3. Retrieve (indexes corpus + runs queries via BaseSearch.search())
    results = retriever.retrieve(corpus, queries)

    # 4. Evaluate
    ndcg, map_score, recall, precision = retriever.evaluate(
        qrels, results, args.k,
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
        "pyserini_baselines": PYSERINI_BASELINES.get(args.dataset),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_results(report, Path(args.output_dir))
    print_summary(report)

    model.cleanup()


if __name__ == "__main__":
    main()
