"""CLI entry point: python -m strata_eval --dataset nfcorpus --mode hybrid"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pytrec_eval
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from .config import DATASETS, K_VALUES, MODES, PYSERINI_BASELINES
from .retriever import StrataSearch
from .redis_search import RedisSearch

ROOT = Path(__file__).resolve().parent.parent

# CQADupStack has 12 subforums, each a self-contained BEIR dataset.
# Standard evaluation runs each independently and macro-averages metrics.
CQADUPSTACK_SUBFORUMS = [
    "android", "english", "gaming", "gis", "mathematica", "physics",
    "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
]


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
    for metric_name in ("ndcg", "map", "recall", "precision", "mrr"):
        values = metrics.get(metric_name, {})
        parts = [f"{k}: {v:.4f}" for k, v in sorted(values.items())]
        print(f"  {metric_name:>10}  {', '.join(parts)}")

    timing = report.get("timing")
    if timing:
        print(f"\n  {'--- Timing ---':^50}")
        print(f"  Index: {timing['index_time_s']:.1f}s  |  Search: {timing['search_time_s']:.1f}s  |  QPS: {timing['queries_per_second']:.1f}")

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
# CQADupStack (multi-subforum dataset)
# ------------------------------------------------------------------

def run_cqadupstack(args) -> None:
    """Run evaluation on all 12 CQADupStack subforums and macro-average."""
    data_path = download_dataset("cqadupstack", Path(args.data_dir))
    cqa_root = Path(data_path)

    all_per_query_ndcg10: dict[str, float] = {}
    subforum_metrics: dict[str, dict] = {}
    total_corpus = 0
    total_queries = 0
    total_index_time = 0.0
    total_search_time = 0.0

    for subforum in CQADUPSTACK_SUBFORUMS:
        subforum_path = cqa_root / subforum
        print(f"\n{'─'*60}")
        print(f"  CQADupStack subforum: {subforum}")
        print(f"{'─'*60}")

        corpus, queries, qrels = GenericDataLoader(
            data_folder=str(subforum_path),
        ).load(split="test")

        if args.retriever == "redis":
            model = RedisSearch(mode=args.mode, redis_url=args.redis_url)
        else:
            model = StrataSearch(mode=args.mode)
        retriever = EvaluateRetrieval(model, k_values=args.k)

        results = retriever.retrieve(corpus, queries)

        ndcg, map_score, recall, precision = retriever.evaluate(
            qrels, results, args.k,
        )
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, args.k, metric="mrr")

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
        per_query_raw = evaluator.evaluate(results)
        for qid, scores in per_query_raw.items():
            all_per_query_ndcg10[f"{subforum}/{qid}"] = scores["ndcg_cut_10"]

        subforum_metrics[subforum] = {
            "corpus_size": len(corpus),
            "num_queries": len(queries),
            "ndcg": ndcg,
            "map": map_score,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        }

        total_corpus += len(corpus)
        total_queries += len(queries)
        total_index_time += model.index_time
        total_search_time += model.search_time

        model.cleanup()

    # Macro-average across subforums
    metric_names = ("ndcg", "map", "recall", "precision", "mrr")
    averaged: dict[str, dict[str, float]] = {}
    n = len(CQADUPSTACK_SUBFORUMS)
    for metric_name in metric_names:
        # Collect all k-level keys from first subforum
        k_keys = list(subforum_metrics[CQADUPSTACK_SUBFORUMS[0]][metric_name].keys())
        averaged[metric_name] = {}
        for k_key in k_keys:
            total = sum(
                subforum_metrics[sf][metric_name][k_key]
                for sf in CQADUPSTACK_SUBFORUMS
            )
            averaged[metric_name][k_key] = round(total / n, 5)

    total_time = total_index_time + total_search_time
    qps = total_queries / total_search_time if total_search_time > 0 else 0
    avg_latency_ms = (total_search_time / total_queries * 1000) if total_queries > 0 else 0

    try:
        import stratadb
        strata_version = getattr(stratadb, "__version__", "unknown")
    except ImportError:
        strata_version = "not installed"

    system_info = {
        "stratadb_version": strata_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }

    report = {
        "dataset": "cqadupstack",
        "mode": args.mode,
        "corpus_size": total_corpus,
        "num_queries": total_queries,
        "k_values": args.k,
        "metrics": averaged,
        "per_subforum": {
            sf: {
                "corpus_size": m["corpus_size"],
                "num_queries": m["num_queries"],
                "metrics": {mn: m[mn] for mn in metric_names},
            }
            for sf, m in subforum_metrics.items()
        },
        "per_query_ndcg10": all_per_query_ndcg10,
        "timing": {
            "index_time_s": round(total_index_time, 2),
            "search_time_s": round(total_search_time, 2),
            "total_time_s": round(total_time, 2),
            "queries_per_second": round(qps, 1),
            "avg_latency_ms": round(avg_latency_ms, 1),
        },
        "system": system_info,
        "pyserini_baselines": PYSERINI_BASELINES.get("cqadupstack"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_results(report, Path(args.output_dir))
    print_summary(report)


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
    parser.add_argument(
        "--db-dir",
        type=str,
        default=None,
        help="Persistent database directory (skips re-indexing on subsequent runs)",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="strata",
        choices=["strata", "redis"],
        help="Search engine to benchmark (default: strata)",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://localhost:6380",
        help="Redis URL for redis retriever (default: redis://localhost:6380)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "cqadupstack":
        run_cqadupstack(args)
        return

    # 1. Download + load dataset
    print(f"Loading BEIR dataset: {args.dataset}")
    data_path = download_dataset(args.dataset, Path(args.data_dir))
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # 2. Create search model and BEIR retriever
    if args.retriever == "redis":
        model = RedisSearch(mode=args.mode, redis_url=args.redis_url)
    else:
        model = StrataSearch(mode=args.mode, db_path=args.db_dir)
    retriever = EvaluateRetrieval(model, k_values=args.k)

    # 3. Retrieve (indexes corpus + runs queries via BaseSearch.search())
    results = retriever.retrieve(corpus, queries)

    # 4. Evaluate standard metrics
    ndcg, map_score, recall, precision = retriever.evaluate(
        qrels, results, args.k,
    )

    # 4b. MRR
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, args.k, metric="mrr")

    # 4c. Per-query NDCG@10 for statistical significance tests
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    per_query_raw = evaluator.evaluate(results)
    per_query_ndcg10 = {
        qid: scores["ndcg_cut_10"]
        for qid, scores in per_query_raw.items()
    }

    # 5. Timing (from instrumented retriever)
    index_time = model.index_time
    search_time = model.search_time
    total_time = index_time + search_time
    num_queries = len(queries)
    qps = num_queries / search_time if search_time > 0 else 0
    avg_latency_ms = (search_time / num_queries * 1000) if num_queries > 0 else 0

    # 6. System metadata
    try:
        import stratadb
        strata_version = getattr(stratadb, "__version__", "unknown")
    except ImportError:
        strata_version = "not installed"

    system_info = {
        "stratadb_version": strata_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }

    # 7. Build + save report
    report = {
        "dataset": args.dataset,
        "mode": args.mode,
        "corpus_size": len(corpus),
        "num_queries": num_queries,
        "k_values": args.k,
        "metrics": {
            "ndcg": ndcg,
            "map": map_score,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        },
        "per_query_ndcg10": per_query_ndcg10,
        "timing": {
            "index_time_s": round(index_time, 2),
            "search_time_s": round(search_time, 2),
            "total_time_s": round(total_time, 2),
            "queries_per_second": round(qps, 1),
            "avg_latency_ms": round(avg_latency_ms, 1),
        },
        "system": system_info,
        "pyserini_baselines": PYSERINI_BASELINES.get(args.dataset),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_results(report, Path(args.output_dir))
    print_summary(report)

    model.cleanup()


if __name__ == "__main__":
    main()
