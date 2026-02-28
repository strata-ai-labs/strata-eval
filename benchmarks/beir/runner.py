"""BEIR benchmark runner — evaluates Strata retrieval quality on BEIR datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytrec_eval
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from lib.schema import BenchmarkResult
from benchmarks.base import BaseBenchmark
from .config import DATASETS, K_VALUES, MODES, PYSERINI_BASELINES
from .retriever import StrataSearch

ROOT = Path(__file__).resolve().parent.parent.parent

# CQADupStack has 12 subforums, each a self-contained BEIR dataset.
# Standard evaluation runs each independently and macro-averages metrics.
CQADUPSTACK_SUBFORUMS = [
    "android", "english", "gaming", "gis", "mathematica", "physics",
    "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
]


class BeirBenchmark(BaseBenchmark):
    name = "beir"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset", nargs="+", required=True,
            choices=list(DATASETS.keys()),
            help="BEIR dataset(s) to evaluate on",
        )
        parser.add_argument(
            "--mode", nargs="+", default=["keyword", "hybrid"],
            choices=MODES,
            help="Search mode(s) (default: keyword hybrid)",
        )
        parser.add_argument(
            "--k", type=int, nargs="+", default=K_VALUES,
            help=f"Cutoff depths for evaluation (default: {K_VALUES})",
        )
        parser.add_argument(
            "--data-dir", type=str, default=str(ROOT / "datasets"),
            help="Directory for downloaded BEIR datasets",
        )
        parser.add_argument(
            "--db-dir", type=str, default=None,
            help="Persistent database directory (skips re-indexing)",
        )
        parser.add_argument(
            "--model", type=str, default="miniLM",
            help="Embedding model for hybrid/hybrid-llm modes (default: miniLM)",
        )

    def validate(self, args: argparse.Namespace) -> bool:
        try:
            from lib.strata_client import StrataClient
            StrataClient._resolve_binary(None)
        except FileNotFoundError:
            print("strata CLI binary not found. Add it to PATH or set STRATA_BIN.")
            return False
        return True

    def download(self, args: argparse.Namespace) -> None:
        raw = getattr(args, "dataset", None) or []
        datasets = raw if isinstance(raw, list) else [raw]
        data_dir = Path(getattr(args, "data_dir", str(ROOT / "datasets")))
        for name in datasets:
            if name not in DATASETS:
                print(f"Unknown BEIR dataset: {name}")
                continue
            print(f"Downloading BEIR dataset: {name}")
            _download_dataset(name, data_dir)
            print(f"  Done: {name}")

    def run(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        all_results: list[BenchmarkResult] = []
        datasets = args.dataset if isinstance(args.dataset, list) else [args.dataset]
        modes = args.mode if isinstance(args.mode, list) else [args.mode]

        for dataset in datasets:
            for mode in modes:
                # Build a per-run args copy
                run_args = argparse.Namespace(**vars(args))
                run_args.dataset = dataset
                run_args.mode = mode

                if dataset == "cqadupstack":
                    all_results.extend(self._run_cqadupstack(run_args))
                else:
                    all_results.extend(self._run_single(run_args))

        return all_results

    # ------------------------------------------------------------------
    # Single dataset
    # ------------------------------------------------------------------

    def _run_single(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        print(f"Loading BEIR dataset: {args.dataset}")
        data_path = _download_dataset(args.dataset, Path(args.data_dir))
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        embed_model = getattr(args, "model", "miniLM")
        model = StrataSearch(mode=args.mode, db_path=args.db_dir, embed_model=embed_model)
        retriever = EvaluateRetrieval(model, k_values=args.k)
        results = retriever.retrieve(corpus, queries)

        ndcg, map_score, recall, precision = retriever.evaluate(qrels, results, args.k)
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, args.k, metric="mrr")

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
        per_query_raw = evaluator.evaluate(results)
        per_query_ndcg10 = {
            qid: scores["ndcg_cut_10"]
            for qid, scores in per_query_raw.items()
        }

        num_queries = len(queries)
        qps = num_queries / model.search_time if model.search_time > 0 else 0
        avg_latency_ms = (model.search_time / num_queries * 1000) if num_queries > 0 else 0

        metrics = {
            "ndcg_at_10": ndcg.get("NDCG@10", 0),
            "ndcg_at_100": ndcg.get("NDCG@100", 0),
            "recall_at_10": recall.get("Recall@10", 0),
            "recall_at_100": recall.get("Recall@100", 0),
            "map_at_10": map_score.get("MAP@10", 0),
            "mrr_at_10": mrr.get("MRR@10", 0),
            "precision_at_10": precision.get("P@10", 0),
            "qps": round(qps, 1),
            "index_time_s": round(model.index_time, 2),
            "search_time_s": round(model.search_time, 2),
            "avg_latency_ms": round(avg_latency_ms, 1),
        }

        baselines = None
        pyserini = PYSERINI_BASELINES.get(args.dataset)
        if pyserini:
            baselines = {
                "pyserini_bm25_flat": {
                    "ndcg_at_10": pyserini["bm25_flat"]["NDCG@10"],
                    "recall_at_100": pyserini["bm25_flat"]["Recall@100"],
                },
                "pyserini_bm25_mf": {
                    "ndcg_at_10": pyserini["bm25_mf"]["NDCG@10"],
                    "recall_at_100": pyserini["bm25_mf"]["Recall@100"],
                },
            }

        mode_label = args.mode if args.mode == "keyword" else f"{args.mode}/{embed_model}"
        result = BenchmarkResult(
            benchmark=f"beir/{args.dataset}/{mode_label}",
            category="beir",
            parameters={
                "dataset": args.dataset,
                "mode": args.mode,
                "embed_model": embed_model if args.mode != "keyword" else None,
                "corpus_size": len(corpus),
                "num_queries": num_queries,
                "k_values": args.k,
            },
            metrics=metrics,
            baselines=baselines,
        )

        _print_summary(args.dataset, args.mode, len(corpus), num_queries,
                       ndcg, map_score, recall, precision, mrr,
                       model.index_time, model.search_time, qps)

        model.cleanup()
        return [result]

    # ------------------------------------------------------------------
    # CQADupStack (12 subforums, macro-averaged)
    # ------------------------------------------------------------------

    def _run_cqadupstack(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        data_path = _download_dataset("cqadupstack", Path(args.data_dir))
        cqa_root = Path(data_path)

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

            embed_model = getattr(args, "model", "miniLM")
            model = StrataSearch(mode=args.mode, embed_model=embed_model)
            retriever = EvaluateRetrieval(model, k_values=args.k)
            results = retriever.retrieve(corpus, queries)

            ndcg, map_score, recall, precision = retriever.evaluate(
                qrels, results, args.k,
            )
            mrr = EvaluateRetrieval.evaluate_custom(qrels, results, args.k, metric="mrr")

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
            k_keys = list(subforum_metrics[CQADUPSTACK_SUBFORUMS[0]][metric_name].keys())
            averaged[metric_name] = {}
            for k_key in k_keys:
                total = sum(
                    subforum_metrics[sf][metric_name][k_key]
                    for sf in CQADUPSTACK_SUBFORUMS
                )
                averaged[metric_name][k_key] = round(total / n, 5)

        qps = total_queries / total_search_time if total_search_time > 0 else 0
        avg_latency_ms = (total_search_time / total_queries * 1000) if total_queries > 0 else 0

        metrics = {
            "ndcg_at_10": averaged["ndcg"].get("NDCG@10", 0),
            "ndcg_at_100": averaged["ndcg"].get("NDCG@100", 0),
            "recall_at_10": averaged["recall"].get("Recall@10", 0),
            "recall_at_100": averaged["recall"].get("Recall@100", 0),
            "map_at_10": averaged["map"].get("MAP@10", 0),
            "mrr_at_10": averaged["mrr"].get("MRR@10", 0),
            "qps": round(qps, 1),
            "index_time_s": round(total_index_time, 2),
            "search_time_s": round(total_search_time, 2),
            "avg_latency_ms": round(avg_latency_ms, 1),
        }

        baselines = None
        pyserini = PYSERINI_BASELINES.get("cqadupstack")
        if pyserini:
            baselines = {
                "pyserini_bm25_flat": {
                    "ndcg_at_10": pyserini["bm25_flat"]["NDCG@10"],
                    "recall_at_100": pyserini["bm25_flat"]["Recall@100"],
                },
                "pyserini_bm25_mf": {
                    "ndcg_at_10": pyserini["bm25_mf"]["NDCG@10"],
                    "recall_at_100": pyserini["bm25_mf"]["Recall@100"],
                },
            }

        result = BenchmarkResult(
            benchmark=f"beir/cqadupstack/{args.mode}",
            category="beir",
            parameters={
                "dataset": "cqadupstack",
                "mode": args.mode,
                "corpus_size": total_corpus,
                "num_queries": total_queries,
                "k_values": args.k,
                "subforums": len(CQADUPSTACK_SUBFORUMS),
            },
            metrics=metrics,
            baselines=baselines,
        )

        _print_summary("cqadupstack", args.mode, total_corpus, total_queries,
                       averaged["ndcg"], averaged["map"],
                       averaged["recall"], averaged["precision"], averaged["mrr"],
                       total_index_time, total_search_time, qps)

        return [result]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _download_dataset(name: str, data_dir: Path) -> str:
    url = DATASETS[name]["url"]
    return util.download_and_unzip(url, str(data_dir))


def _print_summary(
    dataset: str, mode: str, corpus_size: int, num_queries: int,
    ndcg: dict, map_score: dict, recall: dict, precision: dict, mrr: dict,
    index_time: float, search_time: float, qps: float,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}  |  Mode: {mode}")
    print(f"  Corpus: {corpus_size} docs  |  Queries: {num_queries}")
    print(f"{'='*60}")

    for name, values in [("ndcg", ndcg), ("map", map_score), ("recall", recall),
                         ("precision", precision), ("mrr", mrr)]:
        parts = [f"{k}: {v:.4f}" for k, v in sorted(values.items())]
        print(f"  {name:>10}  {', '.join(parts)}")

    print(f"\n  {'--- Timing ---':^50}")
    print(f"  Index: {index_time:.1f}s  |  Search: {search_time:.1f}s  |  QPS: {qps:.1f}")

    baselines = PYSERINI_BASELINES.get(dataset)
    if baselines:
        ndcg10 = ndcg.get("NDCG@10")
        bm25_flat = baselines["bm25_flat"]
        bm25_mf = baselines["bm25_mf"]

        print(f"\n  {'--- Pyserini BM25 Baselines (Lucene) ---':^50}")
        print(f"  {'':>18} {'BM25 flat':>12} {'BM25 mf':>12} {'Strata':>12} {'vs flat':>10}")
        if ndcg10 is not None:
            delta = ndcg10 - bm25_flat["NDCG@10"]
            sign = "+" if delta >= 0 else ""
            print(f"  {'NDCG@10':>18} {bm25_flat['NDCG@10']:>12.4f} {bm25_mf['NDCG@10']:>12.4f} {ndcg10:>12.4f} {sign}{delta:>9.4f}")
        recall100 = recall.get("Recall@100")
        if recall100 is not None:
            delta = recall100 - bm25_flat["Recall@100"]
            sign = "+" if delta >= 0 else ""
            print(f"  {'Recall@100':>18} {bm25_flat['Recall@100']:>12.4f} {bm25_mf['Recall@100']:>12.4f} {recall100:>12.4f} {sign}{delta:>9.4f}")

    print(f"{'='*60}\n")
