"""ANN benchmark runner — evaluates Strata vector search recall and throughput.

Index build uses a persistent StrataClient (for temp-file batch upserts).
Query measurement pipes pre-generated search commands directly to the
``strata`` CLI via :func:`batch_execute` — no Python overhead per query.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import tempfile
import time
from pathlib import Path

from lib.schema import BenchmarkResult
from lib.strata_client import StrataClient, batch_execute
from benchmarks.base import BaseBenchmark
from .config import ANN_DATASETS, DEFAULT_K, DEFAULT_BATCH_SIZE

ROOT = Path(__file__).resolve().parent.parent.parent

# Number of warmup queries piped to the CLI before timed measurement.
_WARMUP_QUERIES = 100


class AnnBenchmark(BaseBenchmark):
    name = "ann"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset", nargs="+",
            default=list(ANN_DATASETS.keys()),
            choices=list(ANN_DATASETS.keys()),
            help="Dataset(s) to evaluate (default: all)",
        )
        parser.add_argument(
            "--k", type=int, nargs="+", default=DEFAULT_K,
            help=f"Recall depths to evaluate (default: {DEFAULT_K})",
        )
        parser.add_argument(
            "--data-dir", type=str,
            default=str(ROOT / "datasets" / "ann"),
            help="Directory for downloaded ANN datasets",
        )
        parser.add_argument(
            "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
            help=f"Batch upsert size for indexing (default: {DEFAULT_BATCH_SIZE})",
        )

    def download(self, args: argparse.Namespace) -> None:
        from lib.download import download_file

        data_dir = Path(getattr(args, "data_dir", str(ROOT / "datasets" / "ann")))
        datasets = getattr(args, "dataset", None) or list(ANN_DATASETS.keys())
        if isinstance(datasets, str):
            datasets = [datasets]

        for name in datasets:
            if name not in ANN_DATASETS:
                print(f"Unknown ANN dataset: {name}")
                continue
            info = ANN_DATASETS[name]
            dest = data_dir / f"{name}.hdf5"
            print(f"Downloading ANN dataset: {name}")
            download_file(info["url"], dest, desc=name)
            print(f"  Done: {name}")

    def validate(self, args: argparse.Namespace) -> bool:
        # Check required imports
        try:
            import h5py  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as e:
            print(f"Missing dependency for ANN benchmarks: {e.name}")
            print("Install with: pip install h5py numpy")
            return False

        try:
            StrataClient._resolve_binary(None)
        except FileNotFoundError:
            print("strata CLI binary not found. Add it to PATH or set STRATA_BIN.")
            return False

        # Check dataset files exist
        data_dir = Path(args.data_dir)
        datasets = args.dataset if isinstance(args.dataset, list) else [args.dataset]
        for name in datasets:
            path = data_dir / f"{name}.hdf5"
            if not path.exists():
                print(f"Dataset file not found: {path}")
                print(f"Run download first: strata-eval download --bench ann --dataset {name}")
                return False
        return True

    def run(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        import numpy as np
        from .datasets import load_dataset

        results: list[BenchmarkResult] = []
        datasets = args.dataset if isinstance(args.dataset, list) else [args.dataset]
        k_values = sorted(args.k)
        max_k = max(k_values)

        for dataset_name in datasets:
            info = ANN_DATASETS[dataset_name]
            hdf5_path = Path(args.data_dir) / f"{dataset_name}.hdf5"

            print(f"\n{'='*60}")
            print(f"  ANN Benchmark: {dataset_name}")
            print(f"  Dimension: {info['dimension']}  |  Metric: {info['metric']}")
            print(f"{'='*60}")

            # Load dataset
            print("  Loading dataset...")
            data = load_dataset(hdf5_path)
            train = data["train"]
            test = data["test"]
            neighbors = data["neighbors"]

            num_train = len(train)
            num_test = len(test)

            # Validate dimensions
            if train.shape[1] != info["dimension"]:
                print(f"  WARNING: Dataset dimension {train.shape[1]} does not match "
                      f"config dimension {info['dimension']}")

            # Validate k vs ground truth
            gt_k = neighbors.shape[1] if len(neighbors.shape) > 1 else 0
            if max_k > gt_k:
                print(f"  WARNING: max k={max_k} exceeds ground truth depth {gt_k}. "
                      f"Recall values for k > {gt_k} will be unreliable.")

            print(f"  Index vectors: {num_train}  |  Query vectors: {num_test}")

            if num_test == 0:
                print("  ERROR: No test vectors in dataset, skipping.")
                continue

            # Build index using persistent client (needs temp-file upserts)
            with tempfile.TemporaryDirectory() as tmpdir:
                with StrataClient(db_path=tmpdir) as client:
                    collection_name = "bench"
                    collection = client.vectors.create(
                        collection_name,
                        train.shape[1],
                        info["metric"],
                    )

                    print(f"  Building index (batch_size={args.batch_size})...")
                    build_start = time.perf_counter()

                    for batch_start in range(0, num_train, args.batch_size):
                        batch_end = min(batch_start + args.batch_size, num_train)
                        entries = [
                            {"key": str(i), "vector": train[i].tolist()}
                            for i in range(batch_start, batch_end)
                        ]
                        collection.upsert(entries)

                        if (batch_start // args.batch_size) % 10 == 0:
                            progress = batch_end / num_train * 100
                            print(f"    {batch_end}/{num_train} ({progress:.0f}%)")

                    build_time_s = time.perf_counter() - build_start
                    print(f"  Index build: {build_time_s:.2f}s")

                    # Get index stats before closing
                    memory_bytes = 0
                    try:
                        stats = collection.stats()
                        if isinstance(stats, dict):
                            memory_bytes = stats.get("memory_bytes", 0)
                    except Exception:
                        pass

                # Client closed — DB is flushed and ready for batch_execute

                # -- Generate query commands (untimed) ---------------------
                print(f"  Generating {num_test} query commands...")
                query_cmds: list[str] = []
                for i in range(num_test):
                    q_json = json.dumps(test[i].tolist())
                    query_cmds.append(
                        f"vector search {shlex.quote(collection_name)} "
                        f"{shlex.quote(q_json)} {max_k}"
                    )

                # -- Warmup via CLI ----------------------------------------
                warmup_count = min(_WARMUP_QUERIES, num_test)
                if warmup_count > 0:
                    print(f"  Warmup: {warmup_count} queries via CLI...")
                    batch_execute(
                        query_cmds[:warmup_count],
                        db_path=tmpdir,
                        parse_responses=False,
                    )

                # -- Query via CLI batch (timed) ---------------------------
                print(f"  Querying {num_test} vectors via CLI (k={max_k})...")
                total_query_s, query_responses = batch_execute(
                    query_cmds,
                    db_path=tmpdir,
                )

                qps = num_test / total_query_s if total_query_s > 0 else 0
                avg_latency_us = (total_query_s / num_test) * 1e6

                print(f"  Query time: {total_query_s:.2f}s  |  QPS: {qps:.1f}")
                print(f"  Avg latency: {avg_latency_us:.1f} us")

                # -- Compute recall@k -------------------------------------
                recall_at_k: dict[str, float] = {}
                for k in k_values:
                    total_recall = 0.0
                    for i in range(num_test):
                        hits = query_responses[i] if i < len(query_responses) else []
                        result_list = hits[:k] if isinstance(hits, list) else []
                        returned_keys: set[str] = set()
                        for h in result_list:
                            if isinstance(h, dict):
                                returned_keys.add(str(h.get("key", h.get("id", ""))))
                            else:
                                returned_keys.add(str(h))

                        gt_depth = min(k, gt_k)
                        true_neighbor_keys = {str(j) for j in neighbors[i][:gt_depth]}

                        denom = min(k, gt_depth)
                        if denom > 0:
                            total_recall += len(returned_keys & true_neighbor_keys) / denom

                    recall_at_k[f"recall_at_{k}"] = round(total_recall / num_test, 5)
                    print(f"  Recall@{k}: {recall_at_k[f'recall_at_{k}']:.5f}")

                # Build result
                metrics: dict[str, object] = {
                    **recall_at_k,
                    "qps": round(qps, 1),
                    "avg_latency_us": round(avg_latency_us, 1),
                    "build_time_s": round(build_time_s, 2),
                    "total_query_time_s": round(total_query_s, 2),
                    "index_memory_bytes": memory_bytes,
                    "num_index_vectors": num_train,
                    "num_queries": num_test,
                }

                result = BenchmarkResult(
                    benchmark=f"ann/{dataset_name}",
                    category="ann",
                    parameters={
                        "dataset": dataset_name,
                        "dimension": int(train.shape[1]),
                        "metric": info["metric"],
                        "train_size": num_train,
                        "test_size": num_test,
                        "k_values": k_values,
                        "batch_size": args.batch_size,
                    },
                    metrics=metrics,
                )

                results.append(result)

        return results
