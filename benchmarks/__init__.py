"""Benchmark registry — lazy imports so missing optional deps don't crash the CLI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseBenchmark


def get_benchmarks() -> dict[str, type[BaseBenchmark]]:
    """Return available benchmark classes, skipping those with missing deps."""
    registry: dict[str, type[BaseBenchmark]] = {}

    def _try_register(name: str, module: str, cls_name: str) -> None:
        try:
            mod = __import__(module, fromlist=[cls_name])
            registry[name] = getattr(mod, cls_name)
        except ImportError as e:
            # Only suppress if the missing module is an expected optional dep.
            # Re-raise if it's an internal import error (typo, broken code).
            missing = getattr(e, "name", None)
            # These are the external optional deps that justify silently skipping.
            expected_missing = {
                "beir", "sentence_transformers", "pytrec_eval",
                "h5py", "numpy", "ragas", "rouge_score", "openai",
            }
            if missing and missing.split(".")[0] in expected_missing:
                pass  # optional dep not installed -- skip benchmark
            else:
                print(f"Warning: failed to load {name} benchmark: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: failed to load {name} benchmark: {e}", file=sys.stderr)

    _try_register("beir", "benchmarks.beir.runner", "BeirBenchmark")
    _try_register("ycsb", "benchmarks.ycsb.runner", "YcsbBenchmark")
    _try_register("ann", "benchmarks.ann.runner", "AnnBenchmark")
    _try_register("graphalytics", "benchmarks.graphalytics.runner", "GraphalyticsBenchmark")
    _try_register("locomo", "benchmarks.locomo.runner", "LocomoBenchmark")
    _try_register("longmemeval", "benchmarks.longmemeval.runner", "LongMemEvalBenchmark")
    _try_register("ragas", "benchmarks.ragas_bench.runner", "RagasBenchmark")
    _try_register("graphrag", "benchmarks.graphrag_bench.runner", "GraphRagBenchmark")

    return registry
