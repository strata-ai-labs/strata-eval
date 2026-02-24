"""YCSB benchmark runner -- evaluates Strata KV throughput and latency.

Note: This is a single-threaded YCSB implementation. Standard YCSB supports
multi-threaded operation, but this version focuses on single-client sequential
performance to isolate database latency from concurrency overhead.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time

from ...schema import BenchmarkResult
from ..base import BaseBenchmark
from .config import (
    DEFAULT_RECORD_COUNT,
    DEFAULT_OPERATION_COUNT,
    DEFAULT_FIELD_COUNT,
    DEFAULT_FIELD_LENGTH,
)
from .workloads import (
    WORKLOADS,
    WorkloadSpec,
    ZipfianGenerator,
    UniformGenerator,
    LatestGenerator,
    generate_value,
    format_key,
)


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentiles(latencies_ns: list[int]) -> dict[str, float]:
    """Compute p50, p95, p99, p99.9 from a list of nanosecond latencies.

    Uses the nearest-rank method. Returns values in **microseconds**.
    """
    if not latencies_ns:
        return {"p50_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "p99_9_us": 0.0}
    latencies_ns.sort()
    n = len(latencies_ns)

    def _pct(p: float) -> float:
        # Nearest-rank: ceil(p/100 * n) - 1, clamped to [0, n-1]
        import math
        idx = min(max(math.ceil(p / 100.0 * n) - 1, 0), n - 1)
        return latencies_ns[idx] / 1000.0  # ns -> us

    return {
        "p50_us": round(_pct(50), 2),
        "p95_us": round(_pct(95), 2),
        "p99_us": round(_pct(99), 2),
        "p99_9_us": round(_pct(99.9), 2),
    }


# ---------------------------------------------------------------------------
# YCSB benchmark
# ---------------------------------------------------------------------------

class YcsbBenchmark(BaseBenchmark):
    name = "ycsb"

    # ---- CLI registration -------------------------------------------------

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--workload",
            nargs="+",
            default=list(WORKLOADS.keys()),
            choices=list(WORKLOADS.keys()),
            help="Workloads to run (default: all a-f)",
        )
        parser.add_argument(
            "--records",
            type=int,
            default=DEFAULT_RECORD_COUNT,
            help=f"Number of records to pre-load (default: {DEFAULT_RECORD_COUNT})",
        )
        parser.add_argument(
            "--ops",
            type=int,
            default=DEFAULT_OPERATION_COUNT,
            help=f"Number of operations in run phase (default: {DEFAULT_OPERATION_COUNT})",
        )
        parser.add_argument(
            "--fields",
            type=int,
            default=DEFAULT_FIELD_COUNT,
            help=f"Number of fields per record (default: {DEFAULT_FIELD_COUNT})",
        )
        parser.add_argument(
            "--field-length",
            type=int,
            default=DEFAULT_FIELD_LENGTH,
            help=f"Length of each field in bytes (default: {DEFAULT_FIELD_LENGTH})",
        )
        parser.add_argument(
            "--distribution",
            type=str,
            default=None,
            choices=["zipfian", "uniform", "latest"],
            help="Override key distribution for all workloads",
        )
        parser.add_argument(
            "--max-scan-length",
            type=int,
            default=100,
            help="Maximum scan length for workload E (default: 100)",
        )

    # ---- Download ---------------------------------------------------------

    def download(self, args: argparse.Namespace) -> None:
        print("YCSB uses synthetic data, no download needed.")

    # ---- Validate ---------------------------------------------------------

    def validate(self, args: argparse.Namespace) -> bool:
        try:
            from stratadb import Strata  # noqa: F401
            return True
        except ImportError:
            print("ERROR: stratadb is not installed. Install it to run YCSB benchmarks.")
            return False

    # ---- Run --------------------------------------------------------------

    def run(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        records = args.records
        if records < 1:
            print("ERROR: --records must be >= 1")
            return []

        results: list[BenchmarkResult] = []
        for wl_key in args.workload:
            spec = WORKLOADS[wl_key]
            dist_name = args.distribution or spec.distribution
            result = self._run_workload(
                spec=spec,
                wl_key=wl_key,
                records=records,
                ops=args.ops,
                field_count=args.fields,
                field_length=args.field_length,
                dist_name=dist_name,
                max_scan_length=args.max_scan_length,
            )
            results.append(result)
        return results

    # ---- Single workload --------------------------------------------------

    def _run_workload(
        self,
        spec: WorkloadSpec,
        wl_key: str,
        records: int,
        ops: int,
        field_count: int,
        field_length: int,
        dist_name: str,
        max_scan_length: int = 100,
    ) -> BenchmarkResult:
        from stratadb import Strata

        print(f"\n{'='*60}")
        print(f"  YCSB {spec.name}: {spec.description}")
        print(f"  records={records}  ops={ops}  fields={field_count}  "
              f"field_length={field_length}  distribution={dist_name}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory() as tmpdir:
            db = Strata.open(tmpdir)

            # -- Load phase ------------------------------------------------
            print("  Loading records...")
            load_start = time.perf_counter()

            # Pre-generate values and batch insert in chunks for efficiency
            BATCH_SIZE = 1000
            for batch_start in range(0, records, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, records)
                for i in range(batch_start, batch_end):
                    key = format_key(i)
                    value = generate_value(field_count, field_length)
                    db.kv.put(key, json.dumps(value))

            load_elapsed = time.perf_counter() - load_start
            load_throughput = records / load_elapsed if load_elapsed > 0 else 0
            print(f"  Load complete: {records} records in {load_elapsed:.2f}s "
                  f"({load_throughput:.0f} ops/s)")

            # -- Execute phase ---------------------------------------------
            print("  Running operations...")
            key_counter = records  # for inserts

            # Build the distribution generator
            gen = self._make_generator(dist_name, records)

            # Build the weighted operation selector
            op_choices, op_weights = self._build_op_selector(spec)

            # Per-operation latency tracking
            latencies: dict[str, list[int]] = {
                "read": [], "update": [], "insert": [], "scan": [], "rmw": [],
            }

            exec_start = time.perf_counter()
            for _ in range(ops):
                op = random.choices(op_choices, weights=op_weights, k=1)[0]

                if op == "read":
                    idx = gen.next()
                    key = format_key(idx)
                    t0 = time.perf_counter_ns()
                    db.kv.get(key)
                    latencies["read"].append(time.perf_counter_ns() - t0)

                elif op == "update":
                    idx = gen.next()
                    key = format_key(idx)
                    # Generate value OUTSIDE the timing window
                    new_value = json.dumps(generate_value(field_count, field_length))
                    t0 = time.perf_counter_ns()
                    db.kv.put(key, new_value)
                    latencies["update"].append(time.perf_counter_ns() - t0)

                elif op == "insert":
                    new_key = format_key(key_counter)
                    # Generate value OUTSIDE the timing window
                    value = json.dumps(generate_value(field_count, field_length))
                    t0 = time.perf_counter_ns()
                    db.kv.put(new_key, value)
                    latencies["insert"].append(time.perf_counter_ns() - t0)
                    key_counter += 1
                    # Update latest-aware generators (incremental O(1) update)
                    if hasattr(gen, "set_n"):
                        gen.set_n(key_counter)

                elif op == "scan":
                    idx = gen.next()
                    key = format_key(idx)
                    scan_length = random.randint(1, max_scan_length)
                    # Range scan: use the full key as the prefix start.
                    # db.kv.list with prefix=key returns keys >= key with
                    # the same prefix, approximating a YCSB range scan.
                    t0 = time.perf_counter_ns()
                    db.kv.list(prefix=key, limit=scan_length)
                    latencies["scan"].append(time.perf_counter_ns() - t0)

                elif op == "rmw":
                    idx = gen.next()
                    key = format_key(idx)
                    # Read-modify-write: read the existing value, modify it,
                    # then write back. Value generation is outside timing.
                    new_field_value = "".join(
                        random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=field_length)
                    )
                    t0 = time.perf_counter_ns()
                    existing = db.kv.get(key)
                    # Modify: replace field0 in the existing record
                    if existing:
                        try:
                            record = json.loads(existing)
                            record["field0"] = new_field_value
                            db.kv.put(key, json.dumps(record))
                        except (json.JSONDecodeError, TypeError):
                            db.kv.put(key, json.dumps({"field0": new_field_value}))
                    else:
                        db.kv.put(key, json.dumps({"field0": new_field_value}))
                    latencies["rmw"].append(time.perf_counter_ns() - t0)

            exec_elapsed = time.perf_counter() - exec_start
            exec_throughput = ops / exec_elapsed if exec_elapsed > 0 else 0

        # -- Aggregate metrics ---------------------------------------------
        all_latencies: list[int] = []
        for lats in latencies.values():
            all_latencies.extend(lats)

        overall_pct = _percentiles(all_latencies)

        metrics: dict[str, object] = {
            "load_time_s": round(load_elapsed, 3),
            "load_throughput_ops": round(load_throughput, 1),
            "exec_time_s": round(exec_elapsed, 3),
            "exec_throughput_ops": round(exec_throughput, 1),
            "overall_p50_us": overall_pct["p50_us"],
            "overall_p95_us": overall_pct["p95_us"],
            "overall_p99_us": overall_pct["p99_us"],
            "overall_p99_9_us": overall_pct["p99_9_us"],
        }

        # Per-operation breakdown
        for op_name, lats in latencies.items():
            if lats:
                pct = _percentiles(lats)
                metrics[f"{op_name}_count"] = len(lats)
                metrics[f"{op_name}_p50_us"] = pct["p50_us"]
                metrics[f"{op_name}_p95_us"] = pct["p95_us"]
                metrics[f"{op_name}_p99_us"] = pct["p99_us"]
                metrics[f"{op_name}_p99_9_us"] = pct["p99_9_us"]

        # -- Print summary -------------------------------------------------
        self._print_summary(spec, dist_name, records, ops, metrics)

        # Benchmark name: use human-readable record count
        rec_label = f"{records // 1000}k" if records >= 1000 else str(records)

        return BenchmarkResult(
            benchmark=f"ycsb/{spec.name}/{rec_label}-{dist_name}",
            category="ycsb",
            parameters={
                "workload": wl_key,
                "workload_name": spec.name,
                "record_count": records,
                "operation_count": ops,
                "field_count": field_count,
                "field_length": field_length,
                "distribution": dist_name,
            },
            metrics=metrics,
        )

    # ---- Helpers ----------------------------------------------------------

    @staticmethod
    def _make_generator(dist_name: str, n: int):
        """Create the appropriate key distribution generator."""
        if dist_name == "zipfian":
            return ZipfianGenerator(n)
        elif dist_name == "uniform":
            return UniformGenerator(n)
        elif dist_name == "latest":
            return LatestGenerator(n)
        else:
            raise ValueError(f"Unknown distribution: {dist_name}")

    @staticmethod
    def _build_op_selector(spec: WorkloadSpec) -> tuple[list[str], list[float]]:
        """Return (op_names, weights) lists for random.choices."""
        ops = []
        weights = []
        if spec.read_proportion > 0:
            ops.append("read")
            weights.append(spec.read_proportion)
        if spec.update_proportion > 0:
            ops.append("update")
            weights.append(spec.update_proportion)
        if spec.insert_proportion > 0:
            ops.append("insert")
            weights.append(spec.insert_proportion)
        if spec.scan_proportion > 0:
            ops.append("scan")
            weights.append(spec.scan_proportion)
        if spec.rmw_proportion > 0:
            ops.append("rmw")
            weights.append(spec.rmw_proportion)
        return ops, weights

    @staticmethod
    def _print_summary(spec: WorkloadSpec, dist_name: str, records: int, ops: int,
                       metrics: dict) -> None:
        print(f"\n  {'--- Results ---':^50}")
        print(f"  Load:    {metrics['load_time_s']:.3f}s  "
              f"({metrics['load_throughput_ops']:.0f} ops/s)")
        print(f"  Execute: {metrics['exec_time_s']:.3f}s  "
              f"({metrics['exec_throughput_ops']:.0f} ops/s)")
        print(f"\n  {'--- Latency (microseconds) ---':^50}")
        print(f"  {'Operation':<12} {'p50':>10} {'p95':>10} {'p99':>10} {'p99.9':>10}")
        print(f"  {'-'*52}")
        print(f"  {'overall':<12} "
              f"{metrics['overall_p50_us']:>10.1f} "
              f"{metrics['overall_p95_us']:>10.1f} "
              f"{metrics['overall_p99_us']:>10.1f} "
              f"{metrics['overall_p99_9_us']:>10.1f}")
        for op_name in ("read", "update", "insert", "scan", "rmw"):
            count_key = f"{op_name}_count"
            if count_key in metrics:
                print(f"  {op_name:<12} "
                      f"{metrics[f'{op_name}_p50_us']:>10.1f} "
                      f"{metrics[f'{op_name}_p95_us']:>10.1f} "
                      f"{metrics[f'{op_name}_p99_us']:>10.1f} "
                      f"{metrics[f'{op_name}_p99_9_us']:>10.1f}"
                      f"  (n={metrics[count_key]})")
        print(f"{'='*60}\n")
