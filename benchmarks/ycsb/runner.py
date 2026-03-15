"""YCSB benchmark runner -- evaluates Strata KV throughput and latency.

All timed operations are executed by piping pre-generated commands directly
to the ``strata`` CLI binary.  Python is used only for workload generation
(untimed) and result recording — there is no Python overhead per database
operation in the timing loop.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import tempfile
import time

from lib.schema import BenchmarkResult
from lib.strata_client import StrataClient, batch_execute
from benchmarks.base import BaseBenchmark
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
            StrataClient._resolve_binary(None)
            return True
        except FileNotFoundError:
            print("ERROR: strata CLI binary not found. Add it to PATH or set STRATA_BIN.")
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

        print(f"\n{'='*60}")
        print(f"  YCSB {spec.name}: {spec.description}")
        print(f"  records={records}  ops={ops}  fields={field_count}  "
              f"field_length={field_length}  distribution={dist_name}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory() as tmpdir:

            # -- Generate load commands (untimed) --------------------------
            print("  Generating load commands...")
            t_gen = time.perf_counter()
            load_cmds: list[str] = []
            for i in range(records):
                key = format_key(i)
                value = json.dumps(generate_value(field_count, field_length))
                load_cmds.append(f"kv put {shlex.quote(key)} {shlex.quote(value)}")
            load_cmds.append("flush")
            gen_time = time.perf_counter() - t_gen
            print(f"  Generated {len(load_cmds)} load commands in {gen_time:.2f}s")

            # -- Execute load via CLI (timed) ------------------------------
            print("  Loading records via CLI...")
            load_elapsed, _ = batch_execute(
                load_cmds,
                db_path=tmpdir,
                parse_responses=False,
            )
            load_throughput = records / load_elapsed if load_elapsed > 0 else 0
            print(f"  Load complete: {records} records in {load_elapsed:.2f}s "
                  f"({load_throughput:.0f} ops/s)")

            # -- Generate execute commands (untimed) -----------------------
            print("  Generating execute commands...")
            t_gen = time.perf_counter()
            key_counter = records
            gen = self._make_generator(dist_name, records)
            op_choices, op_weights = self._build_op_selector(spec)

            exec_cmds: list[str] = []
            op_counts: dict[str, int] = {}

            for _ in range(ops):
                op = random.choices(op_choices, weights=op_weights, k=1)[0]
                op_counts[op] = op_counts.get(op, 0) + 1

                if op == "read":
                    key = format_key(gen.next())
                    exec_cmds.append(f"kv get {shlex.quote(key)}")

                elif op == "update":
                    key = format_key(gen.next())
                    value = json.dumps(generate_value(field_count, field_length))
                    exec_cmds.append(f"kv put {shlex.quote(key)} {shlex.quote(value)}")

                elif op == "insert":
                    key = format_key(key_counter)
                    value = json.dumps(generate_value(field_count, field_length))
                    exec_cmds.append(f"kv put {shlex.quote(key)} {shlex.quote(value)}")
                    key_counter += 1
                    if hasattr(gen, "set_n"):
                        gen.set_n(key_counter)

                elif op == "scan":
                    key = format_key(gen.next())
                    scan_length = random.randint(1, max_scan_length)
                    exec_cmds.append(
                        f"kv list --prefix {shlex.quote(key)} --limit {scan_length}"
                    )

                elif op == "rmw":
                    # Read-modify-write: pre-generate the replacement value.
                    # The CLI will execute get then put sequentially.  We
                    # measure the combined cost of both operations.
                    key = format_key(gen.next())
                    value = json.dumps(generate_value(field_count, field_length))
                    exec_cmds.append(f"kv get {shlex.quote(key)}")
                    exec_cmds.append(f"kv put {shlex.quote(key)} {shlex.quote(value)}")

            gen_time = time.perf_counter() - t_gen
            print(f"  Generated {len(exec_cmds)} execute commands "
                  f"({ops} ops) in {gen_time:.2f}s")

            # -- Execute workload via CLI (timed) --------------------------
            print("  Running operations via CLI...")
            exec_elapsed, _ = batch_execute(
                exec_cmds,
                db_path=tmpdir,
                parse_responses=False,
            )
            exec_throughput = ops / exec_elapsed if exec_elapsed > 0 else 0

        # -- Aggregate metrics ---------------------------------------------
        avg_latency_us = (exec_elapsed / ops * 1e6) if ops > 0 else 0

        metrics: dict[str, object] = {
            "load_time_s": round(load_elapsed, 3),
            "load_throughput_ops": round(load_throughput, 1),
            "exec_time_s": round(exec_elapsed, 3),
            "exec_throughput_ops": round(exec_throughput, 1),
            "avg_latency_us": round(avg_latency_us, 2),
            "total_commands": len(exec_cmds),
        }

        # Per-operation breakdown
        for op_name, count in op_counts.items():
            metrics[f"{op_name}_count"] = count

        # -- Print summary -------------------------------------------------
        self._print_summary(spec, dist_name, records, ops, metrics, op_counts)

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
                       metrics: dict, op_counts: dict) -> None:
        print(f"\n  {'--- Results ---':^50}")
        print(f"  Load:    {metrics['load_time_s']:.3f}s  "
              f"({metrics['load_throughput_ops']:.0f} ops/s)")
        print(f"  Execute: {metrics['exec_time_s']:.3f}s  "
              f"({metrics['exec_throughput_ops']:.0f} ops/s)")
        print(f"  Avg latency: {metrics['avg_latency_us']:.1f} us/op")
        print(f"\n  {'--- Operation Mix ---':^50}")
        for op_name, count in sorted(op_counts.items()):
            print(f"  {op_name:<12} {count:>10}")
        print(f"{'='*60}\n")
