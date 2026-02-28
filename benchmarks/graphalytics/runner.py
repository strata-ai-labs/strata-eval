"""LDBC Graphalytics benchmark runner -- evaluates graph algorithm performance on Strata.

Uses Strata's native graph API via the CLI subprocess wrapper.  BFS runs
natively on the Strata engine.  Other algorithms read the graph via
``graph neighbors`` / ``graph list-nodes`` and execute in Python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

from lib.schema import BenchmarkResult
from benchmarks.base import BaseBenchmark
from .config import (
    ALGORITHMS,
    DEFAULT_CDLP_ITERATIONS,
    DEFAULT_PAGERANK_DAMPING,
    DEFAULT_PAGERANK_ITERATIONS,
    DEFAULT_RUNS,
    LDBC_DATASETS,
)
from .ldbc import load_dataset, load_reference
from . import algorithms as algo
from .algorithms import UNREACHABLE_INT, UNREACHABLE_FLOAT

ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return the *p*-th percentile from a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = min(max(math.ceil(p / 100.0 * n) - 1, 0), n - 1)
    return sorted_vals[idx]


def _fmt_num(n: int | float) -> str:
    """Format a number with thousands separators."""
    return f"{int(n):,}"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_EPSILON = 1e-6


def _validate_exact(
    result: dict[int, int],
    reference: dict[int, str],
    name: str,
) -> tuple[bool, list[str]]:
    """Validate exact integer match (BFS, CDLP)."""
    mismatches: list[str] = []
    for vid, ref_str in reference.items():
        ref_val = int(ref_str)
        actual = result.get(vid)
        if actual is None:
            mismatches.append(f"  vertex {vid}: missing (expected {ref_val})")
        elif actual != ref_val:
            mismatches.append(f"  vertex {vid}: expected {ref_val}, got {actual}")
        if len(mismatches) >= 10:
            break
    ok = len(mismatches) == 0
    return ok, mismatches


def _validate_partition(
    result: dict[int, int],
    reference: dict[int, str],
) -> tuple[bool, list[str]]:
    """Validate WCC partition equivalence (bidirectional).

    Two vertices should be in the same component iff they share the same
    label in *both* the result and the reference -- the actual label values
    may differ. Checks both directions: ref -> result and result -> ref.
    """
    mismatches: list[str] = []
    vids = sorted(reference.keys())

    # Build forward maps: label -> set of vertices
    ref_groups: dict[str, set[int]] = {}
    for vid in vids:
        ref_groups.setdefault(reference[vid], set()).add(vid)

    result_groups: dict[int, set[int]] = {}
    for vid in vids:
        label = result.get(vid)
        if label is None:
            mismatches.append(f"  vertex {vid}: missing from result")
            if len(mismatches) >= 10:
                break
            continue
        result_groups.setdefault(label, set()).add(vid)

    if mismatches:
        return False, mismatches

    # Check ref -> result: every reference group maps to exactly one result group
    for ref_label, ref_set in ref_groups.items():
        labels_seen = {result[vid] for vid in ref_set if vid in result}
        if len(labels_seen) != 1:
            sample = sorted(ref_set)[:5]
            mismatches.append(
                f"  ref group {ref_label}: split into {len(labels_seen)} result groups "
                f"(sample vertices: {sample})"
            )
            if len(mismatches) >= 10:
                break

    # Check result -> ref: every result group maps to exactly one ref group
    if not mismatches:
        for res_label, res_set in result_groups.items():
            ref_labels_seen = {reference[vid] for vid in res_set if vid in reference}
            if len(ref_labels_seen) != 1:
                sample = sorted(res_set)[:5]
                mismatches.append(
                    f"  result group {res_label}: merges {len(ref_labels_seen)} ref groups "
                    f"(sample vertices: {sample})"
                )
                if len(mismatches) >= 10:
                    break

    ok = len(mismatches) == 0
    return ok, mismatches


def _validate_epsilon(
    result: dict[int, float],
    reference: dict[int, str],
    name: str,
    epsilon: float = _EPSILON,
) -> tuple[bool, list[str]]:
    """Validate floating-point results within epsilon (PR, LCC, SSSP).

    Handles sentinel values: both LDBC integer sentinel (2^63-1) and
    math.inf are treated as equivalent for unreachable vertices.
    """
    mismatches: list[str] = []
    for vid, ref_str in reference.items():
        ref_val = float(ref_str)
        actual = result.get(vid)
        if actual is None:
            mismatches.append(f"  vertex {vid}: missing (expected {ref_val})")
        else:
            # Handle sentinel equivalence: LDBC uses large int, we use inf
            ref_is_unreachable = ref_val > 1e18 or math.isinf(ref_val)
            actual_is_unreachable = math.isinf(actual) or actual > 1e18

            if ref_is_unreachable and actual_is_unreachable:
                pass  # Both unreachable, OK
            elif ref_is_unreachable != actual_is_unreachable:
                mismatches.append(
                    f"  vertex {vid}: expected {'unreachable' if ref_is_unreachable else ref_val}, "
                    f"got {'unreachable' if actual_is_unreachable else actual}"
                )
            elif abs(actual - ref_val) > epsilon:
                mismatches.append(
                    f"  vertex {vid}: expected {ref_val}, got {actual} "
                    f"(diff={abs(actual - ref_val):.2e})"
                )
        if len(mismatches) >= 10:
            break
    ok = len(mismatches) == 0
    return ok, mismatches


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class GraphalyticsBenchmark(BaseBenchmark):
    """LDBC Graphalytics benchmark suite."""

    name = "graphalytics"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--algorithm", nargs="+", default=ALGORITHMS,
            choices=ALGORITHMS,
            help=f"Algorithm(s) to run (default: all {len(ALGORITHMS)})",
        )
        parser.add_argument(
            "--dataset", default="example-directed",
            choices=list(LDBC_DATASETS.keys()),
            help="LDBC dataset name (default: example-directed)",
        )
        parser.add_argument(
            "--runs", type=int, default=DEFAULT_RUNS,
            help=f"Number of timed runs per algorithm (default: {DEFAULT_RUNS})",
        )
        parser.add_argument(
            "--data-dir", type=str,
            default=str(ROOT / "datasets" / "graphalytics"),
            help="Directory containing LDBC dataset directories",
        )
        parser.add_argument(
            "--source", type=int, default=None,
            help="BFS/SSSP source vertex (default: from dataset properties)",
        )
        parser.add_argument(
            "--validate", dest="do_validate", action="store_true", default=True,
            help="Enable reference validation (default)",
        )
        parser.add_argument(
            "--no-validate", dest="do_validate", action="store_false",
            help="Disable reference validation",
        )

    def download(self, args: argparse.Namespace) -> None:
        dataset_name = getattr(args, "dataset", "example-directed")
        if isinstance(dataset_name, list):
            dataset_name = dataset_name[0] if dataset_name else "example-directed"
        ds_info = LDBC_DATASETS.get(dataset_name)
        if ds_info is None:
            print(f"Unknown dataset: {dataset_name}")
            return

        if ds_info.get("local"):
            print(f"Dataset '{dataset_name}' is a local example dataset.")
            data_dir = Path(getattr(args, "data_dir", str(ROOT / "datasets" / "graphalytics")))
            target = data_dir / dataset_name
            print(f"Ensure it exists at: {target}")
            print(
                "You can copy it from strata-benchmarks/data/graph/example-directed/ "
                "or create a symlink."
            )
            return

        url = ds_info["url"]
        print(f"Download the dataset archive from:")
        print(f"  {url}")
        print()
        print("Then extract with:")
        print(f"  zstd -d {dataset_name}.tar.zst && tar xf {dataset_name}.tar")
        print()
        data_dir = Path(getattr(args, "data_dir", str(ROOT / "datasets" / "graphalytics")))
        print(f"Place the extracted directory at: {data_dir / dataset_name}")

    def run(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        from lib.strata_client import StrataClient

        dataset_name = args.dataset
        data_dir = Path(args.data_dir) / dataset_name
        algorithms_to_run = args.algorithm
        num_runs = max(1, args.runs)
        do_validate = args.do_validate

        # Load LDBC dataset
        print(f"Loading LDBC dataset: {dataset_name} from {data_dir}")
        ds = load_dataset(data_dir)
        num_v = len(ds.vertices)
        num_e = len(ds.edges)
        print(
            f"  {_fmt_num(num_v)} vertices, {_fmt_num(num_e)} edges, "
            f"{'directed' if ds.directed else 'undirected'}"
        )

        # Determine BFS source vertex
        bfs_source = args.source
        if bfs_source is None:
            source_str = ds.properties.get("algorithms.bfs.source-vertex")
            if source_str is not None:
                bfs_source = int(source_str)
            else:
                bfs_source = ds.vertices[0] if ds.vertices else 0

        # Determine SSSP source vertex (may differ from BFS source)
        sssp_source = args.source
        if sssp_source is None:
            sssp_source_str = ds.properties.get("algorithms.sssp.source-vertex")
            if sssp_source_str is not None:
                sssp_source = int(sssp_source_str)
            else:
                sssp_source = bfs_source  # Fall back to BFS source

        # Read algorithm params from properties file
        pr_iterations = int(ds.properties.get(
            "algorithms.pr.num-iterations",
            str(DEFAULT_PAGERANK_ITERATIONS),
        ))
        pr_damping = float(ds.properties.get(
            "algorithms.pr.damping-factor",
            str(DEFAULT_PAGERANK_DAMPING),
        ))
        cdlp_iterations = int(ds.properties.get(
            "algorithms.cdlp.max-iterations",
            str(DEFAULT_CDLP_ITERATIONS),
        ))

        print(f"  BFS source: {bfs_source}")
        print(f"  SSSP source: {sssp_source}")

        # ------------------------------------------------------------------
        # Load graph into Strata via native graph API
        # ------------------------------------------------------------------
        with tempfile.TemporaryDirectory(prefix="strata_graphalytics_") as tmpdir, \
                StrataClient(db_path=tmpdir) as client:

            client.graph.create("bench")

            # Build bulk-insert payload from LDBC dataset
            nodes = [{"id": str(vid)} for vid in ds.vertices]
            edges = []
            for src, neighbors in ds.adj.items():
                for dst in neighbors:
                    weight = ds.edge_weights.get((src, dst), 1.0)
                    edges.append({
                        "src": str(src), "dst": str(dst),
                        "edge_type": "edge", "weight": float(weight),
                    })
            # For undirected graphs, insert reverse edges too
            if not ds.directed:
                reverse_edges = []
                existing = {(e["src"], e["dst"]) for e in edges}
                for e in edges:
                    if (e["dst"], e["src"]) not in existing:
                        reverse_edges.append({
                            "src": e["dst"], "dst": e["src"],
                            "edge_type": "edge", "weight": e["weight"],
                        })
                edges.extend(reverse_edges)

            print("Loading graph into Strata via bulk-insert...")
            t0 = time.perf_counter()

            # Write payload to temp file for bulk insert
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"nodes": nodes, "edges": edges}, f)
                bulk_file = f.name
            try:
                client.graph.bulk_insert("bench", file_path=bulk_file)
            finally:
                os.unlink(bulk_file)

            load_time = time.perf_counter() - t0
            print(f"  Load time: {load_time:.3f}s")

            # ------------------------------------------------------------------
            # Build adjacency dicts from graph API for Python algorithms
            # ------------------------------------------------------------------
            need_adj = any(a != "bfs" for a in algorithms_to_run)
            directed_adj: dict[int, list[int]] = {}
            undirected_adj: dict[int, list[int]] = {}
            read_time = 0.0

            if need_adj:
                print("Reading adjacency from Strata graph API...")
                t0 = time.perf_counter()
                for vid in ds.vertices:
                    # Directed (outgoing) neighbors
                    nbrs = client.graph.neighbors("bench", str(vid), direction="outgoing")
                    directed_adj[vid] = [int(n.get("id", n.get("node_id", n.get("dst", 0)))) for n in nbrs]
                    # Undirected (both) neighbors
                    nbrs_both = client.graph.neighbors("bench", str(vid), direction="both")
                    undirected_adj[vid] = [int(n.get("id", n.get("node_id", n.get("dst", 0)))) for n in nbrs_both]
                read_time = time.perf_counter() - t0
                print(f"  Read time: {read_time:.3f}s")

            # ------------------------------------------------------------------
            # Run each algorithm
            # ------------------------------------------------------------------
            results: list[BenchmarkResult] = []

            for alg_name in algorithms_to_run:
                print(f"\n{'='*60}")
                print(f"  Algorithm: {alg_name.upper()} ({num_runs} runs)")
                print(f"{'='*60}")

                # Choose the right adjacency list per algorithm
                if alg_name in ("wcc", "lcc"):
                    adj = undirected_adj
                else:
                    adj = directed_adj

                run_times: list[float] = []
                last_result: dict = {}

                for run_idx in range(num_runs):
                    t_start = time.perf_counter()

                    if alg_name == "bfs":
                        # Use native Strata BFS
                        bfs_out = client.graph.bfs(
                            "bench", str(bfs_source), 999999,
                            direction="outgoing",
                        )
                        # Convert to {int_vid: int_depth} format.
                        # JSON dict keys are always strings, so look up str(vid).
                        depths = bfs_out.get("depths", {})
                        last_result = {}
                        for vid in ds.vertices:
                            d = depths.get(str(vid))
                            if d is not None:
                                last_result[vid] = int(d)
                            else:
                                last_result[vid] = UNREACHABLE_INT
                    elif alg_name == "wcc":
                        last_result = algo.wcc(adj)
                    elif alg_name == "pagerank":
                        last_result = algo.pagerank(
                            adj,
                            iterations=pr_iterations,
                            damping=pr_damping,
                        )
                    elif alg_name == "cdlp":
                        last_result = algo.cdlp(adj, iterations=cdlp_iterations)
                    elif alg_name == "lcc":
                        last_result = algo.lcc(adj)
                    elif alg_name == "sssp":
                        last_result = algo.sssp(
                            adj, weights=ds.edge_weights, source=sssp_source,
                        )
                    else:
                        print(f"  Unknown algorithm: {alg_name}, skipping")
                        break

                    elapsed = time.perf_counter() - t_start
                    run_times.append(elapsed)

                    if run_idx == 0:
                        # EVPS = Edges Per Second (standard LDBC metric)
                        evps = num_e / elapsed if elapsed > 0 else 0
                        print(f"  Run 1: {elapsed*1000:.3f}ms (EVPS: {_fmt_num(evps)})")

                if not run_times:
                    continue

                # Validation
                validation_pass: bool | None = None
                if do_validate:
                    ref_name = f"{ds.name}-{alg_name.upper()}"
                    ref_path = ds.data_dir / ref_name
                    if ref_path.exists():
                        reference = load_reference(ref_path)
                        if alg_name in ("bfs", "cdlp"):
                            ok, details = _validate_exact(last_result, reference, alg_name)
                        elif alg_name == "wcc":
                            ok, details = _validate_partition(last_result, reference)
                        elif alg_name in ("pagerank", "lcc", "sssp"):
                            ok, details = _validate_epsilon(last_result, reference, alg_name)
                        else:
                            ok, details = True, []

                        validation_pass = ok
                        if ok:
                            print(f"  Validation: PASS ({len(reference)} vertices checked)")
                        else:
                            print(f"  Validation: FAIL")
                            for d in details:
                                print(d)
                    else:
                        print(f"  Validation: skipped (no reference file: {ref_name})")

                # Statistics
                sorted_times = sorted(run_times)
                avg_time = sum(sorted_times) / len(sorted_times)
                p50 = _percentile(sorted_times, 50)
                p95 = _percentile(sorted_times, 95)
                p99 = _percentile(sorted_times, 99)
                min_time = sorted_times[0]
                max_time = sorted_times[-1]
                # EVPS = Edges Per Second
                evps = num_e / avg_time if avg_time > 0 else 0

                print(f"\n  --- {alg_name.upper()} Summary ({len(run_times)} runs) ---")
                print(f"  avg:  {avg_time*1000:.3f}ms")
                print(f"  p50:  {p50*1000:.3f}ms")
                print(f"  p95:  {p95*1000:.3f}ms")
                print(f"  p99:  {p99*1000:.3f}ms")
                print(f"  min:  {min_time*1000:.3f}ms")
                print(f"  max:  {max_time*1000:.3f}ms")
                print(f"  EVPS: {_fmt_num(evps)}")

                metrics: dict[str, object] = {
                    "evps": evps,
                    "avg_ms": round(avg_time * 1000, 3),
                    "p50_ms": round(p50 * 1000, 3),
                    "p95_ms": round(p95 * 1000, 3),
                    "p99_ms": round(p99 * 1000, 3),
                    "min_ms": round(min_time * 1000, 3),
                    "max_ms": round(max_time * 1000, 3),
                    "load_time_s": round(load_time, 3),
                    "read_time_s": round(read_time, 3),
                    "runs": len(run_times),
                }
                if validation_pass is not None:
                    metrics["validation_pass"] = validation_pass

                source_for_alg = None
                if alg_name == "bfs":
                    source_for_alg = bfs_source
                elif alg_name == "sssp":
                    source_for_alg = sssp_source

                parameters: dict[str, object] = {
                    "dataset": dataset_name,
                    "algorithm": alg_name,
                    "vertices": num_v,
                    "edges": num_e,
                    "directed": ds.directed,
                    "source": source_for_alg,
                }
                # Add algorithm-specific parameters
                if alg_name == "pagerank":
                    parameters["iterations"] = pr_iterations
                    parameters["damping"] = pr_damping
                elif alg_name == "cdlp":
                    parameters["iterations"] = cdlp_iterations

                results.append(BenchmarkResult(
                    benchmark=f"graphalytics/{alg_name}/{dataset_name}/{num_v}V-{num_e}E",
                    category="graphalytics",
                    parameters=parameters,
                    metrics=metrics,
                ))

        print(f"\n{'='*60}")
        print(f"  Graphalytics benchmark complete: {len(results)} algorithm(s)")
        print(f"{'='*60}\n")

        return results
