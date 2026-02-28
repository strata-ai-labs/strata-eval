# Strata-Eval: Epics and Stories

Benchmarking infrastructure to support 12 academic papers on Strata. Organized as
epics (major work areas) and stories (individual deliverables). Each story lists
files to create/modify, complexity (S/M/L), and dependencies.

See RESEARCH_ROADMAP.md for the papers these benchmarks support.

---

## Table of Contents

- [Epic 0: Foundation — Extend StrataClient](#epic-0-foundation--extend-strataclient)
- [Epic 1: Core Infrastructure](#epic-1-core-infrastructure)
- [Epic 2: Paper 1 — Architecture Benchmarks](#epic-2-paper-1--architecture-benchmarks)
- [Epic 3: Paper 2 — Hybrid Search Benchmarks](#epic-3-paper-2--hybrid-search-benchmarks)
- [Epic 4: Paper 3 — Inference Benchmarks (Scaffold)](#epic-4-paper-3--inference-benchmarks-scaffold)
- [Epic 5: Paper 4 — RAG Benchmarks](#epic-5-paper-4--rag-benchmarks)
- [Epic 6: Future Paper Stubs (Papers 5-12)](#epic-6-future-paper-stubs-papers-5-12)
- [Implementation Sequencing](#implementation-sequencing)
- [Verification Criteria](#verification-criteria)

---

## Current State

**Fully implemented benchmarks:**
- BEIR — 15 datasets, keyword + hybrid modes, nDCG/MAP/Recall/MRR metrics
- YCSB — 6 workloads (A-F), throughput + p50/p95/p99 latencies
- ANN — 3 datasets, recall@K + QPS
- Graphalytics — 6 algorithms, LDBC datasets, timing + validation

**Scaffolded/stubbed:**
- LoCoMo (retrieval works, LLM eval TODO)
- RAGAS (retrieval works, LLM eval TODO)
- LongMemEval (structure only)
- GraphRAG (minimal stub)

**Core libraries:**
- `lib/strata_client.py` — CLI subprocess wrapper (KV, Vector, Graph namespaces only)
- `lib/schema.py` — BenchmarkResult, BenchmarkReport dataclasses
- `lib/recorder.py` — JSON result serialization
- `lib/system_info.py` — hardware/git metadata
- `lib/report.py` — Markdown/LaTeX report generation
- `lib/download.py` — dataset downloader
- `benchmarks/base.py` — BaseBenchmark abstract class
- `scripts/run_all.py` — batch runner
- `run.py` — main CLI entry point

**What's missing for publishable research:**
1. No YAML experiment configuration system (everything is CLI-arg-driven)
2. No multi-run execution with seed management
3. No statistical analysis (significance tests, confidence intervals, effect sizes)
4. No cross-experiment comparison engine
5. No ablation support for BEIR (toggling individual pipeline components)
6. No baseline system runners (Redis, Elasticsearch, SQLite, etc.)
7. No hardware profiling integration (CPU/memory sampling during runs)
8. No publication-quality charts (Pareto curves, ablation waterfalls)
9. StrataClient missing JSON, Event, State, and Inference namespaces

---

## Epic 0: Foundation — Extend StrataClient

The CLI wrapper only exposes KV, Vector, and Graph namespaces. Paper 1 microbenchmarks
need JSON, Event, State, and Inference access. This unblocks almost everything else.

### Story 0.1: Add missing primitive namespaces to StrataClient

**Complexity:** L

**What:** Extend `lib/strata_client.py` with full primitive coverage.

**New namespaces:**

| Namespace | Methods |
|-----------|---------|
| `client.json` | `put(collection, key, doc)`, `get(collection, key)`, `delete(collection, key)`, `query(collection, path, value)` |
| `client.event` | `append(event_type, payload)`, `list(event_type, limit)`, `query(event_type, after, before)` |
| `client.state` | `init(cell, value)`, `get(cell)`, `cas(cell, value, expected_version)`, `getv(cell)`, `delete(cell)` |
| `client.inference` | `embed(text)`, `generate(prompt, **kwargs)`, `models_pull(name)`, `models_list()` |
| `client.database` | `branch_create(name)`, `branch_list()`, `branch_delete(name)`, `info()`, `compact()` |

**Also add to existing KV namespace:**
- `cas(key, value, version) -> version`
- `getv(key) -> (value, version)`

**Files to modify:**
- `lib/strata_client.py` — add namespace classes, extend `_unwrap()`
- `tests/test_strata_client.py` — add test classes for each new namespace

**Dependencies:** None

---

## Epic 1: Core Infrastructure

Shared harness that all papers depend on. Built first, used everywhere.
The principle: coexist with the existing CLI-arg approach, don't replace it.

### Story 1.1: Experiment configuration system

**Complexity:** M

**What:** YAML-driven declarative experiment configs for reproducible runs.

**ExperimentConfig schema:**
```yaml
experiment:
  name: "beir-keyword-baseline-v1"
  date: "2026-02-27"
  hypothesis: "BM25 keyword search establishes baseline nDCG@10 on BEIR"

dataset:
  name: "nfcorpus"
  source: "beir"
  split: "test"

system:
  base: "strata"
  version: "0.6.0"
  commit: "auto"  # auto-detected from git

  components:
    bm25: true
    vectors: false
    expansion: false
    reranking: false

  fixed_params:
    bm25_k1: 1.2
    bm25_b: 0.75
    top_k: 10

hardware:  # auto-detected if omitted
  cpu: "auto"
  ram: "auto"
  gpu: "auto"

runs: 5
seed_base: 42
warmup_runs: 1
```

**Files to create:**
- `lib/experiment.py` — `ExperimentConfig` dataclass, `load_experiment(path)`, YAML validation
- `configs/examples/beir_keyword_baseline.yaml` — working example

**Files to modify:**
- `requirements.txt` — add `pyyaml>=6.0`

**Dependencies:** None

---

### Story 1.2: Statistical analysis module

**Complexity:** M

**What:** Pure functions for computing aggregate statistics and significance tests.

**Functions:**
- `aggregate_metrics(runs: list[dict]) -> AggregatedMetrics`
  - Computes mean, std, 95% CI, median, IQR for each metric
- `paired_significance(a: list[float], b: list[float], test="t-test") -> SignificanceResult`
  - Returns p-value, test statistic, effect size (Cohen's d)
- `cohen_d(a: list[float], b: list[float]) -> float`
- `bonferroni_correction(p_values: list[float], alpha=0.05) -> list[bool]`
- `holm_bonferroni(p_values: list[float], alpha=0.05) -> list[bool]`

**Output format compatible with report.py:**
```python
{
    "ndcg_at_10_mean": 0.341,
    "ndcg_at_10_std": 0.003,
    "ndcg_at_10_ci_lo": 0.337,
    "ndcg_at_10_ci_hi": 0.345
}
```

**Files to create:**
- `lib/stats.py`

**Files to modify:**
- `requirements.txt` — add `scipy>=1.10` (optional, pure-Python fallback for t-test)

**Dependencies:** None (standalone module)

---

### Story 1.3: Multi-run executor

**Complexity:** M

**What:** An executor that runs a benchmark N times with deterministic seeds, discards
warmup runs, and aggregates results with statistics.

**Behavior:**
1. Set `random.seed(seed)` and `numpy.random.seed(seed)` before each run
2. Run `warmup_runs` iterations, discard results
3. Run `runs` timed iterations, collect each `list[BenchmarkResult]`
4. Aggregate via stats module
5. Return `ExperimentRun` with individual + aggregate results

**Files to create:**
- `lib/executor.py` — `Executor` class, `ExperimentRun` dataclass

**Files to modify:**
- `lib/schema.py` — add `ExperimentRun` dataclass; add optional `run_index`, `seed`
  fields to `BenchmarkResult`; bump `schema_version` to 2 (backward-compatible)
- `lib/recorder.py` — add `save_experiment()` method
- `benchmarks/base.py` — add optional `set_seed(seed)` method (default no-op)

**Dependencies:** 1.1, 1.2

---

### Story 1.4: Comparison engine

**Complexity:** M

**What:** Load two or more experiment results, align by benchmark name, compute deltas,
run significance tests, produce structured comparison.

**Behavior:**
- Aligns results by `benchmark` field (e.g., `beir/nfcorpus/hybrid`)
- For each shared benchmark: delta, percentage change, significance (via `lib/stats.py`)
- Handles different experiment scopes (reports intersection + notes on missing entries)
- Outputs human-readable summary + structured data for charts

**Files to create:**
- `lib/comparison.py` — `ComparisonEngine`, `ComparisonResult`, `MetricComparison`
- `scripts/compare_experiments.py` — CLI wrapper

**Dependencies:** 1.2

---

### Story 1.5: Publication-ready output

**Complexity:** L

**What:** Publication-quality LaTeX tables and matplotlib charts.

**Chart types:**
- `bar_chart_with_error_bars()` — grouped bars with CI whiskers
- `pareto_curve()` — quality vs latency scatter with Pareto frontier highlighted
- `ablation_waterfall()` — cumulative component contribution chart
- `heatmap_table()` — dataset x system matrix (BEIR-style)
- `radar_chart()` — multi-dimensional system comparison

**LaTeX enhancements:**
- Bold formatting for best result per row
- `*`/`**`/`***` significance markers (p < 0.05 / 0.01 / 0.001)
- `±` standard deviations
- `\cmidrule` grouping, booktabs style

**Files to create:**
- `lib/charts.py` — matplotlib with pgf backend for vector PDF/PGF output
- `scripts/generate_paper_tables.py` — per-paper table/chart CLI

**Files to modify:**
- `lib/report.py` — enhance `_generate_latex()` with significance markers, bold best
- `requirements.txt` — add `matplotlib>=3.7` (optional)

**Dependencies:** 1.2

---

### Story 1.6: Hardware profiler

**Complexity:** M

**What:** Sample CPU, memory, and disk I/O during benchmark execution.

**Behavior:**
- Context manager that spawns a background sampling thread
- Configurable interval (default 100ms)
- Tracks the Strata subprocess specifically (via `StrataClient._proc.pid`)
- Can also monitor external processes (for baseline comparisons)
- Returns `ProfileResult`: time-series + summary (peak_rss_mb, avg_cpu_pct, disk_io)
- Uses `psutil` if available, falls back to `/proc` parsing on Linux

**Files to create:**
- `lib/profiler.py`

**Files to modify:**
- `lib/schema.py` — add optional `profile` field to `BenchmarkResult`
- `lib/system_info.py` — add `gpu_info()` via nvidia-smi parsing

**Dependencies:** None

---

### Story 1.7: Experiment runner script

**Complexity:** S

**What:** Ties together Stories 1.1-1.6 into a single CLI.

**Usage:**
```bash
python scripts/run_experiment.py configs/paper1/kv_microbench.yaml
python scripts/run_experiment.py config.yaml --compare results/baseline.json
```

**Files to create:**
- `scripts/run_experiment.py`

**Files to modify:**
- `run.py` — add `experiment` subcommand

**Dependencies:** 1.1-1.6

---

## Epic 2: Paper 1 — Architecture Benchmarks

Microbenchmarks for each primitive, cross-primitive macrobenchmarks, resource
efficiency comparison against the polyglot stack.

### Story 2.1: Microbenchmark suite — KV

**Complexity:** M

**What:** Benchmark KV operations at varying value sizes and record counts.

**Parameters:**
- Value sizes: 64B, 1KB, 10KB, 100KB
- Record counts: 1K, 10K, 100K, 1M
- Operations: put, get, delete, cas, getv, list

**Metrics:** throughput (ops/s), p50/p95/p99 latency (us)

**Files to create:**
- `benchmarks/microbench/__init__.py`
- `benchmarks/microbench/config.py` — value sizes, record counts, operation types
- `benchmarks/microbench/kv_runner.py` — `KvMicrobenchmark` class

**Pattern:** Follow timing approach from `benchmarks/ycsb/runner.py`

**Dependencies:** 0.1 (cas, getv in client)

---

### Story 2.2: Microbenchmark suite — JSON, Event, State

**Complexity:** M

**What:** Microbenchmarks for the remaining primitives.

**JSON benchmarks:**
- Insert, get, delete, query at varying document sizes (1KB, 10KB, 100KB)
- Collection sizes: 1K, 10K, 100K documents

**Event benchmarks:**
- Append throughput (events/sec)
- List/query latency at varying log sizes
- Hash-chain verification speed (events are SHA-256 chained)

**State benchmarks:**
- init, get, cas throughput
- CAS contention: measure retry rates under concurrent access

**Files to create:**
- `benchmarks/microbench/json_runner.py`
- `benchmarks/microbench/event_runner.py`
- `benchmarks/microbench/state_runner.py`

**Dependencies:** 0.1

---

### Story 2.3: Microbenchmark suite — BM25 search

**Complexity:** M

**What:** BM25 indexing throughput and query latency in isolation (no vectors).
Differs from BEIR: this measures raw performance at different corpus scales
with synthetic data, not retrieval quality with real queries.

**Parameters:**
- Corpus sizes: 1K, 10K, 100K, 1M synthetic documents
- Query sets: 100, 1000 queries
- Document lengths: 50, 200, 1000 tokens

**Metrics:** indexing throughput (docs/s), query latency (p50/p95/p99), QPS

**Files to create:**
- `benchmarks/microbench/search_runner.py`

**Dependencies:** None (search already works via existing client)

---

### Story 2.4: Unified microbenchmark runner

**Complexity:** S

**What:** Orchestrates all microbenchmark sub-runners as a single CLI subcommand.

**Usage:**
```bash
python run.py microbench                          # all primitives
python run.py microbench --primitive kv json       # specific primitives
python run.py microbench --record-count 100000     # override parameters
```

**Files to create:**
- `benchmarks/microbench/runner.py` — `MicrobenchmarkSuite(BaseBenchmark)`

**Files to modify:**
- `benchmarks/__init__.py` — register `microbench`

**Dependencies:** 2.1-2.3

---

### Story 2.5: Cross-primitive macrobenchmarks

**Complexity:** L

**What:** End-to-end workflows exercising multiple primitives together.

**Workflows:**

| Workflow | What it measures |
|----------|-----------------|
| Ingest + auto-embed + search | Write N documents with auto-embedding enabled, then search. Measures write-to-searchable latency. |
| Cross-primitive search | Query that finds results across KV + JSON + Events simultaneously. Measures unified retrieval overhead. |
| Branch + modify + search + discard | Create branch, modify 10K entries, search on branch, delete branch. Measures COW overhead. |
| Auto-embed throughput | Writes/sec with embedding enabled vs disabled. Measures the auto-embed write amplification. |

**Files to create:**
- `benchmarks/macrobench/__init__.py`
- `benchmarks/macrobench/config.py`
- `benchmarks/macrobench/runner.py` — `MacrobenchmarkSuite(BaseBenchmark)`

**Files to modify:**
- `benchmarks/__init__.py` — register `macrobench`

**Dependencies:** 0.1

---

### Story 2.6: Baseline runners — Redis and SQLite

**Complexity:** L

**What:** Run the same microbenchmark workloads against external systems for fair
comparison. Redis is the KV baseline. SQLite is the embedded baseline (same
deployment model as Strata, no network overhead).

**BaselineRunner interface:**
```python
class BaselineRunner(ABC):
    def setup(self, config: dict) -> None: ...
    def run_workload(self, workload: str, params: dict) -> BenchmarkResult: ...
    def teardown(self) -> None: ...
```

**Baselines:**
- **Redis** — KV put/get/delete via redis-py. Docker container.
- **SQLite** — KV (table), JSON (json1 extension), FTS5 (BM25). In-process.
- **(Future) Elasticsearch** — BM25 + vector search. Docker container.
- **(Future) Qdrant** — Vector search. Docker container.

**Files to create:**
- `lib/baselines.py` — `BaselineRunner` ABC
- `benchmarks/polyglot/__init__.py`, `config.py`, `runner.py`
- `benchmarks/polyglot/redis_baseline.py`
- `benchmarks/polyglot/sqlite_baseline.py`
- `docker/docker-compose.baselines.yml` — Redis container config

**Files to modify:**
- `benchmarks/__init__.py` — register `polyglot`
- `requirements.txt` — add `redis>=5.0` (optional)

**Dependencies:** 2.4 (microbenchmark workloads to compare against)

---

### Story 2.7: Resource efficiency — Strata vs polyglot stack

**Complexity:** L

**What:** The headline result for Paper 1. Run the same workload on one Strata
instance vs. Redis + Elasticsearch + Qdrant running simultaneously. Measure total
CPU, memory, and disk usage across all processes.

**Target output:**
```
System              RAM (GB)    Disk (GB)    QPS
Redis+ES+Qdrant     12.4        8.2          2,100
Strata               1.8        2.1          3,400
```

**Files to create:**
- `benchmarks/polyglot/resource_comparison.py`

**Dependencies:** 1.6 (profiler), 2.6 (baseline runners)

---

### Story 2.8: Paper 1 experiment configs

**Complexity:** S

**What:** YAML configs for all Paper 1 experiments.

**Files to create:**
- `configs/paper1/kv_microbench.yaml`
- `configs/paper1/json_microbench.yaml`
- `configs/paper1/event_microbench.yaml`
- `configs/paper1/vector_microbench.yaml`
- `configs/paper1/graph_microbench.yaml`
- `configs/paper1/search_microbench.yaml`
- `configs/paper1/cross_primitive.yaml`
- `configs/paper1/resource_efficiency.yaml`

**Dependencies:** 1.1

---

## Epic 3: Paper 2 — Hybrid Search Benchmarks

Ablation study on BEIR, quality-latency Pareto analysis, external baselines.

### Story 3.1: BEIR ablation support

**Complexity:** M

**What:** Extend the BEIR benchmark to toggle individual pipeline components.

**Ablation matrix (each row adds one component):**

| Configuration | Components Enabled |
|---------------|-------------------|
| BM25 only | bm25 |
| BM25 + vectors | bm25, vectors |
| + expansion | bm25, vectors, expansion |
| + reranking | bm25, vectors, expansion, reranking |
| + position-aware blend | bm25, vectors, expansion, reranking, position_blend |
| + strong signal skip | all (full pipeline) |

**Current limitation:** `benchmarks/beir/retriever.py` has 3 hardcoded modes in
`_SEARCH_KWARGS` (keyword, hybrid, hybrid-llm). This needs to become a builder
that constructs search kwargs from individual component toggles.

**Files to create:**
- `benchmarks/beir/ablation.py` — `AblationConfig`, `run_ablation_suite()`

**Files to modify:**
- `benchmarks/beir/retriever.py` — replace fixed mode mapping with component builder
- `benchmarks/beir/runner.py` — add `--ablation`, `--expand`, `--rerank` flags

**Dependencies:** None

---

### Story 3.2: Quality-latency Pareto analysis

**Complexity:** M

**What:** Sweep search parameters and plot quality vs latency tradeoff.

**Parameter grid:**
- top_k: [10, 50, 100, 500]
- expansion: [off, on]
- reranking: [off, on]
- (Optionally: ef_search for HNSW, k for RRF)

**Output:** (nDCG@10, avg_latency_ms) pairs per configuration, with Pareto-optimal
points identified. Feeds into `lib/charts.py` Pareto curve generator.

**Files to create:**
- `benchmarks/beir/pareto.py` — `ParetoSweep` class

**Dependencies:** 1.5 (charts), 3.1 (component toggles)

---

### Story 3.3: External baselines — Elasticsearch ELSER, ColBERT

**Complexity:** L

**What:** Run BEIR evaluation against Elasticsearch ELSER and ColBERT v2 for fair
comparison in the results table.

**Elasticsearch ELSER:**
1. Start ES 8.x Docker container
2. Deploy ELSER model
3. Index BEIR corpus
4. Run queries
5. Evaluate with pytrec_eval

**ColBERT v2:**
- Run locally via the ColBERT Python library
- Index + search on same BEIR datasets

**DPR:** Use published numbers from the BEIR leaderboard (no need to re-run).

**Files to create:**
- `benchmarks/beir/baselines/__init__.py`
- `benchmarks/beir/baselines/elasticsearch_elser.py`
- `benchmarks/beir/baselines/colbert_baseline.py`

**Dependencies:** 2.6 (Docker baseline infrastructure)

---

### Story 3.4: Paper 2 experiment configs

**Complexity:** S

**Files to create:**
- `configs/paper2/ablation_nfcorpus.yaml`
- `configs/paper2/ablation_full_beir.yaml`
- `configs/paper2/pareto_sweep.yaml`
- `configs/paper2/baselines_comparison.yaml`

**Dependencies:** 1.1

---

## Epic 4: Paper 3 — Inference Benchmarks (Scaffold)

Scaffold only — define the structure, mark implementation as TODO.

### Story 4.1: Inference benchmark scaffold

**Complexity:** S

**What:** Create benchmark directory with config and stub runner.

**Comparison matrix to define in config:**

| Configuration | Search | Embedding | Expansion | Reranking | Generation |
|---------------|--------|-----------|-----------|-----------|------------|
| Strata in-process | in-process | in-process | in-process | in-process | in-process |
| Strata + Ollama | in-process | in-process | Ollama | Ollama | Ollama |
| Strata + cloud API | in-process | in-process | cloud | cloud | cloud |
| LangChain + Ollama + Chroma | Chroma | Ollama | Ollama | Ollama | Ollama |

**Metrics to define in config:**
- Per-stage latency breakdown (search_ms, embed_ms, expand_ms, rerank_ms, generate_ms)
- Token throughput (tokens/sec)
- Total pipeline latency (query in → answer out)
- Network hops count

**Files to create:**
- `benchmarks/inference_bench/__init__.py`
- `benchmarks/inference_bench/config.py`
- `benchmarks/inference_bench/runner.py` — `InferenceBenchmark(BaseBenchmark)` with
  `NotImplementedError` and clear description of what needs implementing

**Files to modify:**
- `benchmarks/__init__.py` — register

**Create:** `configs/paper3/latency_breakdown.yaml`, `configs/paper3/throughput.yaml`

**Dependencies:** 0.1 (inference namespace in client)

---

## Epic 5: Paper 4 — RAG Benchmarks

Complete the existing RAGAS scaffold and add standard RAG datasets.

### Story 5.1: Complete RAGAS implementation

**Complexity:** M

**What:** The existing `benchmarks/ragas_bench/runner.py` has retrieval working
(lines 87-127) but LLM generation and RAGAS evaluation are marked TODO.

**Remaining work:**
1. For each retrieval result, call `client.inference.generate()` (or external API)
   to produce an answer
2. Build a RAGAS `Dataset` from collected (question, answer, contexts, ground_truth)
3. Call `ragas.evaluate()` with faithfulness, answer_relevance, context_precision,
   context_recall
4. Pack results into `BenchmarkResult`

**Files to modify:**
- `benchmarks/ragas_bench/runner.py`

**Dependencies:** 0.1 (inference namespace for generate command)

---

### Story 5.2: Add standard RAG evaluation datasets

**Complexity:** M

**What:** Add dataset downloaders and configs for standard RAG benchmarks.

**Datasets:**
- Natural Questions (open-domain QA)
- TriviaQA
- FinanceBench (domain-specific financial QA)
- HotpotQA (multi-hop reasoning)

**Files to modify:**
- `benchmarks/ragas_bench/config.py` — add dataset definitions with URLs
- `benchmarks/ragas_bench/runner.py` — add `--dataset` argument, download logic

**Dependencies:** 5.1

---

### Story 5.3: RAG baseline runners (stub)

**Complexity:** S (stub only)

**What:** Stubs for comparing Strata RAG against LangChain and LlamaIndex.

**Files to create:**
- `benchmarks/ragas_bench/baselines/__init__.py`
- `benchmarks/ragas_bench/baselines/langchain_baseline.py` — stub with TODO
- `benchmarks/ragas_bench/baselines/llamaindex_baseline.py` — stub with TODO

**Dependencies:** None

---

### Story 5.4: Paper 4 experiment configs

**Complexity:** S

**Files to create:**
- `configs/paper4/rag_nq.yaml`
- `configs/paper4/rag_triviaqa.yaml`
- `configs/paper4/baselines_comparison.yaml`

**Dependencies:** 1.1

---

## Epic 6: Future Paper Stubs (Papers 5-12)

Minimal directory structure with config and stub runners. Each stub has a
`NotImplementedError` with a clear description of what the benchmark will measure
when implemented.

### Story 6.1: Create stub benchmark directories

**Complexity:** M (many files, each small and formulaic)

**Stubs to create:**

| Paper | Directory | What it will benchmark |
|-------|-----------|----------------------|
| 5: Segmented HNSW | `benchmarks/ann/pareto.py` (extend existing) | ef_search sweep, recall vs QPS Pareto curves, build time at 1M-10M scale |
| 6: Graph-Augmented Retrieval | `benchmarks/graph_retrieval/` | BEIR + knowledge graph, ablation of graph signals, structural vs statistical retrieval |
| 7: Recursive Queries | `benchmarks/recursive_query/` | Multi-hop QA (HotpotQA, MuSiQue), RLM-over-Strata vs RLM-over-text, iteration count analysis |
| 8: Agent-First Design | `benchmarks/agent_bench/` | Agent task completion rate, API calls per task, error recovery rate, with/without describe() and rich errors |
| 9: Event Projections | `benchmarks/event_projection/` | Projection throughput (events/sec with varying action counts), materialization consistency, replay performance |
| 10: State Machines | `benchmarks/state_machine/` | FSM validation overhead per state write, agent self-correction rate with/without actionable errors |
| 11: Auto-Embedding | `benchmarks/auto_embed/` | Write amplification, embedding throughput, query freshness (time from write to searchable), model size vs quality tradeoff |
| 12: COW Branching | `benchmarks/cow_branching/` | Branch creation overhead at varying DB sizes, write amplification on branches, merge performance |

**Each stub directory contains:**
- `__init__.py`
- `config.py` — dataset definitions, parameter defaults, metrics to collect
- `runner.py` — `XxxBenchmark(BaseBenchmark)` with `register_args()`, `validate()`,
  `download()` implemented, and `run()` raising `NotImplementedError` with a
  multi-line description of the implementation plan

**Files to modify:**
- `benchmarks/__init__.py` — register all stubs

**Dependencies:** None

---

### Story 6.2: Create stub experiment configs

**Complexity:** S

**Files to create:**
- `configs/paper5/ann_pareto.yaml`
- `configs/paper6/graph_retrieval.yaml`
- `configs/paper7/recursive_query.yaml`
- `configs/paper8/agent_tasks.yaml`
- `configs/paper9/event_projection.yaml`
- `configs/paper10/state_machine.yaml`
- `configs/paper11/auto_embed.yaml`
- `configs/paper12/cow_branching.yaml`

**Dependencies:** 1.1

---

## Implementation Sequencing

### Phase 0: Foundation (1 commit)

```
Story 0.1 — Extend StrataClient with JSON/Event/State/Inference namespaces
```

Unblocks: Epic 2 (microbenchmarks), Epic 4 (inference scaffold), Epic 5 (RAGAS completion)

### Phase 1: Core Infrastructure (5-7 commits)

```
Story 1.1 — Experiment config system ─────┐
Story 1.2 — Stats module (standalone) ────┤
                                          ├──► Story 1.3 — Multi-run executor
Story 1.6 — Hardware profiler ────────────┘

Story 1.2 ────────────────────────────────┬──► Story 1.4 — Comparison engine
                                          └──► Story 1.5 — Charts + LaTeX

Story 1.7 — Experiment runner script (ties 1.1-1.6 together)
```

### Phase 2: Paper 1 Benchmarks (4-5 commits)

```
Stories 2.1-2.3 — Microbenchmarks (KV, JSON, Event, State, BM25)
Story 2.4 — Unified microbenchmark suite
Story 2.5 — Cross-primitive macrobenchmarks
Story 2.6 — Baseline runners (Redis, SQLite)
Story 2.7 — Resource efficiency comparison
Story 2.8 — Paper 1 configs
```

### Phase 3: Paper 2 Benchmarks (3-4 commits)

```
Story 3.1 — BEIR ablation support
Story 3.2 — Pareto analysis
Story 3.3 — External baselines (Elasticsearch ELSER, ColBERT)
Story 3.4 — Paper 2 configs
```

### Phase 4: Paper 3+4 Scaffolds and All Stubs (2-3 commits)

```
Story 4.1 — Inference benchmark scaffold
Stories 5.1-5.4 — RAGAS completion + datasets + configs
Stories 6.1-6.2 — All future paper stubs and configs
```

**Total: ~24 stories across 7 epics, approximately 15-18 commits.**

---

## Verification Criteria

### After Phase 0:
- `python -m pytest tests/test_strata_client.py` passes with new namespace tests

### After Phase 1:
- `python scripts/run_experiment.py configs/examples/beir_keyword_baseline.yaml`
  runs 5 iterations of BEIR/nfcorpus keyword, saves results with mean ± std
- `python scripts/compare_experiments.py result1.json result2.json` produces
  delta + significance test output
- Existing commands (`python run.py beir --dataset nfcorpus`) still work unchanged

### After Phase 2:
- `python run.py microbench --primitive kv --record-count 10000` runs KV microbenchmarks
- `python run.py macrobench` runs cross-primitive workflows
- `python run.py polyglot --baseline redis sqlite` runs baseline comparisons

### After Phase 3:
- `python run.py beir --dataset nfcorpus --ablation` produces full ablation table
- `python scripts/generate_paper_tables.py --paper 2 results/` produces LaTeX + Pareto chart

### After Phase 4:
- All stub benchmarks importable:
  `python -c "from benchmarks import REGISTRY; print(list(REGISTRY.keys()))"`
  lists all registered suites including stubs
- `python run.py ragas --corpus data.jsonl --questions qa.jsonl` completes end-to-end

---

## Architectural Decisions

1. **Coexistence, not replacement.** YAML configs are an additive layer. Existing
   CLI commands continue to work unchanged.

2. **Stateless benchmarks.** Runner classes stay stateless. The executor manages
   multi-run loops, seed management, and profiling externally.

3. **Optional dependencies.** Every external system (Redis, Elasticsearch, scipy,
   matplotlib, ragas) is an optional import, following the lazy-import pattern in
   `benchmarks/__init__.py`.

4. **Schema backward compatibility.** The `schema_version` field enables the
   report loader to handle both v1 and v2 formats. The 30+ existing result files
   in `results/` must continue to load.

5. **Docker for external baselines.** Redis, Elasticsearch, Neo4j managed via
   Docker Compose for reproducibility. Benchmarks fail gracefully with a clear
   message if Docker is unavailable.
