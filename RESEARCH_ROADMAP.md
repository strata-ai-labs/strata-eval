# Strata Research Roadmap

This document maps the academic research landscape for Strata — potential papers,
benchmarking methodology, evaluation strategies, and how the body of work compounds
over time. The goal is to build a rigorous, peer-reviewable foundation that gives
Strata serious academic credibility and creates a moat that is expensive to replicate.

---

## Table of Contents

1. [Strategic Overview](#strategic-overview)
2. [What Makes Strata Novel](#what-makes-strata-novel)
3. [Paper Roadmap](#paper-roadmap)
   - [Paper 1: Architecture](#paper-1-strata-the-multi-primitive-embedded-database)
   - [Paper 2: Hybrid Search](#paper-2-typed-hybrid-search-with-position-aware-blending)
   - [Paper 3: In-Process Inference](#paper-3-inference-embedded-in-the-database)
   - [Paper 4: RAG Quality](#paper-4-native-rag-with-zero-serialization-overhead)
   - [Paper 5: Vector Index](#paper-5-segmented-hnsw)
   - [Paper 6: Graph-Augmented Retrieval](#paper-6-graph-augmented-hybrid-retrieval)
   - [Paper 7: Recursive Query Execution](#paper-7-recursive-query-execution-over-indexed-primitives)
   - [Paper 8: Agent-First Database Design](#paper-8-agent-first-database-design)
   - [Paper 9: Event Projections](#paper-9-event-sourced-multi-primitive-materialization)
   - [Paper 10: Graph-Validated State Machines](#paper-10-database-native-state-machines-via-graph-ontology)
   - [Paper 11: Auto-Embedding](#paper-11-auto-embedding-as-a-database-primitive)
   - [Paper 12: Copy-on-Write Branching](#paper-12-copy-on-write-branching-for-multi-primitive-databases)
4. [Benchmarking Methodology](#benchmarking-methodology)
   - [How Research Teams Benchmark](#how-research-teams-benchmark)
   - [Raw Benchmarking](#raw-benchmarking)
   - [Comparative Benchmarking](#comparative-benchmarking)
   - [Statistical Rigor](#statistical-rigor)
5. [Benchmark Infrastructure](#benchmark-infrastructure)
   - [Experiment Configuration](#experiment-configuration)
   - [Results Storage](#results-storage)
   - [Comparison Engine](#comparison-engine)
6. [Sequencing and Dependencies](#sequencing-and-dependencies)
7. [References](#references)

---

## Strategic Overview

The research strategy is built on compounding returns:

- **Paper 1** (architecture) forces construction of the benchmarking harness — experiment
  configs, results store, comparison engine, baseline runners. This is 80% of the
  infrastructure work.
- **Papers 2-4** reuse the harness, adding one new benchmark dimension each time
  (retrieval quality, RAG quality, inference latency). Each takes a fraction of the
  effort of Paper 1.
- **Papers 5+** are incremental. By then, every new Strata feature slots into the
  existing evaluation framework.

The moat is not any single paper. It is the accumulated body of work where each
result cross-references the last, run on the same harness, against the same baselines,
with the same rigor. That is what makes it hard to dismiss and expensive to replicate.

Every subsequent paper opens with: *"We build on Strata [1], a multi-primitive
embedded database that..."* If that reference does not exist, every paper has to
spend two pages explaining what Strata is before it can get to its actual contribution.
The architecture paper is the foundation.

---

## What Makes Strata Novel

No other single embedded system has all of the following in one process:

| Primitive / Capability | Standalone Equivalent | In Strata |
|---|---|---|
| KV with CAS | Redis, RocksDB | Yes |
| JSON documents | MongoDB, SQLite JSON | Yes |
| Append-only hash-chained events | Kafka, EventStoreDB | Yes |
| State cells with CAS | etcd, Consul | Yes |
| Property graph with ontology | Neo4j, DGraph | Yes |
| Dense vectors (Segmented HNSW) | Pinecone, Qdrant, FAISS | Yes |
| BM25 inverted index | Elasticsearch, Tantivy | Yes |
| Local LLM inference (llama.cpp) | Ollama, vLLM | Yes |
| Cloud LLM inference (3 providers) | OpenAI API | Yes |
| Copy-on-write branching | Git (but for data) | Yes |
| Auto-embedding on write | Nothing does this natively | Yes |
| Cross-primitive search | Nothing does this | Yes |

Each primitive alone is not novel. **The convergence is the contribution.**
Convergence creates emergent capabilities — cross-primitive search, zero-serialization
RAG, branched experimentation, auto-embedding — that do not exist when these are
separate systems.

---

## Paper Roadmap

### Paper 1: Strata — The Multi-Primitive Embedded Database

**Venue:** VLDB, OSDI, or SOSP (systems)

**Thesis:** Running KV, JSON, events, state, graph, vectors, and LLM inference in a
single embedded process achieves competitive performance on each primitive individually
while enabling cross-primitive capabilities that are impossible when these are separate
systems.

**Why it comes first:** Every other paper cites this one. It establishes what Strata
is, proves the architecture is sound, and builds the benchmarking harness that all
subsequent papers reuse.

**Evaluation structure:**

1. **Microbenchmarks** — Each primitive vs. its standalone equivalent
   - KV: ops/sec for get, set, CAS at varying value sizes (YCSB workloads A-F)
     vs. Redis, RocksDB, SQLite
   - JSON: document insert/query throughput at varying collection sizes vs. MongoDB,
     SQLite JSON1
   - Events: append throughput, hash-chain verification speed vs. Kafka (embedded
     mode), EventStoreDB
   - Vectors: build time, query latency, recall@10 at varying dataset sizes
     (ann-benchmarks protocol) vs. FAISS, hnswlib
   - Graph: traversal latency (BFS, k-hop neighbors) at varying graph sizes vs.
     Neo4j embedded, SQLite with recursive CTEs
   - BM25: indexing throughput, query latency vs. Tantivy, SQLite FTS5

2. **Macrobenchmarks** — End-to-end workflows exercising multiple primitives
   - "Ingest 1M documents, auto-embed, then search" — full write+embed+search pipeline
   - "Branch, modify 10K entries, search on branch, merge" — COW overhead
   - "Cross-primitive search across KV+JSON+Events" — unified retrieval

3. **Resource efficiency** — Memory, disk, CPU footprint
   - One Strata instance vs. the polyglot stack (Redis + Elasticsearch + Neo4j +
     Kafka + Qdrant) running the same workload
   - Show the "convergence tax" — ideally small or negative due to eliminated
     serialization overhead

4. **Capability demonstration** — Things only Strata can do
   - Auto-embed on write: write a KV entry, immediately search it semantically
   - Branch all primitives atomically: experiment on a branch, discard or merge
   - Cross-primitive search: one query finds results across KV, JSON, events, state

**Baselines:**
- SQLite (embedded relational)
- RocksDB (embedded KV)
- DuckDB (embedded analytical)
- LanceDB (embedded vectors)
- The polyglot stack: Redis + Elasticsearch + Neo4j + Kafka + Qdrant

**Key result to target:** Strata is within 80-90% of each standalone system on its
home workload, while using 3-5x less total resources than running them all, and
enabling capabilities none of them can replicate.

---

### Paper 2: Typed Hybrid Search with Position-Aware Blending

**Venue:** SIGIR, ECIR, or CIKM (information retrieval)

**Thesis:** Routing typed query expansions (lex, vec, hyde) to their matching index
(BM25 vs. HNSW) combined with position-aware blending of RRF and reranker signals
outperforms both fixed-weight fusion and untyped expansion.

**Novel contributions:**
- Typed expansion routing: lex expansions go to BM25 only, vec/hyde go to HNSW only.
  No existing system does this routing.
- Position-aware blending: RRF and reranker signals weighted differently by rank
  position (top-3: trust RRF more; rank 11+: trust reranker more). Novel formula.
- Strong signal detection: BM25 probe that skips the expensive pipeline when
  confidence is high. Latency optimization with quality preservation analysis.

**Evaluation:**
- BEIR benchmark suite (18 datasets), nDCG@10 headline metric
- Ablation study: BM25 only → +vectors → +RRF → +typed expansion → +reranking
  (fixed blend) → +position-aware blend → +strong signal skip
- Each row adds one component, showing marginal contribution and latency cost
- Quality-latency Pareto chart

**Baselines:**
- BM25 (Lucene/Tantivy)
- DPR (dense passage retrieval)
- ColBERT v2 (late interaction)
- Elasticsearch ELSER (learned sparse)
- Cohere Rerank + vector search (cloud RAG baseline)
- qmd (local hybrid, architecturally similar)

**Existing infrastructure:** BEIR benchmarks already partially implemented in
strata-eval with keyword and hybrid results across 15+ datasets.

---

### Paper 3: Inference Embedded in the Database

**Venue:** VLDB, EuroSys, or MLSys (systems)

**Thesis:** Co-locating LLM inference (via llama.cpp) inside the database process
eliminates serialization and network overhead for search-adjacent inference tasks
(query expansion, reranking, embedding), yielding measurable latency and throughput
improvements over the standard architecture of separate inference servers.

**Evaluation:**
- Latency breakdown: search → embed → expand → rerank → generate pipeline
  - In-process (Strata) vs. localhost inference server vs. remote API
  - Measure serialization overhead, network round-trips, context switching
- Throughput: queries/sec under concurrent load
- Token throughput: tokens/sec for generation, embedding operations
- Memory efficiency: shared memory between database and inference vs. separate heaps

**Baselines:**
- Strata with external Ollama (same machine, network hop)
- Strata with cloud API (OpenAI/Anthropic)
- LangChain + Ollama + ChromaDB (standard stack)
- vLLM + Qdrant (optimized separate services)

---

### Paper 4: Native RAG with Zero-Serialization Overhead

**Venue:** ACL, EMNLP, or NeurIPS (AI/ML)

**Thesis:** A single-call RAG primitive (db.ask()) that runs search and generation
in the same process achieves comparable answer quality to framework-based RAG
(LangChain, LlamaIndex) with significantly lower latency and no integration code.

**Evaluation:**
- RAG quality metrics:
  - Answer correctness (exact match, F1 against gold answers)
  - Faithfulness (does the answer only use retrieved context? LLM-as-judge)
  - Citation precision (are source references accurate?)
  - Answer relevance
- End-to-end latency: query in → answer out
  - Breakdown: search time, context assembly, generation time
- Datasets: Natural Questions, TriviaQA, FinanceBench, HotpotQA

**Baselines:**
- Closed-book LLM (no retrieval)
- Naive RAG: vector search + GPT-4
- LangChain + Pinecone + GPT-4
- LlamaIndex + ChromaDB + GPT-4
- Strata db.ask() with local model
- Strata db.ask() with cloud model

---

### Paper 5: Segmented HNSW

**Venue:** VLDB, NeurIPS, or ICML (depending on framing)

**Thesis:** [Depends on the specific novelty of Strata's HNSW implementation —
segmentation strategy, incremental build properties, or memory layout.]

**Evaluation:**
- ann-benchmarks protocol
- Datasets: GloVe-100, SIFT-128, GloVe-200, deep-image-96, fashion-mnist
- Metric: Recall@10 vs. queries/second (Pareto curve)
- Build time and memory footprint at varying dataset sizes (10K to 10M vectors)

**Baselines:**
- FAISS-HNSW
- hnswlib
- Annoy
- ScaNN
- Vamana/DiskANN

---

### Paper 6: Graph-Augmented Hybrid Retrieval

**Venue:** SIGIR, KDD, or NeurIPS (IR/AI)

**Thesis:** Fusing BM25 + dense vectors + graph proximity + ontology-guided query
decomposition in a single engine outperforms each signal alone. The graph provides
structural reasoning that statistical retrieval cannot replicate.

**Novel contributions:**
- Graph proximity boost after RRF fusion (Mode B from issue #1270)
- Ontology-guided query understanding (LLM reads ontology, decomposes query into
  typed retrieval plan)
- Graph-context reranking (enriching reranker snippets with graph neighborhood)
- Three-signal blending (RRF + reranker + graph proximity, position-aware)

**Evaluation:**
- BEIR + domain-specific benchmarks with knowledge graphs
  (build a medical/financial graph with ontology)
- Ablation: BM25+vectors → +graph boost → +ontology expansion → +graph-context
  reranking → +three-signal blending
- Compare against: Elasticsearch, ColBERT, Microsoft GraphRAG, standard vector RAG

**Key experiment:** Show that graph augmentation helps most on queries requiring
structural reasoning (multi-hop, entity-centric) and least on simple keyword queries.
Characterize when the graph signal adds value.

**Depends on:** Paper 1 (architecture), Paper 2 (hybrid search baseline)

---

### Paper 7: Recursive Query Execution Over Indexed Primitives

**Venue:** NeurIPS, ICML, or ACL (AI/ML)

**Thesis:** RLMs (Recursive Language Models) operating over indexed database
primitives (BM25, vectors, graph, ontology) outperform both single-shot RAG and
RLMs operating over raw text, because the model can efficiently navigate structured
data instead of grepping strings.

**Novel contributions:**
- Replacing RLM's Python REPL (string slicing, regex) with indexed search + graph
  traversal changes the computational complexity of what the model can explore
- Three-tier implementation: iterative search (no code execution) → sandboxed code
  execution over Strata API → branch-scoped recursive exploration

**Evaluation:**
- Long-context QA benchmarks (from the original RLM paper)
- Multi-hop QA: HotpotQA, MuSiQue
- Compare: vanilla RAG, RLM-over-text (reference implementation), RLM-over-Strata
- Key experiment: show that an 8B model with Strata's tools matches or exceeds a
  70B model with raw-text RLM

**Depends on:** Paper 1 (architecture), Paper 6 (graph-augmented search)

---

### Paper 8: Agent-First Database Design

**Venue:** CHI, UIST, or NeurIPS (HCI/AI)

**Thesis:** Systematic design principles for LLM-consumable database APIs —
describe() introspection, actionable errors, explain mode, progressive capability
disclosure — measurably improve agent task completion rates and reduce error recovery
cycles.

**Novel contributions:**
- Formal study of how API design choices affect LLM agent performance
- Principles: discovery over documentation, errors as instructions, tool schemas as
  documentation, consistent patterns, minimal bootstrapping, graceful degradation
- Quantitative measurement of agent self-correction rates

**Evaluation:**
- A/B test agent performance:
  - With vs. without describe() introspection
  - Rich vs. generic error messages
  - With vs. without explain mode
  - With vs. without fuzzy matching on typos
- Metrics: task completion rate, number of API calls to complete task, error
  recovery rate, time to first successful query
- Agent baselines: Claude, GPT-4, open-source models (test generalization across
  LLM capabilities)

**Depends on:** Paper 1 (architecture), implementation of #1274

---

### Paper 9: Event-Sourced Multi-Primitive Materialization

**Venue:** VLDB or SIGMOD (databases)

**Thesis:** When event projections automatically materialize events into KV, JSON,
graph, and state primitives — all of which are auto-embedded and searchable — the
database becomes a self-organizing knowledge base that eliminates the need for
separate event streaming, projection services, and materialized view management.

**Novel contributions:**
- A single event produces multi-primitive artifacts (KV + JSON + Graph + State)
- All artifacts are auto-embedded and searchable via hybrid search
- Projection replay from immutable event log guarantees consistency
- No existing system combines event sourcing with multi-modal materialization +
  auto-embedding

**Evaluation:**
- Projection throughput: events/sec with varying numbers of projection actions
- Consistency: verify materialized views match event log under concurrent writes
- Query quality: db.ask() answers when querying projected views vs. raw events
  vs. both
- Compare against: Kafka + Flink + PostgreSQL (standard event sourcing stack),
  EventStoreDB + custom projections

**Depends on:** Paper 1 (architecture), implementation of RFC: Event Projections

---

### Paper 10: Database-Native State Machines via Graph Ontology

**Venue:** VLDB, SIGMOD, or ICSE (databases/software engineering)

**Thesis:** When a database exposes valid state transitions as a queryable graph
ontology with actionable error messages, LLM agents achieve higher task completion
rates on stateful workflows with fewer invalid operations.

**Novel contributions:**
- Using property graph ontologies (object types = states, link types = transitions)
  to define and validate state machines
- State machines become first-class, queryable, graph-defined entities
- Rich error messages include valid transitions and natural-language suggestions
- Agents can introspect the FSM via ontology_summary() and
  state_valid_transitions()

**Evaluation:**
- Agent success rate on stateful tasks (pipeline management, workflow orchestration)
  with FSM validation + rich errors vs. without
- Self-correction rate: how often agents recover from invalid transitions using the
  error message alone
- Compare: application-layer FSM (invisible to agent), database constraint-based
  FSM (PostgreSQL triggers), Strata graph-validated FSM

**Depends on:** Paper 1 (architecture), Paper 8 (agent-first design), implementation
of RFC: Graph-Validated State Transitions

---

### Paper 11: Auto-Embedding as a Database Primitive

**Venue:** VLDB or SIGIR (databases/IR)

**Thesis:** Write-time automatic embedding — where every KV set, JSON insert, and
event append is immediately vector-indexed — changes the economics of vector search
by trading write amplification for guaranteed query freshness, eliminating batch
embedding pipelines entirely.

**Evaluation:**
- Write amplification: throughput with auto-embed on vs. off
- Query freshness: time from write to searchable (Strata: immediate vs. batch
  pipelines: minutes to hours)
- Retrieval quality: does inline embedding with a small local model match quality
  of batch embedding with a large cloud model?
- Cost analysis: total compute for write-time embed vs. periodic batch re-embed

**Baselines:**
- Batch embedding pipeline (standard approach)
- Streaming embedding (Kafka + embedding service)
- Strata auto-embed (synchronous, in-process)

---

### Paper 12: Copy-on-Write Branching for Multi-Primitive Databases

**Venue:** VLDB or CIDR (databases)

**Thesis:** Git-like copy-on-write branching applied to a multi-primitive database
(KV, JSON, events, state, graph, vectors) enables low-cost experimentation — A/B
testing search configurations, trying different ontology schemas, comparing
embedding models — without affecting production data.

**Evaluation:**
- Branch creation overhead at varying database sizes
- Write amplification on branches (COW efficiency)
- Merge performance and conflict resolution
- Case studies: parameter tuning, ontology A/B testing, embedding model comparison
- Compare: database snapshots (PostgreSQL), Redis BGSAVE, filesystem snapshots (ZFS)

---

## Benchmarking Methodology

### How Research Teams Benchmark

Research evaluation has two fundamental questions:

1. **"Does it work?"** — Raw/intrinsic benchmarking. Absolute performance on
   standardized tasks.
2. **"Does it work better?"** — Comparative benchmarking. Performance relative to
   baselines and state-of-the-art.

Different research communities have different evaluation cultures:

| Community | Cares About | Standard Benchmarks |
|---|---|---|
| Systems (VLDB, OSDI) | Throughput, latency, scalability, resource efficiency | YCSB, TPC, ann-benchmarks |
| IR (SIGIR, ECIR) | Retrieval quality on standardized datasets | BEIR, MTEB, MS MARCO |
| AI/ML (NeurIPS, ACL) | Task accuracy, comparison to SotA models | NQ, TriviaQA, HotpotQA, MMLU |
| HCI (CHI, UIST) | User/agent task completion, usability | Custom task suites with user studies |

### Raw Benchmarking

#### Standardized Benchmark Suites

Never invent your own evaluation dataset if a community-standard one exists.
Reviewers immediately distrust results on custom datasets.

- **BEIR** — 18 retrieval datasets spanning diverse domains. nDCG@10 is the
  headline metric. Gold standard for IR papers.
- **MTEB** — Massive text embedding benchmark. For evaluating embedding quality.
- **ann-benchmarks** — Standard protocol for approximate nearest neighbor indexes.
  Recall@K vs. QPS Pareto curves.
- **YCSB** — Yahoo Cloud Serving Benchmark. Standard for KV store performance
  (workloads A-F covering different read/write ratios).
- **Domain-specific:** FinanceBench (financial QA), MedQA (medical),
  HotpotQA/MuSiQue (multi-hop reasoning).

#### Metrics — Never Report Just One

**Retrieval quality:**

| Metric | What It Measures |
|---|---|
| nDCG@10 | Ranking quality (graded relevance) |
| MAP | Average precision across recall levels |
| MRR | How quickly you find the first relevant result |
| Recall@100 | Coverage — did you find all relevant docs? |
| Precision@1 | Is the top result correct? |

**System performance:**

| Metric | What It Measures |
|---|---|
| p50/p95/p99 latency | Tail latency matters, not just average |
| Throughput (QPS) | Queries per second at saturation |
| Index build time | Write-path cost |
| Memory footprint | Practical deployment constraint |
| Index size on disk | Storage cost |

**RAG quality:**

| Metric | What It Measures |
|---|---|
| Answer accuracy / F1 | Is the answer correct? |
| Faithfulness | Does the answer only use retrieved context? |
| Citation precision | Are source citations accurate? |
| Answer relevance | Does it actually address the question? |

#### Experimental Protocol

This separates papers that get accepted from those that get desk-rejected:

1. **Pre-registered experiments.** Decide what you are measuring and why before
   running anything. Write the evaluation section outline first. This prevents
   cherry-picking results.

2. **Controlled variables.** Change exactly one thing per experiment. If testing
   graph-augmented retrieval, the embedding model, chunk size, BM25 parameters,
   and everything else must be identical between control and treatment.

3. **Hardware specification.** Every paper includes a hardware table:
   ```
   CPU: AMD EPYC 7763, 64 cores
   RAM: 256 GB DDR4
   GPU: NVIDIA A100 80GB (if applicable)
   Storage: NVMe SSD
   OS: Ubuntu 22.04
   ```

4. **Multiple runs with statistics.** Never report a single number. Run each
   experiment 3-5 times minimum. Report mean +/- standard deviation, or median
   with interquartile range.

5. **Warm-up runs.** Discard the first 1-2 runs to avoid cold cache effects.

6. **Fixed seeds.** Set random seeds for reproducibility. Document them.

### Comparative Benchmarking

#### Baseline Selection

Three categories of baselines are required:

**Simple baselines (sanity check):**
- BM25 only (no neural anything)
- TF-IDF + cosine similarity
- Random retrieval (establishes floor)

These prove your problem is not trivially solvable. If BM25 alone gets 95% nDCG,
a fancy graph-augmented pipeline needs to justify its complexity.

**Strong baselines (state of the art):**
- The current best system on the benchmark
- The system most similar to yours architecturally
- The system from the most cited recent paper in the area

**Ablation baselines (your system minus components):**

This is the most important category. Remove one component at a time from the full
system and measure the degradation:

```
Full system:     BM25 + vectors + expansion + reranking + graph boost
Ablation 1:      BM25 + vectors + expansion + reranking              (no graph)
Ablation 2:      BM25 + vectors + expansion              + graph     (no reranking)
Ablation 3:      BM25 + vectors             + reranking  + graph     (no expansion)
Ablation 4:      BM25           + expansion + reranking  + graph     (no vectors)
Ablation 5:             vectors + expansion + reranking  + graph     (no BM25)
```

This proves each component contributes. If removing a component does not change
results, you cannot claim it matters.

#### Fair Comparison Practices

- **Use the competitor's best configuration.** Comparing your tuned system against
  an untuned baseline is the fastest way to get rejected. Use published optimal
  hyperparameters, or tune baselines with the same effort as your own system.

- **Same data, same splits.** Everyone uses the same train/dev/test split. For
  BEIR, these are fixed.

- **Same compute budget (when relevant).** If your system uses 10x the compute,
  that is a critical detail. Consider a quality-per-dollar analysis.

- **Report losses, not just wins.** Every system has datasets where it
  underperforms. A credible paper shows the full matrix and explains why certain
  datasets favor the baseline.

#### The Results Table

Centerpiece of every evaluation section:

```
Table 1: nDCG@10 on BEIR benchmark (mean of 3 runs)

Dataset      BM25   DPR    ColBERT  ELSER  Strata  Strata+G
-------      ----   ---    -------  -----  ------  --------
MS MARCO     0.228  0.311  0.344    0.338  0.341   0.352*
NFCorpus     0.325  0.189  0.338    0.341  0.346   0.371*
SciFact      0.665  0.318  0.671    0.688  0.692   0.701
FiQA         0.236  0.112  0.356    0.348  0.361   0.359
...
Average      0.381  0.247  0.412    0.419  0.425   0.441*

* = statistically significant improvement (p < 0.05, paired t-test)
Strata+G = Strata with graph-augmented retrieval
```

Every cell is filled. Losses are shown. Significance is marked.

### Statistical Rigor

- Run each experiment 3-5 times minimum
- Report mean +/- standard deviation
- Use paired t-test or Wilcoxon signed-rank test for significance
- Report effect size (Cohen's d) when claiming improvements
- p < 0.05 threshold, but report exact p-values
- For multiple comparisons, apply Bonferroni or Holm-Bonferroni correction

---

## Benchmark Infrastructure

The harness that powers all papers. Build once, reuse everywhere.

### Experiment Configuration

Each experiment is a declarative config, not a script with hardcoded parameters:

```yaml
experiment:
  name: "graph-boost-ablation-v3"
  date: "2026-02-27"
  hypothesis: "Graph proximity boost improves nDCG@10 on entity-centric datasets"

dataset:
  name: "nfcorpus"
  source: "beir"
  split: "test"

system:
  base: "strata"
  version: "0.6.0"
  commit: "abc123"

  components:
    bm25: true
    vectors: true
    expansion: true
    reranking: true
    graph_boost: true          # the variable being tested
    graph_weight: 0.3

  fixed_params:
    bm25_k1: 1.2
    bm25_b: 0.75
    embed_model: "nomic-embed-text-v1.5"
    expansion_model: "qwen3:8b"
    rerank_model: "qwen3:8b"
    top_k: 10
    rrf_k: 60

hardware:
  cpu: "AMD Ryzen 9 7950X"
  ram: "64GB"
  gpu: "none"
  storage: "NVMe"

runs: 5
seed_base: 42
warmup_runs: 1
```

### Results Storage

Every run produces a structured result artifact:

```json
{
  "experiment": "graph-boost-ablation-v3",
  "dataset": "nfcorpus",
  "run": 3,
  "seed": 44,
  "timestamp": "2026-02-27T14:23:01Z",
  "git_commit": "abc123",
  "metrics": {
    "ndcg@10": 0.371,
    "map": 0.298,
    "mrr": 0.412,
    "recall@100": 0.634,
    "precision@1": 0.389
  },
  "latency": {
    "p50_ms": 12.3,
    "p95_ms": 45.1,
    "p99_ms": 89.7
  },
  "system": {
    "version": "0.6.0",
    "components_enabled": ["bm25", "vectors", "expansion", "reranking", "graph_boost"]
  },
  "hardware": {
    "cpu": "AMD Ryzen 9 7950X",
    "ram_gb": 64,
    "gpu": "none"
  }
}
```

### Comparison Engine

After all runs complete, automated pipeline:

1. Aggregate results across runs (mean, std, confidence intervals)
2. Run significance tests (paired t-test per dataset, Bonferroni correction)
3. Generate LaTeX results tables
4. Generate plots (bar charts with error bars, Pareto curves, radar charts)
5. Flag regressions against previous experiment runs
6. Produce a reproducibility package (configs, seeds, model versions, dataset
   download scripts)

---

## Sequencing and Dependencies

```
Paper 1: Architecture ─────────────────────────────────┐
  │                                                     │
  ├──► Paper 2: Hybrid Search                           │
  │      │                                              │
  │      ├──► Paper 6: Graph-Augmented Retrieval        │
  │      │      │                                       │
  │      │      └──► Paper 7: Recursive Queries         │
  │      │                                              │
  │      └──► Paper 4: Native RAG                       │
  │                                                     │
  ├──► Paper 3: In-Process Inference                    │
  │                                                     │
  ├──► Paper 5: Segmented HNSW                          │
  │                                                     │
  ├──► Paper 11: Auto-Embedding                         │
  │                                                     │
  ├──► Paper 12: COW Branching                          │
  │                                                     │
  └──► Paper 8: Agent-First Design ─────────────────────┤
         │                                              │
         └──► Paper 10: Graph-Validated State Machines  │
                                                        │
Paper 9: Event Projections ◄────────────────────────────┘
```

**Phase 1 (now):** Paper 1 — Architecture. Build the harness.

**Phase 2 (after harness):** Papers 2, 3, 5, 11, 12 — independent papers that
each evaluate one aspect of the existing system. Can be written in parallel.

**Phase 3 (after roadmap features ship):** Papers 4, 6, 7, 8, 9, 10 — depend
on features from issues #1269-#1274 and the RFCs.

---

## References

### Strata Issues
- [#1269](https://github.com/strata-ai-labs/strata-core/issues/1269) — Hybrid search pipeline
- [#1270](https://github.com/strata-ai-labs/strata-core/issues/1270) — Graph-Augmented Hybrid Search
- [#1272](https://github.com/strata-ai-labs/strata-core/issues/1272) — Recursive Query Execution
- [#1273](https://github.com/strata-ai-labs/strata-core/issues/1273) — Native RAG: db.ask()
- [#1274](https://github.com/strata-ai-labs/strata-core/issues/1274) — Agent-First API Design

### Strata RFCs
- RFC: Event Projections to Other Primitives
- RFC: Graph-Validated State Transitions
- RFC: State-Event Audit Trail

### Benchmarking Standards
- [BEIR](https://github.com/beir-cellar/beir) — Heterogeneous benchmark for information retrieval (Thakur et al., 2021)
- [MTEB](https://github.com/embeddings-benchmark/mteb) — Massive Text Embedding Benchmark
- [ann-benchmarks](https://ann-benchmarks.com/) — Benchmarking approximate nearest neighbor algorithms
- [YCSB](https://github.com/brianfrankcooper/YCSB) — Yahoo Cloud Serving Benchmark (Cooper et al., 2010)

### Related Academic Work
- [GraphRAG](https://arxiv.org/abs/2404.16130) — Microsoft, knowledge graph enhanced RAG
- [RLM](https://arxiv.org/abs/2512.24601) — Recursive Language Models (Zhang, Kraska, Khattab, MIT)
- [PageIndex](https://github.com/VectifyAI/PageIndex) — Vectorless reasoning-based RAG
- [qmd](https://github.com/tobi/qmd) — Local hybrid search with typed expansion
- [Elasticsearch ELSER](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search-elser.html) — Learned sparse encoder
- Reciprocal Rank Fusion — Cormack, Clarke, Buettcher (2009)
