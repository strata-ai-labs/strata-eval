# strata-eval

Comprehensive benchmarks for [StrataDB](https://github.com/strata-ai-labs/strata). Covers information retrieval, key-value workloads, vector search, graph algorithms, and RAG evaluation.

## Benchmark Suites

| Suite | What it tests | Metrics | Status |
|-------|---------------|---------|--------|
| **BEIR** | IR retrieval quality (15 datasets) | nDCG@10, Recall@100, MAP, MRR | Implemented |
| **YCSB** | Key-value workloads A-F | Throughput (ops/s), p50/p95/p99 latency | Implemented |
| **ANN** | Vector search quality + speed | Recall@k, QPS, build time | Implemented |
| **Graphalytics** | Graph algorithms (LDBC) | EVPS, correctness validation | Implemented |
| **LoCoMo** | Long-context conversational memory | F1, ROUGE-L | Scaffold (needs LLM) |
| **LongMemEval** | Long-term memory (5 abilities) | Accuracy, Recall@k | Stub (needs LLM) |
| **RAGAS** | RAG pipeline quality | Faithfulness, relevance, precision, recall | Scaffold (needs LLM) |
| **GraphRAG-Bench** | Graph-based RAG | Accuracy, reasoning quality | Stub (needs LLM) |

**Implemented** = fully functional. **Scaffold** = retrieval pipeline works, LLM evaluation marked TODO. **Stub** = structure defined, full implementation needed.

## Setup

```bash
pip install stratadb
pip install -r requirements.txt

# BEIR benchmarks (optional)
pip install beir sentence-transformers pytrec_eval

# ANN benchmarks (optional)
pip install h5py
```

## Quick Start

```bash
# BEIR — information retrieval
python run.py beir --dataset nfcorpus --mode hybrid

# YCSB — key-value workloads
python run.py ycsb --workload a --records 100000 --ops 100000

# ANN — vector search
python run.py download --bench ann --dataset glove-25-angular
python run.py ann --dataset glove-25-angular

# LDBC Graphalytics — graph algorithms
python run.py graphalytics --algorithm bfs --dataset example-directed --runs 10

# Generate report from all results
python run.py report
python run.py report --format latex
```

## BEIR Benchmarks

15 datasets from 3.6K to 8.8M documents. Evaluates BM25 keyword search and hybrid (BM25 + MiniLM vectors + RRF fusion).

```bash
python run.py beir --dataset nfcorpus --mode hybrid
python run.py beir --dataset scifact --mode keyword
python run.py beir --dataset nfcorpus --mode hybrid --k 10 100 1000
```

| Dataset | Docs | Queries | Domain |
|---------|------|---------|--------|
| `nfcorpus` | 3.6K | 323 | Nutrition/medical |
| `scifact` | 5K | 300 | Scientific claims |
| `arguana` | 8.7K | 1,406 | Argument retrieval |
| `scidocs` | 25K | 1,000 | Scientific papers |
| `trec-covid` | 171K | 50 | COVID-19 literature |
| + 10 more | up to 8.8M | | Various |

## YCSB Benchmarks

Standard Yahoo Cloud Serving Benchmark workloads measuring throughput and latency. Single-threaded sequential execution.

```bash
python run.py ycsb --workload a b c d e f
python run.py ycsb --workload a --records 1000000 --distribution zipfian
```

| Workload | Description | Read/Write |
|----------|-------------|------------|
| A | Update heavy | 50/50 |
| B | Read mostly | 95/5 |
| C | Read only | 100/0 |
| D | Read latest | 95/5 insert |
| E | Short scans | 95 scan/5 insert |
| F | Read-modify-write | 50/50 |

## ANN Benchmarks

Vector search recall vs. QPS using standard ann-benchmarks.com datasets. Reports a single operating point at Strata's default search parameters (single-threaded sequential queries).

```bash
python run.py download --bench ann --dataset sift-128-euclidean
python run.py ann --dataset sift-128-euclidean
```

| Dataset | Dimensions | Vectors | Metric |
|---------|-----------|---------|--------|
| `sift-128-euclidean` | 128 | 1M | Euclidean |
| `glove-100-angular` | 100 | 1.18M | Cosine |
| `glove-25-angular` | 25 | 1.18M | Cosine |

## LDBC Graphalytics

Graph algorithm benchmarks with LDBC reference validation. Stores graphs in Strata KV as adjacency lists, reads back, and runs algorithms in Python.

```bash
python run.py graphalytics --algorithm bfs wcc pagerank --dataset example-directed
```

Algorithms: BFS, WCC, PageRank, CDLP, LCC, SSSP

## Batch Runner

Run all Phase 1 benchmarks and generate reports:

```bash
python scripts/run_all.py                                    # all Phase 1 benchmarks
python scripts/run_all.py --bench beir ycsb --latex          # specific suites + LaTeX
python scripts/run_all.py --bench ycsb --workload a b c      # YCSB subset
python scripts/run_all.py --clean                            # fresh start
```

## Results

All benchmarks write JSON results to `results/`. Generate reports:

```bash
python run.py report                    # Markdown summary
python run.py report --format latex     # LaTeX tables for papers
python run.py report --bench beir ann   # specific suites
```

## Phase 2 Benchmarks

LoCoMo and RAGAS have retrieval pipelines implemented; LLM evaluation is marked with TODOs. LongMemEval and GraphRAG-Bench are stubs with pipeline structure defined. All require `STRATA_MODEL_ENDPOINT` and `STRATA_MODEL_NAME` environment variables.
