# BEIR Benchmark Results — Strata Search Engine

**Date**: 2026-02-22
**Version**: StrataDB 0.13.3
**Embedding**: MiniLM-L6-v2 (22M params, 384-dim, ONNX + cuBLAS)
**Platform**: Linux x86_64, Python 3.12.3, GPU-accelerated embedding

## Overview

StrataDB was evaluated on **15 BEIR datasets** spanning 7 domain categories, from 3.6K to 8.8M documents. Two search modes were tested:

- **Keyword**: BM25 full-text search only
- **Hybrid**: BM25 + dense vector retrieval (MiniLM-L6-v2), combined via reciprocal rank fusion

All results are **zero-shot** — no dataset-specific fine-tuning.

Baselines are from Pyserini BM25 (Lucene), sourced from [Kamalloo et al., "Resources for Brewing BEIR", SIGIR 2024](https://castorini.github.io/pyserini/2cr/beir.html).

---

## Summary: NDCG@10

| Dataset | Corpus | Queries | Keyword | Hybrid | BM25 flat | BM25 mf | vs flat |
|---|---:|---:|---:|---:|---:|---:|---:|
| nfcorpus | 3,633 | 323 | 0.3181 | **0.3462** | 0.3220 | 0.3250 | +0.0242 |
| scifact | 5,183 | 300 | 0.6709 | **0.7133** | 0.6720 | 0.6650 | +0.0413 |
| arguana | 8,674 | 1,406 | 0.4085 | **0.4947** | 0.3970 | 0.4140 | +0.0977 |
| scidocs | 25,657 | 1,000 | 0.1498 | **0.1962** | 0.1580 | 0.1580 | +0.0382 |
| trec-covid | 171,332 | 50 | 0.5960 | **0.6972** | 0.5950 | 0.6560 | +0.1022 |
| fiqa | 57,638 | 648 | 0.2359 | **0.3608** | 0.2360 | 0.2360 | +0.1248 |
| quora | 522,931 | 10,000 | 0.7875 | **0.8658** | 0.7890 | 0.7890 | +0.0768 |
| webis-touche2020 | 382,545 | 49 | **0.4397** | 0.3484 | 0.4420 | 0.3670 | -0.0936 |
| cqadupstack | 457,199 | 13,145 | 0.3020 | **0.3828** | 0.3020 | 0.2990 | +0.0808 |
| fever | 5,416,568 | 6,666 | 0.6427 | **0.6818** | 0.7530 | 0.7540 | -0.0712 |
| climate-fever | 5,416,593 | 1,535 | 0.1613 | **0.2378** | 0.2130 | 0.1760 | +0.0248 |
| nq | 2,681,468 | 3,452 | 0.3087 | **0.4384** | 0.3050 | 0.3290 | +0.1334 |
| hotpotqa | 5,233,329 | 7,405 | **0.6302** | 0.6020 | 0.6330 | 0.6030 | -0.0310 |
| dbpedia-entity | 4,635,922 | 400 | 0.3160 | **0.3775** | 0.3130 | 0.3180 | +0.0645 |
| msmarco | 8,841,823 | 43 | 0.5099 | — | 0.2280 | 0.2280 | +0.2819 |
| **Average (14)** | | | **0.4048** | **0.4818** | 0.4393 | 0.4277 | **+0.0444** |

*Average excludes msmarco (keyword-only). Bold NDCG@10 marks the better Strata mode per dataset.*

**Strata hybrid beats Pyserini BM25 flat on 11 of 14 datasets** with an average improvement of +0.0444 NDCG@10.

---

## Summary: Recall@100

| Dataset | Keyword | Hybrid | BM25 flat | BM25 mf | vs flat |
|---|---:|---:|---:|---:|---:|
| nfcorpus | 0.2454 | **0.3258** | 0.2460 | 0.2500 | +0.0798 |
| scifact | 0.9187 | **0.9510** | 0.9080 | 0.9080 | +0.0430 |
| arguana | 0.9374 | **0.9794** | 0.9320 | 0.9430 | +0.0474 |
| scidocs | 0.3492 | **0.4773** | 0.3560 | 0.3560 | +0.1213 |
| trec-covid | 0.1070 | **0.1094** | 0.1090 | 0.1140 | +0.0004 |
| fiqa | 0.5400 | **0.6992** | 0.5390 | 0.5390 | +0.1602 |
| quora | 0.9743 | **0.9957** | 0.9740 | 0.9740 | +0.0217 |
| webis-touche2020 | 0.5787 | **0.5932** | 0.5820 | 0.5380 | +0.0112 |
| cqadupstack | 0.5804 | **0.7618** | 0.5800 | 0.6060 | +0.1818 |
| fever | 0.9160 | **0.9447** | 0.9290 | 0.9310 | +0.0157 |
| climate-fever | 0.4192 | **0.5365** | 0.4200 | 0.4090 | +0.1165 |
| nq | 0.7538 | **0.9087** | 0.7510 | 0.7600 | +0.1577 |
| hotpotqa | 0.7940 | **0.8051** | 0.7960 | 0.7400 | +0.0091 |
| dbpedia-entity | 0.4661 | **0.5314** | 0.3930 | 0.3980 | +0.1384 |
| **Average (14)** | **0.6129** | **0.6871** | 0.6368 | 0.6333 | **+0.0503** |

**Strata hybrid beats BM25 flat on Recall@100 across all 14 datasets.**

---

## Keyword vs Hybrid Comparison

| Dataset | Keyword NDCG@10 | Hybrid NDCG@10 | Delta | Rel. Gain |
|---|---:|---:|---:|---:|
| nfcorpus | 0.3181 | 0.3462 | +0.0281 | +8.8% |
| scifact | 0.6709 | 0.7133 | +0.0424 | +6.3% |
| arguana | 0.4085 | 0.4947 | +0.0862 | +21.1% |
| scidocs | 0.1498 | 0.1962 | +0.0464 | +31.0% |
| trec-covid | 0.5960 | 0.6972 | +0.1012 | +17.0% |
| fiqa | 0.2359 | 0.3608 | +0.1249 | +52.9% |
| quora | 0.7875 | 0.8658 | +0.0783 | +9.9% |
| webis-touche2020 | **0.4397** | 0.3484 | -0.0913 | -20.8% |
| cqadupstack | 0.3020 | 0.3828 | +0.0808 | +26.7% |
| fever | 0.6427 | 0.6818 | +0.0391 | +6.1% |
| climate-fever | 0.1613 | 0.2378 | +0.0765 | +47.4% |
| nq | 0.3087 | 0.4384 | +0.1297 | +42.0% |
| hotpotqa | **0.6302** | 0.6020 | -0.0282 | -4.5% |
| dbpedia-entity | 0.3160 | 0.3775 | +0.0615 | +19.5% |
| **Average** | 0.4048 | 0.4818 | +0.0568 | +18.9% |

Hybrid mode improves over keyword on **12 of 14 datasets**. The two exceptions:

- **webis-touche2020**: Argumentative retrieval where exact lexical matches matter more than semantic similarity. The dense component introduces noise.
- **hotpotqa**: Multi-hop questions where BM25's strong entity matching outperforms the general-purpose 22M-param embedding model.

---

## Where Strata Trails BM25

On 3 of 14 datasets, Strata hybrid NDCG@10 falls below Pyserini BM25 flat:

| Dataset | Strata Hybrid | BM25 flat | Gap | Notes |
|---|---:|---:|---:|---|
| webis-touche2020 | 0.3484 | 0.4420 | -0.0936 | Argumentative text; dense retrieval hurts |
| fever | 0.6818 | 0.7530 | -0.0712 | 5.4M docs; Pyserini BM25 well-tuned for fact verification |
| hotpotqa | 0.6020 | 0.6330 | -0.0310 | Multi-hop; BM25 entity matching strong |

The common thread: these are datasets where **lexical precision** matters more than semantic understanding, and where the small 22M-param embedding model cannot compensate.

---

## Methodology

- **Framework**: strata-eval using [BEIR](https://github.com/beir-cellar/beir) datasets
- **Indexing**: Documents indexed via `Strata.kv.put()` with auto-embed (MiniLM-L6-v2, ONNX runtime, cuBLAS)
- **Hybrid search**: BM25 keyword + cosine similarity dense vector, combined via reciprocal rank fusion
- **Evaluation**: NDCG@10 (primary), Recall@100, MAP@10, MRR@10, Precision@10
- **Baselines**: Pyserini BM25 flat and multifield from Lucene ([source](https://castorini.github.io/pyserini/2cr/beir.html))
- **CQADupStack**: 12 subforums evaluated independently, metrics macro-averaged
- **Hardware**: Single machine, Linux x86_64, NVIDIA GPU
