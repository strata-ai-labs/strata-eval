# strata-eval

IR quality benchmarks for [Strata](https://github.com/strata-ai-labs/strata) hybrid search using [BEIR](https://github.com/beir-cellar/beir).

## What this measures

Strata provides a full hybrid search pipeline: BM25 keyword search + MiniLM vector search + RRF fusion. This repo evaluates retrieval quality on standard BEIR benchmarks so we can track improvements over time.

| Mode | What it tests | LLM required? |
|------|---------------|---------------|
| `keyword` | BM25 only | No |
| `hybrid` | BM25 + MiniLM vectors + RRF fusion | No |
| `hybrid-llm` | Full pipeline with LLM query expansion + reranking | Yes |

**Primary metrics**: `keyword` and `hybrid` — pure retrieval quality, no LLM needed.
**Optional**: `hybrid-llm` — shows ceiling with LLM-assisted search. Requires `STRATA_MODEL_ENDPOINT` and `STRATA_MODEL_NAME` env vars.

## Setup

```bash
pip install -e .
```

## Usage

```bash
# Run on a single dataset
python -m beir --dataset nfcorpus --mode hybrid

# Keyword-only baseline
python -m beir --dataset nfcorpus --mode keyword

# Try a different dataset
python -m beir --dataset scifact --mode hybrid

# Custom cutoff depths
python -m beir --dataset nfcorpus --mode hybrid --k 10 100 1000

# With LLM-assisted search (requires model endpoint)
export STRATA_MODEL_ENDPOINT="http://localhost:11434/v1"
export STRATA_MODEL_NAME="qwen3:1.7b"
python -m beir --dataset nfcorpus --mode hybrid-llm
```

## Available datasets

| Dataset | Docs | Queries | Domain |
|---------|------|---------|--------|
| `nfcorpus` | 3.6K | 323 | Nutrition/medical |
| `scifact` | 5K | 300 | Scientific claims |
| `arguana` | 8.7K | 1,406 | Argument retrieval |
| `scidocs` | 25K | 1,000 | Scientific papers |
| `trec-covid` | 171K | 50 | COVID-19 literature |

Start with `nfcorpus` or `scifact` — they're small enough for fast iteration.

## Interpreting results

- **nDCG@10** is the primary metric (standard in IR evaluation)
- Higher is better; 1.0 is perfect ranking
- Compare `keyword` vs `hybrid` to measure the value of vector search + fusion
- Results are saved as JSON in the `results/` directory

## Output

Each run prints a summary and writes a JSON file to `results/`:

```
============================================================
  Dataset: nfcorpus  |  Mode: hybrid
  Corpus: 3633 docs  |  Queries: 323
============================================================
        ndcg  NDCG@10: 0.3241, NDCG@100: 0.2876
         map  MAP@10: 0.0512, MAP@100: 0.0698
      recall  Recall@10: 0.0821, Recall@100: 0.2143
   precision  P@10: 0.0312, P@100: 0.0089
============================================================
```
