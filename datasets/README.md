# Datasets

Each benchmark suite stores its data in its own subdirectory.
All contents are gitignored — download or generate them before running benchmarks.

```
datasets/
  ann/            # HDF5 files from ann-benchmarks.com (sift-128, glove-*)
  beir/           # BEIR IR datasets (15 datasets, auto-downloaded as zips)
  graphalytics/   # LDBC Graphalytics graphs (.v, .e, .properties files)
  graphrag/       # GraphRAG-Bench from HuggingFace
  locomo/         # LoCoMo conversational memory dataset (JSON)
  longmemeval/    # LongMemEval chat histories from HuggingFace
  ragas/          # User-provided corpus.jsonl + questions.jsonl for RAGAS
```

## Downloading

```bash
# BEIR (auto-downloads on first run, or explicitly):
python run.py download --bench beir --dataset nfcorpus scifact

# ANN:
python run.py download --bench ann --dataset sift-128-euclidean

# Graphalytics:
python run.py download --bench graphalytics --dataset example-directed

# Others — see per-benchmark instructions:
python run.py download --bench locomo
python run.py download --bench longmemeval
python run.py download --bench ragas
python run.py download --bench graphrag
```
