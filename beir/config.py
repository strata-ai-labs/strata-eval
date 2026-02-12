"""Dataset definitions and evaluation constants for BEIR benchmarks."""

# BEIR datasets — start with small ones for fast iteration.
# The "url" field is used by beir.util.download_and_unzip().
BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

DATASETS = {
    "nfcorpus": {
        "url": f"{BEIR_BASE}/nfcorpus.zip",
        "docs": "3.6K",
        "queries": 323,
    },
    "scifact": {
        "url": f"{BEIR_BASE}/scifact.zip",
        "docs": "5K",
        "queries": 300,
    },
    "arguana": {
        "url": f"{BEIR_BASE}/arguana.zip",
        "docs": "8.7K",
        "queries": 1406,
    },
    "scidocs": {
        "url": f"{BEIR_BASE}/scidocs.zip",
        "docs": "25K",
        "queries": 1000,
    },
    "trec-covid": {
        "url": f"{BEIR_BASE}/trec-covid.zip",
        "docs": "171K",
        "queries": 50,
    },
}

# Cutoff depths for evaluation metrics (nDCG@k, Recall@k, etc.)
K_VALUES = [10, 100]

# Strata search modes
MODES = ["keyword", "hybrid", "hybrid-llm"]

# ---------------------------------------------------------------------------
# Pyserini BM25 reference baselines (Lucene, reproducible via Pyserini).
# Source: Kamalloo et al., "Resources for Brewing BEIR", SIGIR 2024.
# https://castorini.github.io/pyserini/2cr/beir.html
#
# "flat"  = single-field (title + text concatenated)
# "mf"    = multifield  (title and text as separate Lucene fields)
# ---------------------------------------------------------------------------
PYSERINI_BASELINES = {
    "nfcorpus": {
        "bm25_flat":  {"NDCG@10": 0.322, "Recall@100": 0.246},
        "bm25_mf":    {"NDCG@10": 0.325, "Recall@100": 0.250},
    },
    "scifact": {
        "bm25_flat":  {"NDCG@10": 0.672, "Recall@100": 0.908},
        "bm25_mf":    {"NDCG@10": 0.665, "Recall@100": 0.908},
    },
    "arguana": {
        "bm25_flat":  {"NDCG@10": 0.397, "Recall@100": 0.932},
        "bm25_mf":    {"NDCG@10": 0.414, "Recall@100": 0.943},
    },
    "scidocs": {
        "bm25_flat":  {"NDCG@10": 0.158, "Recall@100": 0.356},
        "bm25_mf":    {"NDCG@10": 0.158, "Recall@100": 0.356},
    },
    "trec-covid": {
        "bm25_flat":  {"NDCG@10": 0.595, "Recall@100": 0.109},
        "bm25_mf":    {"NDCG@10": 0.656, "Recall@100": 0.114},
    },
}
