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
