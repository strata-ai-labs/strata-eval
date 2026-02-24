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
    "fiqa": {
        "url": f"{BEIR_BASE}/fiqa.zip",
        "docs": "57K",
        "queries": 648,
    },
    "quora": {
        "url": f"{BEIR_BASE}/quora.zip",
        "docs": "523K",
        "queries": 10000,
    },
    "webis-touche2020": {
        "url": f"{BEIR_BASE}/webis-touche2020.zip",
        "docs": "382K",
        "queries": 49,
    },
    "cqadupstack": {
        "url": f"{BEIR_BASE}/cqadupstack.zip",
        "docs": "457K",
        "queries": 13145,
    },
    "fever": {
        "url": f"{BEIR_BASE}/fever.zip",
        "docs": "5.42M",
        "queries": 6666,
    },
    "climate-fever": {
        "url": f"{BEIR_BASE}/climate-fever.zip",
        "docs": "5.42M",
        "queries": 1535,
    },
    "nq": {
        "url": f"{BEIR_BASE}/nq.zip",
        "docs": "2.68M",
        "queries": 3452,
    },
    "hotpotqa": {
        "url": f"{BEIR_BASE}/hotpotqa.zip",
        "docs": "5.23M",
        "queries": 7405,
    },
    "dbpedia-entity": {
        "url": f"{BEIR_BASE}/dbpedia-entity.zip",
        "docs": "4.63M",
        "queries": 400,
    },
    "msmarco": {
        "url": f"{BEIR_BASE}/msmarco.zip",
        "docs": "8.84M",
        "queries": 6980,
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
    "fiqa": {
        "bm25_flat":  {"NDCG@10": 0.236, "Recall@100": 0.539},
        "bm25_mf":    {"NDCG@10": 0.236, "Recall@100": 0.539},
    },
    "quora": {
        "bm25_flat":  {"NDCG@10": 0.789, "Recall@100": 0.974},
        "bm25_mf":    {"NDCG@10": 0.789, "Recall@100": 0.974},
    },
    "webis-touche2020": {
        "bm25_flat":  {"NDCG@10": 0.442, "Recall@100": 0.582},
        "bm25_mf":    {"NDCG@10": 0.367, "Recall@100": 0.538},
    },
    "cqadupstack": {
        "bm25_flat":  {"NDCG@10": 0.302, "Recall@100": 0.580},
        "bm25_mf":    {"NDCG@10": 0.299, "Recall@100": 0.606},
    },
    "fever": {
        "bm25_flat":  {"NDCG@10": 0.753, "Recall@100": 0.929},
        "bm25_mf":    {"NDCG@10": 0.754, "Recall@100": 0.931},
    },
    "climate-fever": {
        "bm25_flat":  {"NDCG@10": 0.213, "Recall@100": 0.420},
        "bm25_mf":    {"NDCG@10": 0.176, "Recall@100": 0.409},
    },
    "nq": {
        "bm25_flat":  {"NDCG@10": 0.305, "Recall@100": 0.751},
        "bm25_mf":    {"NDCG@10": 0.329, "Recall@100": 0.760},
    },
    "hotpotqa": {
        "bm25_flat":  {"NDCG@10": 0.633, "Recall@100": 0.796},
        "bm25_mf":    {"NDCG@10": 0.603, "Recall@100": 0.740},
    },
    "dbpedia-entity": {
        "bm25_flat":  {"NDCG@10": 0.313, "Recall@100": 0.393},
        "bm25_mf":    {"NDCG@10": 0.318, "Recall@100": 0.398},
    },
    "msmarco": {
        "bm25_flat":  {"NDCG@10": 0.228, "Recall@100": 0.654},
        "bm25_mf":    {"NDCG@10": 0.228, "Recall@100": 0.654},
    },
}
