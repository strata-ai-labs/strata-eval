"""Dataset definitions, algorithm list, and default parameters for LDBC Graphalytics."""

ALGORITHMS = ["bfs", "wcc", "pagerank", "cdlp", "lcc", "sssp"]

LDBC_DOWNLOAD_BASE = "https://datasets.ldbcouncil.org/graphalytics"

LDBC_DATASETS = {
    "example-directed": {
        "local": True,  # included in repo
        "directed": True,
    },
    "graph500-22": {
        "url": f"{LDBC_DOWNLOAD_BASE}/graph500-22.tar.zst",
        "directed": False,
    },
    "graph500-23": {
        "url": f"{LDBC_DOWNLOAD_BASE}/graph500-23.tar.zst",
        "directed": False,
    },
}

DEFAULT_RUNS = 10
DEFAULT_PAGERANK_ITERATIONS = 20
DEFAULT_PAGERANK_DAMPING = 0.85
DEFAULT_CDLP_ITERATIONS = 10
