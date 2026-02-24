"""Configuration for RAGAS (Retrieval Augmented Generation Assessment) benchmark."""

# RAGAS metrics (all 0-1 scale, higher is better)
METRICS = {
    "faithfulness": "Answer consistency with retrieved context (detects hallucinations)",
    "answer_relevance": "How well the answer addresses the question",
    "context_precision": "Signal-to-noise ratio of retrieved context",
    "context_recall": "Completeness of retrieved context for answering",
}

# Default RAG pipeline parameters
DEFAULT_K = 5           # Number of passages to retrieve
DEFAULT_CHUNK_SIZE = 512  # Characters per chunk for corpus splitting

# Required environment variables for LLM evaluation
LLM_ENV_VARS = ["STRATA_MODEL_ENDPOINT", "STRATA_MODEL_NAME"]
