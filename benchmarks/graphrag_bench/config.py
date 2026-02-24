"""Configuration for GraphRAG-Bench benchmark."""

# Dataset source
GRAPHRAG_DATASET = {
    "url": "https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench",
    "description": "Graph-based RAG evaluation: knowledge graph construction + retrieval + reasoning",
}

# Metrics
METRICS = {
    "exact_match": "Binary correctness of generated answer",
    "lexical_overlap": "Longest common subsequence at word level",
    "reasoning_quality": "Whether reasoning follows evidence chain",
}

# Required environment variables for LLM evaluation
LLM_ENV_VARS = ["STRATA_MODEL_ENDPOINT", "STRATA_MODEL_NAME"]
