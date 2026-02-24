"""Configuration for LongMemEval benchmark."""

# Dataset source
LONGMEMEVAL_DATASET = {
    "repo": "https://github.com/xiaowu0162/LongMemEval",
    "description": "500 questions across 115K-1.5M token multi-session chat histories",
}

# Five core evaluation abilities
ABILITIES = [
    "information_extraction",   # IE: retrieve specific details from distant history
    "multi_session_reasoning",  # MR: aggregate facts across sessions
    "temporal_reasoning",       # TR: leverage time cues and timestamps
    "knowledge_updates",        # KU: track information overwriting/invalidation
    "abstention",               # Know when insufficient information exists
]

# Dataset sizes
SIZES = {
    "S": {"description": "LongMemEval_S: 115K tokens", "max_tokens": 115_000},
    "M": {"description": "LongMemEval_M: 1.5M tokens", "max_tokens": 1_500_000},
}

# Required environment variables for LLM-judged evaluation
LLM_ENV_VARS = ["STRATA_MODEL_ENDPOINT", "STRATA_MODEL_NAME"]
