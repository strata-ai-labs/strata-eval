"""Configuration for LoCoMo (Long-Context Conversational Memory) benchmark."""

# Dataset source — from the LoCoMo GitHub repository (SNAP Research)
# See: https://github.com/snap-research/locomo
LOCOMO_DATASET = {
    "url": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "description": "Long-context conversational memory with ~300 turns, ~9K tokens avg",
}

# Evaluation metrics (F1 and ROUGE-L are standard LoCoMo metrics)
METRICS = ["f1", "rouge_l"]

# Required environment variables for LLM evaluation
LLM_ENV_VARS = ["STRATA_MODEL_ENDPOINT", "STRATA_MODEL_NAME"]
