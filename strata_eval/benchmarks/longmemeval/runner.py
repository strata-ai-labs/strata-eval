"""LongMemEval benchmark — long-term memory evaluation for chat assistants.

Evaluates 5 memory abilities across 500 questions on 115K-1.5M token histories.
Requires an LLM for judging answer quality.

Status: STUB — CLI arguments and pipeline structure defined. Both retrieval
pipeline and LLM evaluation require implementation. See TODOs below.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ...schema import BenchmarkResult
from ..base import BaseBenchmark
from .config import ABILITIES, LLM_ENV_VARS, LONGMEMEVAL_DATASET, SIZES

ROOT = Path(__file__).resolve().parent.parent.parent.parent


class LongMemEvalBenchmark(BaseBenchmark):
    name = "longmemeval"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--data-dir", type=str, default=str(ROOT / "datasets" / "longmemeval"),
            help="Directory for LongMemEval dataset",
        )
        parser.add_argument(
            "--size", default="S", choices=list(SIZES.keys()),
            help="Dataset size: S (115K tokens) or M (1.5M tokens)",
        )
        parser.add_argument(
            "--mode", default="hybrid", choices=["keyword", "hybrid"],
            help="Search mode for retrieval (default: hybrid)",
        )
        parser.add_argument(
            "--k", type=int, default=20,
            help="Number of passages to retrieve per question (default: 20)",
        )
        parser.add_argument(
            "--ability", nargs="*", default=None,
            choices=ABILITIES,
            help="Filter to specific abilities (default: all)",
        )

    def download(self, args: argparse.Namespace) -> None:
        print(f"LongMemEval dataset must be cloned from: {LONGMEMEVAL_DATASET['repo']}")
        print(f"  git clone {LONGMEMEVAL_DATASET['repo']}")
        print()
        print("Or download from HuggingFace:")
        print("  pip install datasets")
        print('  from datasets import load_dataset')
        print('  ds = load_dataset("xiaowu0162/longmemeval")')
        data_dir = Path(getattr(args, "data_dir", str(ROOT / "datasets" / "longmemeval")))
        print(f"  Place dataset files in: {data_dir}")

    def validate(self, args: argparse.Namespace) -> bool:
        missing = [v for v in LLM_ENV_VARS if not os.environ.get(v)]
        if missing:
            print(f"LongMemEval requires LLM env vars: {', '.join(missing)}")
            print("Set STRATA_MODEL_ENDPOINT and STRATA_MODEL_NAME to enable LLM-judged evaluation.")
            return False
        return True

    def run(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        """Run LongMemEval evaluation.

        Pipeline (all steps require implementation):
        1. Load multi-session chat histories from data_dir
        2. Index sessions into Strata KV with auto_embed
        3. For each of 500 questions, retrieve relevant context
        4. Generate answer using LLM
        5. Judge accuracy using LLM (GPT-4o as judge)
        6. Group results by ability (IE, MR, TR, KU, Abstention)

        Dataset format (from xiaowu0162/LongMemEval):
          - sessions: list of chat sessions, each containing turns
          - questions: list of {question, answer, ability, session_refs}
        """
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"LongMemEval dataset not found at {data_dir}. "
                f"Run: python -m strata_eval download --bench longmemeval"
            )

        raise NotImplementedError(
            "LongMemEval benchmark requires full implementation. Steps needed:\n"
            "  1. Dataset loading — parse the LongMemEval JSON/HF format\n"
            "  2. Strata indexing — index sessions via db.kv.put() with auto_embed\n"
            "  3. Retrieval — db.search() for each of the 500 questions\n"
            "  4. LLM answer generation — db.generate() or external API\n"
            "  5. LLM-judged accuracy — GPT-4o as judge per the paper\n"
            "  6. Per-ability metric aggregation (IE, MR, TR, KU, Abstention)\n"
            "Set STRATA_MODEL_ENDPOINT and STRATA_MODEL_NAME to enable."
        )
