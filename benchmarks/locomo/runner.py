"""LoCoMo benchmark — long-context conversational memory evaluation.

Measures a system's ability to recall facts from long multi-session conversations.
Requires an LLM for answer generation and evaluation.

Status: SCAFFOLD — retrieval pipeline implemented, LLM evaluation marked TODO.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from lib.schema import BenchmarkResult
from benchmarks.base import BaseBenchmark
from .config import LOCOMO_DATASET, LLM_ENV_VARS

ROOT = Path(__file__).resolve().parent.parent.parent


class LocomoBenchmark(BaseBenchmark):
    name = "locomo"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--data-dir", type=str, default=str(ROOT / "datasets" / "locomo"),
            help="Directory for LoCoMo dataset",
        )
        parser.add_argument(
            "--mode", default="hybrid", choices=["keyword", "hybrid"],
            help="Search mode for retrieval (default: hybrid)",
        )
        parser.add_argument(
            "--k", type=int, default=10,
            help="Number of turns to retrieve per question (default: 10)",
        )

    def download(self, args: argparse.Namespace) -> None:
        from lib.download import download_file
        data_dir = Path(getattr(args, "data_dir", str(ROOT / "datasets" / "locomo")))
        dest = data_dir / "locomo.json"
        download_file(LOCOMO_DATASET["url"], dest, desc="LoCoMo dataset")

    def validate(self, args: argparse.Namespace) -> bool:
        missing = [v for v in LLM_ENV_VARS if not os.environ.get(v)]
        if missing:
            print(f"LoCoMo requires LLM env vars: {', '.join(missing)}")
            print("Set STRATA_MODEL_ENDPOINT and STRATA_MODEL_NAME to enable evaluation.")
            return False
        return True

    def run(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        """Run LoCoMo evaluation.

        Pipeline:
        1. Load conversation dataset
        2. Index conversation turns into Strata KV with auto_embed via CLI
        3. For each QA pair, retrieve relevant turns via client.search()
        4. TODO: Generate answer via LLM
        5. TODO: Compute F1, ROUGE-L, FactScore
        """
        from lib.strata_client import StrataClient

        data_dir = Path(args.data_dir)
        dataset_path = data_dir / "locomo.json"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"LoCoMo dataset not found at {dataset_path}. "
                f"Run: python run.py download --bench locomo"
            )

        with open(dataset_path) as f:
            conversations = json.load(f)

        print(f"Loaded {len(conversations)} conversations")

        results = []
        for conv_idx, conversation in enumerate(conversations[:5]):  # Start with first 5
            turns = conversation.get("turns", conversation.get("dialogue", []))
            qa_pairs = conversation.get("qa_pairs", conversation.get("questions", []))

            if not turns or not qa_pairs:
                continue

            # Index turns into Strata
            with tempfile.TemporaryDirectory() as tmpdir, \
                    StrataClient(db_path=tmpdir, auto_embed=(args.mode == "hybrid")) as client:

                t0 = time.perf_counter()
                for i, turn in enumerate(turns):
                    text = turn if isinstance(turn, str) else turn.get("text", str(turn))
                    client.kv.put(f"turn:{i:06d}", text)
                client.flush()
                index_time = time.perf_counter() - t0

                # Retrieve for each QA pair
                t1 = time.perf_counter()
                retrieval_results = []
                for qa in qa_pairs:
                    question = qa if isinstance(qa, str) else qa.get("question", str(qa))
                    hits = client.search(question, k=args.k, mode=args.mode, primitives=["kv"])
                    retrieval_results.append({
                        "question": question,
                        "retrieved": [h["entity"] for h in hits],
                    })
                search_time = time.perf_counter() - t1

                # TODO: Generate answers using LLM
                # TODO: Compute metrics (F1, ROUGE-L, FactScore)

                print(f"  Conversation {conv_idx}: {len(turns)} turns, "
                      f"{len(qa_pairs)} QA pairs, "
                      f"index={index_time:.2f}s, search={search_time:.2f}s")

        raise NotImplementedError(
            "LoCoMo benchmark requires LLM integration for answer generation and evaluation. "
            "Set STRATA_MODEL_ENDPOINT and STRATA_MODEL_NAME, then implement the TODO sections "
            "in locomo/runner.py for: (1) LLM answer generation, (2) F1/ROUGE-L/FactScore computation."
        )
