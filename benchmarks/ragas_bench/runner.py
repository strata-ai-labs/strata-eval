"""RAGAS benchmark — RAG pipeline quality evaluation.

Evaluates retrieval-augmented generation quality using reference-free metrics:
faithfulness, answer relevance, context precision, context recall.

Status: SCAFFOLD — RAG retrieval pipeline implemented, LLM generation + RAGAS evaluation marked TODO.
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
from .config import DEFAULT_CHUNK_SIZE, DEFAULT_K, LLM_ENV_VARS, METRICS

ROOT = Path(__file__).resolve().parent.parent.parent


class RagasBenchmark(BaseBenchmark):
    name = "ragas"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--corpus", type=str, required=False,
            help="Path to corpus file (JSONL with 'text' field per line)",
        )
        parser.add_argument(
            "--questions", type=str, required=False,
            help="Path to questions file (JSONL with 'question' and 'answer' fields)",
        )
        parser.add_argument(
            "--data-dir", type=str, default=str(ROOT / "datasets" / "ragas"),
            help="Directory for RAGAS datasets",
        )
        parser.add_argument(
            "--k", type=int, default=DEFAULT_K,
            help=f"Passages to retrieve per question (default: {DEFAULT_K})",
        )
        parser.add_argument(
            "--mode", default="hybrid", choices=["keyword", "hybrid"],
            help="Search mode for retrieval (default: hybrid)",
        )

    def download(self, args: argparse.Namespace) -> None:
        print("RAGAS evaluation requires a user-provided corpus and question set.")
        print("Prepare two JSONL files:")
        print('  corpus.jsonl:    {"text": "document text", "id": "doc1"}')
        print('  questions.jsonl: {"question": "...", "answer": "ground truth", "contexts": ["..."]}')
        data_dir = Path(getattr(args, "data_dir", str(ROOT / "datasets" / "ragas")))
        print(f"Place files in: {data_dir}")

    def validate(self, args: argparse.Namespace) -> bool:
        missing = [v for v in LLM_ENV_VARS if not os.environ.get(v)]
        if missing:
            print(f"RAGAS requires LLM env vars: {', '.join(missing)}")
            print("Set STRATA_MODEL_ENDPOINT and STRATA_MODEL_NAME to enable evaluation.")
            return False
        return True

    def run(self, args: argparse.Namespace) -> list[BenchmarkResult]:
        """Run RAGAS evaluation.

        Pipeline:
        1. Index corpus into Strata KV with auto_embed
        2. For each test question, retrieve top-k passages
        3. TODO: Generate answer using LLM via db.generate()
        4. TODO: Evaluate with RAGAS metrics (faithfulness, relevance, precision, recall)
        """
        from stratadb import Strata

        data_dir = Path(args.data_dir)
        corpus_path = Path(args.corpus) if args.corpus else data_dir / "corpus.jsonl"
        questions_path = Path(args.questions) if args.questions else data_dir / "questions.jsonl"

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}. "
                f"Run: python run.py download --bench ragas"
            )

        # Load corpus
        corpus = []
        with open(corpus_path) as f:
            for line in f:
                corpus.append(json.loads(line))
        print(f"Loaded {len(corpus)} documents")

        # Load questions
        questions = []
        if questions_path.exists():
            with open(questions_path) as f:
                for line in f:
                    questions.append(json.loads(line))
        print(f"Loaded {len(questions)} questions")

        # Index corpus into Strata
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Strata.open(tmpdir, auto_embed=(args.mode == "hybrid"))

            t0 = time.perf_counter()
            for doc in corpus:
                doc_id = doc.get("id", doc.get("_id", str(hash(doc["text"][:100]))))
                db.kv.put(str(doc_id), doc["text"])
            db.flush()
            index_time = time.perf_counter() - t0
            print(f"Indexed {len(corpus)} documents in {index_time:.1f}s")

            # Retrieve for each question
            t1 = time.perf_counter()
            retrieval_results = []
            for q in questions:
                question_text = q["question"]
                hits = db.search(question_text, k=args.k, mode=args.mode, primitives=["kv"])
                contexts = [db.kv.get(h["entity"]) for h in hits]
                retrieval_results.append({
                    "question": question_text,
                    "contexts": contexts,
                    "ground_truth": q.get("answer", ""),
                })
            search_time = time.perf_counter() - t1
            print(f"Retrieved for {len(questions)} questions in {search_time:.1f}s")

        # TODO: Generate answers using LLM
        # for result in retrieval_results:
        #     context_str = "\n\n".join(result["contexts"])
        #     answer = db.generate(
        #         model=os.environ["STRATA_MODEL_NAME"],
        #         prompt=f"Context:\n{context_str}\n\nQuestion: {result['question']}\nAnswer:",
        #     )
        #     result["answer"] = answer["text"]
        #
        # TODO: Evaluate with RAGAS
        # from ragas import evaluate
        # from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        # from datasets import Dataset
        #
        # eval_dataset = Dataset.from_dict({
        #     "question": [r["question"] for r in retrieval_results],
        #     "answer": [r["answer"] for r in retrieval_results],
        #     "contexts": [r["contexts"] for r in retrieval_results],
        #     "ground_truth": [r["ground_truth"] for r in retrieval_results],
        # })
        # scores = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        #
        # return [BenchmarkResult(
        #     benchmark=f"ragas/{args.mode}",
        #     category="ragas",
        #     parameters={...},
        #     metrics={
        #         "faithfulness": scores["faithfulness"],
        #         "answer_relevance": scores["answer_relevancy"],
        #         "context_precision": scores["context_precision"],
        #         "context_recall": scores["context_recall"],
        #     },
        # )]

        raise NotImplementedError(
            "RAGAS benchmark requires LLM integration for answer generation and metric evaluation. "
            "Set STRATA_MODEL_ENDPOINT and STRATA_MODEL_NAME, then implement the TODO sections "
            "in ragas_bench/runner.py for: (1) LLM answer generation, (2) RAGAS metric computation "
            "(pip install ragas)."
        )
