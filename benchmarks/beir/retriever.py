"""StrataSearch — BEIR BaseSearch adapter for the Strata search engine."""

from __future__ import annotations

import os
import tempfile
import time

from beir.retrieval.search import BaseSearch
from stratadb import Strata
from tqdm import tqdm


class StrataSearch(BaseSearch):
    """BEIR-compatible search adapter that indexes a corpus into Strata and
    runs queries using its hybrid search pipeline.

    Implements the ``BaseSearch.search()`` contract: given corpus + queries +
    top_k, returns ``{query_id: {doc_id: score}}``.
    """

    # Maps CLI mode names to db.search() kwargs.
    _SEARCH_KWARGS = {
        "keyword": {"mode": "keyword"},
        "hybrid": {"mode": "hybrid"},
        "hybrid-llm": {"mode": "hybrid", "expand": True, "rerank": True},
    }

    def __init__(self, mode: str = "hybrid", db_path: str | None = None):
        if mode == "hybrid-llm":
            endpoint = os.environ.get("STRATA_MODEL_ENDPOINT")
            model = os.environ.get("STRATA_MODEL_NAME")
            if not endpoint or not model:
                raise RuntimeError(
                    "hybrid-llm mode requires STRATA_MODEL_ENDPOINT and "
                    "STRATA_MODEL_NAME environment variables"
                )
        self.mode = mode
        self.db_path = db_path
        self._db = None
        self._tmpdir = None
        self.index_time: float = 0.0
        self.search_time: float = 0.0

    def _open_db(self) -> Strata:
        """Lazily open a Strata database."""
        if self._db is None:
            if self.db_path:
                os.makedirs(self.db_path, exist_ok=True)
                db_dir = self.db_path
            else:
                self._tmpdir = tempfile.TemporaryDirectory()
                db_dir = self._tmpdir.name
            use_embed = self.mode != "keyword"
            self._db = Strata.open(db_dir, auto_embed=use_embed)
            if use_embed:
                self._db.models_pull("miniLM")
            if self.mode == "hybrid-llm":
                self._db.configure_model(
                    endpoint=os.environ["STRATA_MODEL_ENDPOINT"],
                    model=os.environ["STRATA_MODEL_NAME"],
                    api_key=os.environ.get("STRATA_MODEL_API_KEY"),
                )
        return self._db

    # ------------------------------------------------------------------
    # BaseSearch interface
    # ------------------------------------------------------------------

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        db = self._open_db()

        # Index corpus (skip if DB already has data)
        t0 = time.perf_counter()
        existing_keys = db.kv.list()
        if len(existing_keys) >= len(corpus):
            print(f"Database already contains {len(existing_keys)} docs, skipping indexing")
        else:
            for doc_id, doc in tqdm(corpus.items(), desc="Indexing"):
                text = f"{doc.get('title', '')} {doc['text']}".strip()
                db.kv.put(doc_id, text)
            db.flush()
        self.index_time = time.perf_counter() - t0

        # Run queries — parallel for keyword mode, sequential for hybrid/embed modes.
        # CUDA contexts are thread-local, so hybrid search (which embeds queries on
        # the GPU) must stay on the main thread that created the context during indexing.
        search_kwargs = self._SEARCH_KWARGS[self.mode]
        results: dict[str, dict[str, float]] = {}

        t1 = time.perf_counter()
        if self.mode == "keyword":
            # Keyword mode: no CUDA, safe to parallelize across threads
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _search_one(qid_query):
                qid, query_text = qid_query
                hits = db.search(query_text, k=top_k, primitives=["kv"], **search_kwargs)
                return qid, {h["entity"]: h["score"] for h in hits}

            max_workers = min(8, os.cpu_count() or 1)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_search_one, (qid, qt)): qid
                    for qid, qt in queries.items()
                }
                for future in tqdm(
                    as_completed(futures), total=len(queries), desc="Searching"
                ):
                    qid, hits = future.result()
                    results[qid] = hits
        else:
            # Hybrid/embed mode: must run on main thread (CUDA context affinity)
            for qid, query_text in tqdm(queries.items(), desc="Searching"):
                hits = db.search(query_text, k=top_k, primitives=["kv"], **search_kwargs)
                results[qid] = {h["entity"]: h["score"] for h in hits}

        self.search_time = time.perf_counter() - t1
        return results

    def encode(self, *args, **kwargs):
        raise NotImplementedError("StrataSearch is a full search engine, not an encoder")

    def search_from_files(self, *args, **kwargs):
        raise NotImplementedError("StrataSearch is a full search engine, not an encoder")

    def cleanup(self):
        """Clean up the database. Only removes temp directories, not persistent ones."""
        self._db = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
