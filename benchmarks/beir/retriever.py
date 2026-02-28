"""StrataSearch — BEIR BaseSearch adapter for the Strata search engine."""

from __future__ import annotations

import os
import tempfile
import time

from beir.retrieval.search import BaseSearch
from lib.strata_client import StrataClient
from tqdm import tqdm


class StrataSearch(BaseSearch):
    """BEIR-compatible search adapter that indexes a corpus into Strata and
    runs queries using its hybrid search pipeline.

    Implements the ``BaseSearch.search()`` contract: given corpus + queries +
    top_k, returns ``{query_id: {doc_id: score}}``.
    """

    # Maps CLI mode names to client.search() kwargs.
    _SEARCH_KWARGS = {
        "keyword": {"mode": "keyword"},
        "hybrid": {"mode": "hybrid"},
        "hybrid-llm": {"mode": "hybrid", "expand": True, "rerank": True},
    }

    def __init__(self, mode: str = "hybrid", db_path: str | None = None,
                 embed_model: str = "miniLM"):
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
        self.embed_model = embed_model
        self._client: StrataClient | None = None
        self._tmpdir = None
        self.index_time: float = 0.0
        self.search_time: float = 0.0

    def _open_db(self) -> StrataClient:
        """Lazily open a Strata database via the CLI."""
        if self._client is None:
            if self.db_path:
                os.makedirs(self.db_path, exist_ok=True)
                db_dir = self.db_path
            else:
                self._tmpdir = tempfile.TemporaryDirectory()
                db_dir = self._tmpdir.name
            use_embed = self.mode != "keyword"
            self._client = StrataClient(db_path=db_dir, auto_embed=use_embed)
            if use_embed:
                self._client.setup()
            if self.mode == "hybrid-llm":
                self._client.configure_model(
                    endpoint=os.environ["STRATA_MODEL_ENDPOINT"],
                    model=os.environ["STRATA_MODEL_NAME"],
                    api_key=os.environ.get("STRATA_MODEL_API_KEY"),
                )
        return self._client

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
        client = self._open_db()

        # Index corpus (skip if DB already has data)
        t0 = time.perf_counter()
        existing_keys = client.kv.list()
        if len(existing_keys) >= len(corpus):
            print(f"Database already contains {len(existing_keys)} docs, skipping indexing")
        else:
            for doc_id, doc in tqdm(corpus.items(), desc="Indexing"):
                text = f"{doc.get('title', '')} {doc['text']}".strip()
                client.kv.put(doc_id, text)
            client.flush()
        self.index_time = time.perf_counter() - t0

        # Run queries sequentially — CLI pipe is single-threaded
        search_kwargs = self._SEARCH_KWARGS[self.mode]
        results: dict[str, dict[str, float]] = {}

        t1 = time.perf_counter()
        for qid, query_text in tqdm(queries.items(), desc="Searching"):
            hits = client.search(query_text, k=top_k, primitives=["kv"], **search_kwargs)
            results[qid] = {h["entity"]: h["score"] for h in hits}

        self.search_time = time.perf_counter() - t1
        return results

    def encode(self, *args, **kwargs):
        raise NotImplementedError("StrataSearch is a full search engine, not an encoder")

    def search_from_files(self, *args, **kwargs):
        raise NotImplementedError("StrataSearch is a full search engine, not an encoder")

    def cleanup(self):
        """Clean up the database. Only removes temp directories, not persistent ones."""
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
