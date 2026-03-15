"""StrataSearch — BEIR BaseSearch adapter for the Strata search engine.

All database operations (indexing, search) are executed by piping
pre-generated commands directly to the ``strata`` CLI binary via
:func:`batch_execute`.  Python is used only for dataset handling and
metric computation — there is no Python overhead per database operation
in the timing loop.
"""

from __future__ import annotations

import os
import shlex
import tempfile

from beir.retrieval.search import BaseSearch
from lib.strata_client import StrataClient, batch_execute
from tqdm import tqdm


class StrataSearch(BaseSearch):
    """BEIR-compatible search adapter that indexes a corpus into Strata and
    runs queries using its hybrid search pipeline.

    Implements the ``BaseSearch.search()`` contract: given corpus + queries +
    top_k, returns ``{query_id: {doc_id: score}}``.
    """

    # Maps CLI mode names to search command flags.
    _SEARCH_FLAGS = {
        "keyword": ["--mode", "keyword"],
        "hybrid": ["--mode", "hybrid"],
        "hybrid-llm": ["--mode", "hybrid", "--expand", "--rerank"],
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
        self._db_dir: str | None = None
        self._tmpdir = None
        self._setup_done = False
        self.index_time: float = 0.0
        self.search_time: float = 0.0

    def _ensure_db_dir(self) -> str:
        """Create or return the database directory."""
        if self._db_dir is not None:
            return self._db_dir
        if self.db_path:
            os.makedirs(self.db_path, exist_ok=True)
            self._db_dir = self.db_path
        else:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._db_dir = self._tmpdir.name
        return self._db_dir

    def _ensure_setup(self) -> None:
        """One-time setup: download embedding model, configure LLM endpoint.

        Uses a short-lived StrataClient for interactive setup commands,
        then closes it so batch_execute can open the same database.
        """
        if self._setup_done:
            return
        db_dir = self._ensure_db_dir()
        use_embed = self.mode != "keyword"
        if use_embed or self.mode == "hybrid-llm":
            with StrataClient(db_path=db_dir, auto_embed=use_embed) as client:
                if use_embed:
                    client.setup()
                if self.mode == "hybrid-llm":
                    client.configure_model(
                        endpoint=os.environ["STRATA_MODEL_ENDPOINT"],
                        model=os.environ["STRATA_MODEL_NAME"],
                        api_key=os.environ.get("STRATA_MODEL_API_KEY"),
                    )
        self._setup_done = True

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
        self._ensure_setup()
        db_dir = self._ensure_db_dir()
        use_embed = self.mode != "keyword"

        # -- Index corpus via CLI batch ------------------------------------
        # Check if DB already has data (via a short-lived client)
        needs_index = True
        with StrataClient(db_path=db_dir, auto_embed=use_embed) as client:
            existing_keys = client.kv.list()
            if len(existing_keys) >= len(corpus):
                print(f"Database already contains {len(existing_keys)} docs, skipping indexing")
                needs_index = False

        if needs_index:
            print("Generating index commands...")
            index_cmds: list[str] = []
            for doc_id, doc in tqdm(corpus.items(), desc="Preparing"):
                text = f"{doc.get('title', '')} {doc['text']}".strip()
                index_cmds.append(
                    f"kv put {shlex.quote(doc_id)} {shlex.quote(text)}"
                )
            index_cmds.append("flush")

            print(f"Indexing {len(corpus)} documents via CLI...")
            self.index_time, _ = batch_execute(
                index_cmds,
                db_path=db_dir,
                auto_embed=use_embed,
                parse_responses=False,
            )
            print(f"  Index time: {self.index_time:.1f}s")
        else:
            self.index_time = 0.0

        # -- Search via CLI batch ------------------------------------------
        mode_flags = self._SEARCH_FLAGS[self.mode]
        search_cmds: list[str] = []
        query_ids: list[str] = []

        for qid, query_text in queries.items():
            parts = ["search", shlex.quote(query_text), str(top_k)]
            parts.extend(mode_flags)
            parts.extend(["--primitives", "kv"])
            search_cmds.append(" ".join(parts))
            query_ids.append(qid)

        print(f"Searching {len(queries)} queries via CLI...")
        self.search_time, responses = batch_execute(
            search_cmds,
            db_path=db_dir,
            auto_embed=use_embed,
        )

        # -- Parse results -------------------------------------------------
        results: dict[str, dict[str, float]] = {}
        for qid, hits in zip(query_ids, responses):
            if isinstance(hits, list):
                results[qid] = {h["entity"]: h["score"] for h in hits}
            else:
                results[qid] = {}

        return results

    def encode(self, *args, **kwargs):
        raise NotImplementedError("StrataSearch is a full search engine, not an encoder")

    def search_from_files(self, *args, **kwargs):
        raise NotImplementedError("StrataSearch is a full search engine, not an encoder")

    def cleanup(self):
        """Clean up the database. Only removes temp directories, not persistent ones."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        self._db_dir = None
        self._setup_done = False
