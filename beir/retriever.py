"""StrataRetriever — indexes a BEIR corpus into Strata and runs queries."""

from __future__ import annotations

import os

from stratadb import Strata
from tqdm import tqdm


class StrataRetriever:
    """Thin wrapper that maps BEIR's corpus/query dicts to Strata operations."""

    # Maps CLI mode names to db.search() kwargs.
    _SEARCH_KWARGS = {
        "keyword": {"mode": "keyword"},
        "hybrid": {"mode": "hybrid"},
        "hybrid-llm": {"mode": "hybrid", "expand": True, "rerank": True},
    }

    def __init__(self, db_path: str, mode: str = "hybrid"):
        if mode == "hybrid-llm":
            endpoint = os.environ.get("STRATA_MODEL_ENDPOINT")
            model = os.environ.get("STRATA_MODEL_NAME")
            if not endpoint or not model:
                raise RuntimeError(
                    "hybrid-llm mode requires STRATA_MODEL_ENDPOINT and "
                    "STRATA_MODEL_NAME environment variables"
                )

        self.db = Strata.open(db_path, auto_embed=True)
        self.mode = mode

        if mode == "hybrid-llm":
            api_key = os.environ.get("STRATA_MODEL_API_KEY")
            self.db.configure_model(
                endpoint=os.environ["STRATA_MODEL_ENDPOINT"],
                model=os.environ["STRATA_MODEL_NAME"],
                api_key=api_key,
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, corpus: dict) -> None:
        """Index a BEIR corpus into Strata's KV store.

        Each document is stored as ``db.kv.put(doc_id, title + " " + text)``.
        With ``auto_embed=True``, Strata automatically creates shadow vector
        entries for hybrid search.
        """
        for doc_id, doc in tqdm(corpus.items(), desc="Indexing"):
            text = f"{doc.get('title', '')} {doc['text']}".strip()
            self.db.kv.put(doc_id, text)
        self.db.flush()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, queries: dict, k: int = 100) -> dict:
        """Run queries and return results in BEIR format.

        Returns:
            ``{query_id: {doc_id: score, ...}, ...}``
        """
        kwargs = self._SEARCH_KWARGS[self.mode]

        results: dict[str, dict[str, float]] = {}
        for qid, query_text in tqdm(queries.items(), desc="Searching"):
            hits = self.db.search(query_text, k=k, primitives=["kv"], **kwargs)
            results[qid] = {h["entity"]: h["score"] for h in hits}
        return results
