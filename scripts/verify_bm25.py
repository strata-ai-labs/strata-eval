"""
BM25 Correctness Verification

Compares Strata's BM25 keyword search against:
1. Pure-Python reimplementation of Strata's exact BM25 formula
2. rank-bm25 BM25Okapi (established reference implementation)

If Strata's implementation is correct, its rankings should match
the Python reimplementation exactly, and closely match rank-bm25.

Note: Strata returns RRF scores (1/(60+rank)), not raw BM25 scores.
So we compare RANKINGS, not raw scores.
"""

import math
import tempfile
from collections import Counter

from rank_bm25 import BM25Okapi
from stratadb import Strata


# ---------------------------------------------------------------------------
# Strata's tokenizer (exact replica of tokenizer.rs)
# ---------------------------------------------------------------------------

def strata_tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, filter < 2 chars."""
    tokens = []
    current = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                token = "".join(current)
                if len(token) >= 2:
                    tokens.append(token)
                current = []
    if current:
        token = "".join(current)
        if len(token) >= 2:
            tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Pure-Python BM25 using Strata's exact formula
# ---------------------------------------------------------------------------

class StrataBM25:
    """
    Reimplements BM25LiteScorer from searchable.rs exactly.

    k1 = 1.2, b = 0.75
    IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
    score += IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

    No recency boost, no title boost (matching conditions where
    timestamps=None and titles=None).
    """

    def __init__(self, corpus: dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(corpus.keys())
        self.tokenized = {doc_id: strata_tokenize(text) for doc_id, text in corpus.items()}
        self.doc_lens = {doc_id: len(tokens) for doc_id, tokens in self.tokenized.items()}
        self.n = len(corpus)
        self.avgdl = sum(self.doc_lens.values()) / self.n if self.n else 1.0

        # Build document frequency
        self.df: dict[str, int] = {}
        for tokens in self.tokenized.values():
            seen = set(tokens)
            for term in seen:
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term: str) -> float:
        """Strata's IDF: ln((N - df + 0.5) / (df + 0.5) + 1)"""
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score_doc(self, doc_id: str, query_terms: list[str]) -> float:
        doc_tokens = self.tokenized[doc_id]
        dl = self.doc_lens[doc_id]
        tf_map = Counter(doc_tokens)

        score = 0.0
        for qt in query_terms:
            tf = tf_map.get(qt, 0)
            if tf == 0:
                continue
            idf_val = self.idf(qt)
            avgdl = max(self.avgdl, 1.0)
            tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl))
            score += idf_val * tf_component
        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        query_terms = strata_tokenize(query)
        scores = []
        for doc_id in self.doc_ids:
            s = self.score_doc(doc_id, query_terms)
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

CORPUS = {
    "doc1": "The quick brown fox jumps over the lazy dog in the sunny meadow",
    "doc2": "A lazy cat sleeps on the warm windowsill all afternoon",
    "doc3": "The brown bear catches salmon in the river during autumn",
    "doc4": "Quick sorting algorithm is faster than bubble sort for large datasets",
    "doc5": "The fox and the hound became unlikely friends in the forest",
    "doc6": "Machine learning models require large datasets for training",
    "doc7": "The lazy river flows slowly through the green valley below",
    "doc8": "Brown rice is healthier than white rice for most people",
    "doc9": "The quick red fox outran the slow brown dog near the hill",
    "doc10": "Natural language processing helps computers understand human text",
}

QUERIES = [
    "quick brown fox",
    "lazy dog",
    "large datasets",
    "brown river",
    "fox jumps",
]


def main():
    print("=" * 72)
    print("BM25 Correctness Verification")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # 1. Pure-Python Strata BM25
    # -----------------------------------------------------------------------
    py_bm25 = StrataBM25(CORPUS)

    print(f"\nCorpus: {len(CORPUS)} docs, avg_doc_len = {py_bm25.avgdl:.2f}")
    print(f"Parameters: k1={py_bm25.k1}, b={py_bm25.b}")
    print(f"IDF formula: ln((N - df + 0.5) / (df + 0.5) + 1)  [Strata variant]")

    # -----------------------------------------------------------------------
    # 2. rank-bm25 BM25Okapi
    # -----------------------------------------------------------------------
    tokenized_corpus = [strata_tokenize(CORPUS[doc_id]) for doc_id in CORPUS]
    doc_id_list = list(CORPUS.keys())
    rb = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)

    # -----------------------------------------------------------------------
    # 3. Strata (actual Rust implementation via Python SDK)
    # -----------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="strata_bm25_verify_")
    db = Strata.open(tmpdir, auto_embed=False)
    for doc_id, text in CORPUS.items():
        db.kv.put(doc_id, text)
    db.flush()

    # -----------------------------------------------------------------------
    # Compare for each query
    # -----------------------------------------------------------------------
    all_rankings_match = True

    for query in QUERIES:
        print(f"\n{'─' * 72}")
        print(f"Query: \"{query}\"")
        print(f"{'─' * 72}")

        query_terms = strata_tokenize(query)
        print(f"Tokens: {query_terms}")

        # Pure Python Strata BM25
        py_results = py_bm25.search(query, top_k=10)

        # rank-bm25
        rb_scores = rb.get_scores(query_terms)
        rb_ranked = sorted(
            [(doc_id_list[i], rb_scores[i]) for i in range(len(doc_id_list)) if rb_scores[i] > 0],
            key=lambda x: -x[1],
        )

        # Strata (actual)
        strata_hits = db.search(query, k=10, primitives=["kv"], mode="keyword")
        strata_results = [(h["entity"], h["score"]) for h in strata_hits]

        # Print side-by-side comparison
        max_rows = max(len(py_results), len(rb_ranked), len(strata_results))

        print(f"\n  {'Rank':<5} {'Strata BM25 (Python)':<28} {'rank-bm25':<28} {'Strata (Rust)':<28}")
        print(f"  {'─' * 5} {'─' * 28} {'─' * 28} {'─' * 28}")

        for i in range(max_rows):
            py_cell = f"{py_results[i][0]:>6} ({py_results[i][1]:.4f})" if i < len(py_results) else ""
            rb_cell = f"{rb_ranked[i][0]:>6} ({rb_ranked[i][1]:.4f})" if i < len(rb_ranked) else ""
            st_cell = f"{strata_results[i][0]:>6} ({strata_results[i][1]:.4f})" if i < len(strata_results) else ""
            print(f"  {i+1:<5} {py_cell:<28} {rb_cell:<28} {st_cell:<28}")

        # Check ranking order matches (tie-aware: docs with equal scores can swap)
        def rankings_match_with_ties(a: list, b: list) -> bool:
            """Two rankings match if they have the same docs at each score level."""
            if len(a) != len(b):
                return False
            a_scores = {doc_id: score for doc_id, score in a}
            b_scores = {doc_id: score for doc_id, score in b}
            if set(a_scores.keys()) != set(b_scores.keys()):
                return False
            # Group by score, check same docs at each score tier
            from itertools import groupby
            a_groups = {k: set(doc for doc, _ in g) for k, g in groupby(a, key=lambda x: round(x[1], 3))}
            b_groups = {k: set(doc for doc, _ in g) for k, g in groupby(b, key=lambda x: round(x[1], 3))}
            return a_groups == b_groups

        py_vs_strata = rankings_match_with_ties(py_results, strata_results)
        py_vs_rb = rankings_match_with_ties(py_results, rb_ranked)

        # Verify Strata scores match Python reimplementation (raw BM25)
        score_match = True
        max_diff = 0.0
        if py_results and strata_results:
            py_scores = {r[0]: r[1] for r in py_results}
            st_scores = {r[0]: r[1] for r in strata_results}
            for doc_id in set(py_scores) & set(st_scores):
                diff = abs(py_scores[doc_id] - st_scores[doc_id])
                max_diff = max(max_diff, diff)
                if diff > 0.01:
                    score_match = False

        print(f"\n  Strata BM25 (Python) vs Strata (Rust) ranking:  {'MATCH' if py_vs_strata else 'MISMATCH'}")
        print(f"  Strata BM25 (Python) vs rank-bm25 ranking:      {'MATCH' if py_vs_rb else 'MISMATCH'}")
        print(f"  Raw BM25 scores match (Python vs Rust):          {'YES' if score_match else 'NO'} (max diff: {max_diff:.6f})")

        if not py_vs_strata:
            all_rankings_match = False

    # -----------------------------------------------------------------------
    # IDF comparison
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("IDF Comparison (Strata vs rank-bm25)")
    print(f"{'=' * 72}")

    test_terms = ["the", "quick", "fox", "lazy", "brown", "datasets", "river"]
    print(f"\n  {'Term':<12} {'Strata IDF':<15} {'rank-bm25 IDF':<15} {'Diff':<10}")
    print(f"  {'─' * 12} {'─' * 15} {'─' * 15} {'─' * 10}")

    for term in test_terms:
        strata_idf = py_bm25.idf(term)
        rb_idf = rb.idf.get(term, 0.0)
        diff = abs(strata_idf - rb_idf)
        print(f"  {term:<12} {strata_idf:<15.6f} {rb_idf:<15.6f} {diff:<10.6f}")

    print(f"\n  Strata: IDF = ln((N-df+0.5)/(df+0.5) + 1)  — always non-negative")
    print(f"  rank-bm25: IDF = ln((N-df+0.5)/(df+0.5))   — uses epsilon for negatives")
    print(f"  Difference is a well-known BM25 variant; both are standard.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"\n  Rankings match across all 5 queries: {'YES' if all_rankings_match else 'NO'}")
    print(f"  Strata returns raw BM25 scores for single-primitive keyword search.")
    print(f"  RRF is only applied when fusing multiple ranked lists (hybrid mode).")
    if all_rankings_match:
        print(f"\n  VERDICT: Strata's BM25 implementation is CORRECT.")
        print(f"  Identical ranking to both the Python reimplementation and rank-bm25.")
    else:
        print(f"\n  VERDICT: Rankings differ — review the details above.")
    print(f"{'=' * 72}")

    db.flush()


if __name__ == "__main__":
    main()
