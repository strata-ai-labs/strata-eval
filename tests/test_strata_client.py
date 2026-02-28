"""Tests for StrataClient — the CLI subprocess wrapper.

Requires the ``strata`` binary to be available (via PATH, STRATA_BIN env,
or ../strata-core/target/release/strata).
"""

from __future__ import annotations

import tempfile
import unittest

from lib.strata_client import StrataClient, StrataError, _unwrap


def _have_strata() -> bool:
    """Return True if we can construct a StrataClient."""
    try:
        with tempfile.TemporaryDirectory() as d:
            c = StrataClient(db_path=d, cache=True)
            c.close()
        return True
    except (FileNotFoundError, StrataError):
        return False


# ======================================================================
# Unit tests for _unwrap (no binary needed)
# ======================================================================

class TestUnwrap(unittest.TestCase):
    """Test the response unwrapping logic independently of the CLI."""

    def test_unit_returns_none(self):
        self.assertIsNone(_unwrap("Unit"))

    def test_version_int(self):
        self.assertEqual(_unwrap({"Version": 3}), 3)

    def test_version_zero(self):
        self.assertEqual(_unwrap({"Version": 0}), 0)

    def test_maybe_versioned_with_string_value(self):
        payload = {"MaybeVersioned": {"value": {"String": "hello"}, "version": 1}}
        self.assertEqual(_unwrap(payload), "hello")

    def test_maybe_versioned_null(self):
        self.assertIsNone(_unwrap({"MaybeVersioned": None}))

    def test_maybe_versioned_with_int_value(self):
        payload = {"MaybeVersioned": {"value": {"Int": 42}, "version": 5}}
        self.assertEqual(_unwrap(payload), 42)

    def test_keys_list(self):
        self.assertEqual(_unwrap({"Keys": ["a", "b", "c"]}), ["a", "b", "c"])

    def test_keys_empty(self):
        self.assertEqual(_unwrap({"Keys": []}), [])

    def test_search_results(self):
        hits = [{"entity": "doc1", "score": 0.9}]
        self.assertEqual(_unwrap({"SearchResults": hits}), hits)

    def test_vector_matches(self):
        matches = [{"key": "a", "score": 0.5}]
        self.assertEqual(_unwrap({"VectorMatches": matches}), matches)

    def test_vector_stats(self):
        stats = {"count": 100, "memory_bytes": 4096}
        self.assertEqual(_unwrap({"VectorStats": stats}), stats)

    def test_graph_bfs(self):
        bfs_data = {"visited": ["0", "1"], "depths": {"0": 0, "1": 1}}
        self.assertEqual(_unwrap({"GraphBfs": bfs_data}), bfs_data)

    def test_graph_neighbors(self):
        nbrs = [{"id": "1", "edge_type": "edge"}]
        self.assertEqual(_unwrap({"GraphNeighbors": nbrs}), nbrs)

    def test_graph_nodes(self):
        self.assertEqual(_unwrap({"GraphNodes": ["a", "b"]}), ["a", "b"])

    def test_pong_dict(self):
        self.assertEqual(_unwrap({"Pong": {"version": "1.2.3"}}), "1.2.3")

    def test_pong_string(self):
        self.assertEqual(_unwrap({"Pong": "1.2.3"}), "1.2.3")

    def test_info(self):
        info = {"db_size": 1024}
        self.assertEqual(_unwrap({"Info": info}), info)

    def test_error_raises(self):
        with self.assertRaises(StrataError) as ctx:
            _unwrap({"error": "something went wrong"})
        self.assertIn("something went wrong", str(ctx.exception))

    def test_none_passthrough(self):
        self.assertIsNone(_unwrap(None))

    def test_bare_list_passthrough(self):
        self.assertEqual(_unwrap([1, 2, 3]), [1, 2, 3])

    def test_bare_int_passthrough(self):
        self.assertEqual(_unwrap(42), 42)

    def test_bare_float_passthrough(self):
        self.assertAlmostEqual(_unwrap(3.14), 3.14)

    def test_unknown_dict_passthrough(self):
        d = {"foo": "bar", "baz": 1}
        self.assertEqual(_unwrap(d), d)

    def test_unknown_single_key_passthrough(self):
        d = {"UnknownVariant": {"data": 1}}
        self.assertEqual(_unwrap(d), d)


# ======================================================================
# Binary resolution tests (no binary needed for some)
# ======================================================================

class TestBinaryResolution(unittest.TestCase):
    def test_explicit_path_returned(self):
        self.assertEqual(StrataClient._resolve_binary("/usr/bin/strata"), "/usr/bin/strata")

    def test_none_with_no_binary_raises(self):
        import os
        import shutil
        # Only test if strata is genuinely not available
        if shutil.which("strata") or os.environ.get("STRATA_BIN"):
            self.skipTest("strata is available, can't test missing-binary path")
        # Temporarily hide the fallback path
        orig = os.environ.pop("STRATA_BIN", None)
        try:
            # If which("strata") also returns something, we can't test this
            StrataClient._resolve_binary(None)
            # If it succeeds, the fallback path exists — skip
            self.skipTest("fallback binary path exists")
        except FileNotFoundError:
            pass  # Expected
        finally:
            if orig is not None:
                os.environ["STRATA_BIN"] = orig


# ======================================================================
# Integration tests (require strata binary)
# ======================================================================

@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestKvNamespace(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.client = StrataClient(db_path=self._tmpdir.name, cache=True)

    def tearDown(self):
        self.client.close()
        self._tmpdir.cleanup()

    def test_put_get_roundtrip(self):
        self.client.kv.put("hello", "world")
        self.assertEqual(self.client.kv.get("hello"), "world")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.client.kv.get("nonexistent"))

    def test_put_overwrite(self):
        """Putting the same key twice should overwrite the value."""
        self.client.kv.put("key", "first")
        self.assertEqual(self.client.kv.get("key"), "first")
        self.client.kv.put("key", "second")
        self.assertEqual(self.client.kv.get("key"), "second")

    def test_empty_value(self):
        self.client.kv.put("empty", "")
        result = self.client.kv.get("empty")
        self.assertEqual(result, "")

    def test_large_value(self):
        """Values up to ~64KB should work through the pipe."""
        large = "x" * 50_000
        self.client.kv.put("big", large)
        self.assertEqual(self.client.kv.get("big"), large)

    def test_json_value_roundtrip(self):
        """JSON-encoded values (like YCSB uses) should round-trip correctly."""
        import json
        value = json.dumps({"field0": "abc", "field1": "def"})
        self.client.kv.put("json_key", value)
        got = self.client.kv.get("json_key")
        self.assertEqual(json.loads(got), {"field0": "abc", "field1": "def"})

    def test_special_characters(self):
        self.client.kv.put("key with spaces", "value with 'quotes'")
        self.assertEqual(self.client.kv.get("key with spaces"), "value with 'quotes'")

    def test_unicode_value(self):
        self.client.kv.put("uni", "caf\u00e9 \u2603 \U0001f600")
        self.assertEqual(self.client.kv.get("uni"), "caf\u00e9 \u2603 \U0001f600")

    def test_newlines_collapsed_to_spaces(self):
        """Newlines in values are collapsed to spaces by the pipe protocol."""
        self.client.kv.put("nl", "line1\nline2\nline3")
        result = self.client.kv.get("nl")
        # Newlines become spaces due to pipe protocol
        self.assertEqual(result, "line1 line2 line3")

    def test_list_keys(self):
        self.client.kv.put("a", "1")
        self.client.kv.put("b", "2")
        self.client.kv.put("c", "3")
        keys = self.client.kv.list()
        self.assertIn("a", keys)
        self.assertIn("b", keys)
        self.assertIn("c", keys)
        self.assertEqual(len(keys), 3)

    def test_list_prefix(self):
        self.client.kv.put("prefix:one", "1")
        self.client.kv.put("prefix:two", "2")
        self.client.kv.put("other", "3")
        keys = self.client.kv.list(prefix="prefix:")
        self.assertEqual(sorted(keys), ["prefix:one", "prefix:two"])

    def test_list_limit(self):
        for i in range(10):
            self.client.kv.put(f"k{i:02d}", str(i))
        keys = self.client.kv.list(limit=3)
        self.assertEqual(len(keys), 3)

    def test_list_empty_db(self):
        keys = self.client.kv.list()
        self.assertEqual(keys, [])

    def test_delete(self):
        self.client.kv.put("del_me", "val")
        self.assertEqual(self.client.kv.get("del_me"), "val")
        self.client.kv.delete("del_me")
        self.assertIsNone(self.client.kv.get("del_me"))

    def test_delete_nonexistent_no_error(self):
        """Deleting a key that doesn't exist should not raise."""
        self.client.kv.delete("never_existed")

    def test_many_operations_sequential(self):
        """Verify the pipe stays healthy across many sequential operations."""
        for i in range(200):
            self.client.kv.put(f"seq:{i}", f"val:{i}")
        for i in range(200):
            self.assertEqual(self.client.kv.get(f"seq:{i}"), f"val:{i}")


@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestVectorNamespace(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.client = StrataClient(db_path=self._tmpdir.name, cache=True)

    def tearDown(self):
        self.client.close()
        self._tmpdir.cleanup()

    def test_create_upsert_search(self):
        coll = self.client.vectors.create("test", 3, "cosine")
        coll.upsert([
            {"key": "a", "vector": [1.0, 0.0, 0.0]},
            {"key": "b", "vector": [0.0, 1.0, 0.0]},
            {"key": "c", "vector": [0.0, 0.0, 1.0]},
        ])
        results = coll.search([1.0, 0.0, 0.0], k=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        # The nearest neighbor to [1,0,0] should be "a"
        top_keys = set()
        for r in results:
            if isinstance(r, dict):
                top_keys.add(r.get("key", r.get("id", "")))
        self.assertIn("a", top_keys)

    def test_search_returns_scores(self):
        coll = self.client.vectors.create("scored", 2, "cosine")
        coll.upsert([
            {"key": "x", "vector": [1.0, 0.0]},
            {"key": "y", "vector": [0.0, 1.0]},
        ])
        results = coll.search([1.0, 0.0], k=2)
        for r in results:
            if isinstance(r, dict):
                self.assertIn("score", r)

    def test_multi_batch_upsert(self):
        """Multiple upsert calls should accumulate vectors."""
        coll = self.client.vectors.create("multi", 2, "cosine")
        coll.upsert([{"key": "0", "vector": [1.0, 0.0]}])
        coll.upsert([{"key": "1", "vector": [0.0, 1.0]}])
        results = coll.search([1.0, 0.0], k=10)
        self.assertGreaterEqual(len(results), 2)

    def test_stats(self):
        coll = self.client.vectors.create("stats_test", 4, "cosine")
        coll.upsert([{"key": "x", "vector": [1.0, 2.0, 3.0, 4.0]}])
        stats = coll.stats()
        self.assertIsInstance(stats, dict)

    def test_empty_search(self):
        """Searching an empty collection should return an empty list."""
        coll = self.client.vectors.create("empty_coll", 2, "cosine")
        results = coll.search([1.0, 0.0], k=5)
        self.assertIsInstance(results, list)


@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestGraphNamespace(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.client = StrataClient(db_path=self._tmpdir.name, cache=True)

    def tearDown(self):
        self.client.close()
        self._tmpdir.cleanup()

    def _create_linear_graph(self, name: str = "g"):
        """Create a simple linear graph: 0 -> 1 -> 2 -> 3."""
        self.client.graph.create(name)
        self.client.graph.bulk_insert(
            name,
            nodes=[{"id": "0"}, {"id": "1"}, {"id": "2"}, {"id": "3"}],
            edges=[
                {"src": "0", "dst": "1", "edge_type": "edge", "weight": 1.0},
                {"src": "1", "dst": "2", "edge_type": "edge", "weight": 1.0},
                {"src": "2", "dst": "3", "edge_type": "edge", "weight": 1.0},
            ],
        )

    def test_create_and_bulk_insert(self):
        self._create_linear_graph()

    def test_bulk_insert_via_file_path(self):
        """Test the file_path parameter for bulk_insert."""
        import json
        import os
        import tempfile as tf
        self.client.graph.create("fp_g")
        payload = {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"src": "a", "dst": "b", "edge_type": "link", "weight": 2.0}],
        }
        with tf.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        try:
            self.client.graph.bulk_insert("fp_g", file_path=path)
        finally:
            os.unlink(path)

    def test_bfs_depths(self):
        """BFS from node 0 should produce correct depth values."""
        self._create_linear_graph("bfs_g")
        result = self.client.graph.bfs("bfs_g", "0", 100, direction="outgoing")
        self.assertIsInstance(result, dict)
        self.assertIn("depths", result)
        depths = result["depths"]
        # Node 0 should be at depth 0, node 1 at depth 1, etc.
        self.assertEqual(int(depths.get("0", depths.get(0, -1))), 0)
        self.assertEqual(int(depths.get("1", depths.get(1, -1))), 1)
        self.assertEqual(int(depths.get("2", depths.get(2, -1))), 2)
        self.assertEqual(int(depths.get("3", depths.get(3, -1))), 3)

    def test_bfs_visited(self):
        """BFS should visit all reachable nodes."""
        self._create_linear_graph("bfs_v")
        result = self.client.graph.bfs("bfs_v", "0", 100, direction="outgoing")
        visited = result.get("visited", [])
        self.assertEqual(len(visited), 4)

    def test_bfs_unreachable(self):
        """Nodes not reachable from the source should not appear in depths."""
        self.client.graph.create("unreach_g")
        self.client.graph.bulk_insert(
            "unreach_g",
            nodes=[{"id": "0"}, {"id": "1"}, {"id": "99"}],
            edges=[{"src": "0", "dst": "1", "edge_type": "edge", "weight": 1.0}],
        )
        result = self.client.graph.bfs("unreach_g", "0", 100, direction="outgoing")
        depths = result.get("depths", {})
        self.assertNotIn("99", depths)
        self.assertNotIn(99, depths)

    def test_neighbors_outgoing(self):
        self._create_linear_graph("nb_g")
        result = self.client.graph.neighbors("nb_g", "1", direction="outgoing")
        self.assertIsInstance(result, list)
        # Node 1 has one outgoing edge to node 2
        ids = set()
        for n in result:
            if isinstance(n, dict):
                ids.add(n.get("id", n.get("node_id", n.get("dst", ""))))
            else:
                ids.add(str(n))
        self.assertIn("2", ids)

    def test_neighbors_empty(self):
        """A leaf node with no outgoing edges should return empty list."""
        self._create_linear_graph("leaf_g")
        result = self.client.graph.neighbors("leaf_g", "3", direction="outgoing")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_list_nodes(self):
        self.client.graph.create("ln_g")
        self.client.graph.bulk_insert(
            "ln_g",
            nodes=[{"id": "a"}, {"id": "b"}, {"id": "c"}],
            edges=[],
        )
        nodes = self.client.graph.list_nodes("ln_g")
        self.assertIsInstance(nodes, list)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(sorted(nodes), ["a", "b", "c"])


@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestSearch(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.client = StrataClient(db_path=self._tmpdir.name, cache=True)

    def tearDown(self):
        self.client.close()
        self._tmpdir.cleanup()

    def test_search_keyword_returns_relevant_doc(self):
        self.client.kv.put("doc1", "the quick brown fox jumps over the lazy dog")
        self.client.kv.put("doc2", "a completely unrelated document about cars")
        self.client.flush()
        results = self.client.search("fox", k=5, mode="keyword", primitives=["kv"])
        self.assertIsInstance(results, list)
        if results:
            # doc1 should score higher for "fox"
            entities = [h["entity"] for h in results]
            self.assertIn("doc1", entities)

    def test_search_result_structure(self):
        """Each hit should have 'entity' and 'score' fields."""
        self.client.kv.put("d1", "test document")
        self.client.flush()
        results = self.client.search("test", k=5, mode="keyword", primitives=["kv"])
        for hit in results:
            self.assertIn("entity", hit)
            self.assertIn("score", hit)
            self.assertIsInstance(hit["score"], (int, float))

    def test_search_empty_db(self):
        self.client.flush()
        results = self.client.search("anything", k=5, mode="keyword", primitives=["kv"])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)


@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestFlush(unittest.TestCase):
    def test_flush_does_not_error(self):
        with tempfile.TemporaryDirectory() as d:
            with StrataClient(db_path=d, cache=True) as client:
                client.kv.put("k", "v")
                client.flush()  # Should not raise


@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestPing(unittest.TestCase):
    def test_ping_returns_version_string(self):
        with tempfile.TemporaryDirectory() as d:
            with StrataClient(db_path=d, cache=True) as client:
                version = client.ping()
                self.assertIsInstance(version, str)
                self.assertGreater(len(version), 0)


@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestContextManager(unittest.TestCase):
    def test_with_block(self):
        with tempfile.TemporaryDirectory() as d:
            with StrataClient(db_path=d, cache=True) as client:
                client.kv.put("ctx", "val")
                self.assertEqual(client.kv.get("ctx"), "val")

    def test_double_close(self):
        """Calling close() twice should not raise."""
        with tempfile.TemporaryDirectory() as d:
            client = StrataClient(db_path=d, cache=True)
            client.close()
            client.close()  # Should not raise


@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestErrorHandling(unittest.TestCase):
    def test_send_after_close_raises(self):
        with tempfile.TemporaryDirectory() as d:
            client = StrataClient(db_path=d, cache=True)
            client.close()
            with self.assertRaises(StrataError):
                client.kv.put("key", "value")

    def test_get_after_close_raises(self):
        with tempfile.TemporaryDirectory() as d:
            client = StrataClient(db_path=d, cache=True)
            client.close()
            with self.assertRaises(StrataError):
                client.kv.get("key")


@unittest.skipUnless(_have_strata(), "strata binary not found")
class TestThreadSafety(unittest.TestCase):
    def test_concurrent_kv_operations(self):
        """Multiple threads should be able to use the client safely."""
        import threading
        errors: list[Exception] = []

        with tempfile.TemporaryDirectory() as d:
            with StrataClient(db_path=d, cache=True) as client:
                # Pre-populate
                for i in range(50):
                    client.kv.put(f"t:{i}", f"v:{i}")

                def reader(start: int, count: int):
                    try:
                        for i in range(start, start + count):
                            client.kv.get(f"t:{i % 50}")
                    except Exception as e:
                        errors.append(e)

                threads = [
                    threading.Thread(target=reader, args=(0, 20)),
                    threading.Thread(target=reader, args=(10, 20)),
                    threading.Thread(target=reader, args=(20, 20)),
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=30)

                self.assertEqual(len(errors), 0, f"Thread errors: {errors}")


if __name__ == "__main__":
    unittest.main()
