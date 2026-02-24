"""LDBC Graphalytics dataset parser.

Parses the standard LDBC file formats:
- ``.v``  -- one vertex ID (int) per line
- ``.e``  -- ``src dst [weight]`` per line (space-separated)
- ``.properties`` -- Java properties format with graph metadata
- reference output -- ``vertex_id value`` per line

Ported from the Rust implementation in strata-benchmarks/benches/graph/ldbc.rs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# File parsers
# ---------------------------------------------------------------------------

def load_properties(path: Path | str) -> dict[str, str]:
    """Parse a ``.properties`` file (Java properties: key=value).

    Lines starting with ``#`` are comments and are skipped.
    Returns a dict with keys like ``graph.name``, ``graph.directed``, etc.
    """
    path = Path(path)
    props: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
    return props


def load_vertices(path: Path | str) -> list[int]:
    """Parse a ``.v`` file -- one vertex ID per line."""
    path = Path(path)
    vertices: list[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            vertices.append(int(line))
    return vertices


def load_edges(path: Path | str) -> list[tuple[int, int, float]]:
    """Parse a ``.e`` file -- ``src dst [weight]`` per line (space-separated).

    Returns list of (src, dst, weight) tuples. Weight defaults to 1.0 if
    not present in the file.
    """
    path = Path(path)
    edges: list[tuple[int, int, float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            edges.append((int(parts[0]), int(parts[1]), 1.0))
        elif len(parts) == 3:
            edges.append((int(parts[0]), int(parts[1]), float(parts[2])))
        else:
            raise ValueError(f"bad edge line (expected 2 or 3 fields): '{line}'")
    return edges


def load_reference(path: Path | str) -> dict[int, str]:
    """Parse a reference output file -- ``vertex_id value`` per line.

    Returns ``{vertex_id: value_string}`` where the value is kept as a raw
    string so the caller can parse it according to the algorithm type
    (int for BFS/WCC/CDLP, float for PageRank/LCC/SSSP).
    """
    path = Path(path)
    ref: dict[int, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"bad reference line: '{line}'")
        ref[int(parts[0])] = parts[1]
    return ref


# ---------------------------------------------------------------------------
# Adjacency builder
# ---------------------------------------------------------------------------

def build_adjacency(
    vertices: list[int],
    edges: list[tuple[int, int, float]],
    directed: bool,
) -> dict[int, list[int]]:
    """Build an adjacency list dict from vertices and edges.

    If *directed* is ``False``, both directions are added for each edge.
    Every vertex in *vertices* is guaranteed to appear as a key even if it
    has no edges.
    """
    adj: dict[int, list[int]] = {v: [] for v in vertices}
    for src, dst, _weight in edges:
        adj.setdefault(src, []).append(dst)
        if not directed:
            adj.setdefault(dst, []).append(src)
    return adj


def build_undirected_adjacency(
    vertices: list[int],
    edges: list[tuple[int, int, float]],
) -> dict[int, list[int]]:
    """Build an undirected adjacency list, deduplicating and removing self-loops.

    Used by WCC, LCC, and other algorithms that operate on the underlying
    undirected graph regardless of the input graph's directedness.
    """
    adj: dict[int, set[int]] = {v: set() for v in vertices}
    for src, dst, _weight in edges:
        if src == dst:
            continue  # Skip self-loops
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)
    # Convert sets to sorted lists for deterministic behavior
    return {v: sorted(neighbors) for v, neighbors in adj.items()}


def build_edge_weights(
    edges: list[tuple[int, int, float]],
) -> dict[tuple[int, int], float]:
    """Build an edge weight map from the edge list."""
    weights: dict[tuple[int, int], float] = {}
    for src, dst, weight in edges:
        weights[(src, dst)] = weight
    return weights


# ---------------------------------------------------------------------------
# Dataset dataclass
# ---------------------------------------------------------------------------

@dataclass
class LdbcDataset:
    """An LDBC Graphalytics dataset (vertices + edges + metadata)."""

    name: str
    directed: bool
    vertices: list[int]
    edges: list[tuple[int, int, float]]
    adj: dict[int, list[int]]              # directed adjacency (as-is)
    undirected_adj: dict[int, list[int]]   # undirected, deduplicated, no self-loops
    edge_weights: dict[tuple[int, int], float]
    properties: dict[str, str]
    data_dir: Path


def load_dataset(data_dir: Path | str) -> LdbcDataset:
    """Load all files from an LDBC dataset directory and validate counts.

    Expects files named ``<name>.v``, ``<name>.e``, and optionally
    ``<name>.properties`` where ``<name>`` is the directory's basename.
    """
    data_dir = Path(data_dir)
    name = data_dir.name

    v_path = data_dir / f"{name}.v"
    e_path = data_dir / f"{name}.e"
    props_path = data_dir / f"{name}.properties"

    vertices = load_vertices(v_path)
    edges = load_edges(e_path)

    # Parse properties (optional)
    directed = True
    properties: dict[str, str] = {}
    if props_path.exists():
        properties = load_properties(props_path)
        if properties.get("graph.directed") == "false":
            directed = False

        # Validate counts if properties file provides them
        expected_v = properties.get("meta.vertices")
        if expected_v is not None:
            expected_v_int = int(expected_v)
            if len(vertices) != expected_v_int:
                raise ValueError(
                    f"vertex count mismatch: file has {len(vertices)}, "
                    f"properties says {expected_v_int}"
                )

        expected_e = properties.get("meta.edges")
        if expected_e is not None:
            expected_e_int = int(expected_e)
            if len(edges) != expected_e_int:
                raise ValueError(
                    f"edge count mismatch: file has {len(edges)}, "
                    f"properties says {expected_e_int}"
                )

    adj = build_adjacency(vertices, edges, directed)
    undirected_adj = build_undirected_adjacency(vertices, edges)
    edge_weights = build_edge_weights(edges)

    return LdbcDataset(
        name=name,
        directed=directed,
        vertices=vertices,
        edges=edges,
        adj=adj,
        undirected_adj=undirected_adj,
        edge_weights=edge_weights,
        properties=properties,
        data_dir=data_dir,
    )
