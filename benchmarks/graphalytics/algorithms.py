"""Graph algorithm implementations for LDBC Graphalytics.

BFS and SSSP operate on the directed adjacency list.
WCC, LCC operate on the undirected adjacency list (both edge directions).
PageRank operates on the directed adjacency list (follows out-edges).
CDLP operates on the directed adjacency list.
"""

from __future__ import annotations

from collections import deque, Counter
import heapq
import math


# Sentinel for unreachable vertices (matches LDBC i64::MAX).
UNREACHABLE_INT = 2**63 - 1
UNREACHABLE_FLOAT = float("inf")


def bfs(adj: dict[int, list[int]], source: int) -> dict[int, int]:
    """BFS returning ``{vertex_id: depth}``.

    Unreachable vertices get depth ``2**63 - 1`` (LDBC convention).
    Operates on the provided adjacency list (directed or undirected).
    """
    depths: dict[int, int] = {v: UNREACHABLE_INT for v in adj}
    if source not in adj:
        return depths
    depths[source] = 0
    queue: deque[int] = deque([source])

    while queue:
        node = queue.popleft()
        d = depths[node]
        for neighbor in adj.get(node, []):
            if depths.get(neighbor, UNREACHABLE_INT) == UNREACHABLE_INT:
                depths[neighbor] = d + 1
                queue.append(neighbor)

    return depths


def wcc(adj: dict[int, list[int]]) -> dict[int, int]:
    """Weakly connected components.

    IMPORTANT: *adj* must be the **undirected** adjacency list (both edge
    directions included). The caller is responsible for passing the undirected
    view of the graph.

    Returns ``{vertex_id: component_id}`` where *component_id* is the
    smallest vertex ID in the component.
    """
    component: dict[int, int] = {}
    unvisited = set(adj.keys())

    while unvisited:
        # Pick an unvisited vertex
        start = min(unvisited)
        queue: deque[int] = deque([start])
        members: list[int] = []

        while queue:
            node = queue.popleft()
            if node not in unvisited:
                continue
            unvisited.discard(node)
            members.append(node)
            for neighbor in adj.get(node, []):
                if neighbor in unvisited:
                    queue.append(neighbor)

        comp_id = min(members)
        for m in members:
            component[m] = comp_id

    return component


def pagerank(
    adj: dict[int, list[int]],
    iterations: int = 20,
    damping: float = 0.85,
) -> dict[int, float]:
    """PageRank via power iteration.

    Operates on the **directed** adjacency list (out-edges).
    Returns ``{vertex_id: rank}``.
    """
    n = len(adj)
    if n == 0:
        return {}

    init_rank = 1.0 / n
    rank: dict[int, float] = {v: init_rank for v in adj}

    # Pre-compute out-degree for each vertex
    out_degree: dict[int, int] = {v: len(neighbors) for v, neighbors in adj.items()}

    # Build reverse adjacency (who points to me?)
    rev_adj: dict[int, list[int]] = {v: [] for v in adj}
    for v, neighbors in adj.items():
        for u in neighbors:
            if u in rev_adj:
                rev_adj[u].append(v)

    for _ in range(iterations):
        new_rank: dict[int, float] = {}
        # Dangling node contribution (nodes with out-degree 0)
        dangling_sum = sum(rank[v] for v in adj if out_degree[v] == 0)

        for v in adj:
            incoming = sum(
                rank[u] / out_degree[u]
                for u in rev_adj[v]
                if out_degree[u] > 0
            )
            new_rank[v] = (
                (1.0 - damping) / n
                + damping * (incoming + dangling_sum / n)
            )
        rank = new_rank

    return rank


def cdlp(adj: dict[int, list[int]], iterations: int = 10) -> dict[int, int]:
    """Community Detection via Label Propagation.

    Each vertex starts with its own ID as label.  Each iteration, a vertex
    adopts the most frequent label among its neighbors (tie-break: smallest
    label).  Returns ``{vertex_id: community_label}``.
    """
    label: dict[int, int] = {v: v for v in adj}

    for _ in range(iterations):
        new_label: dict[int, int] = {}
        for v in adj:
            neighbors = adj[v]
            if not neighbors:
                new_label[v] = label[v]
                continue
            # Count neighbor labels
            counts: Counter[int] = Counter(label[n] for n in neighbors)
            max_count = max(counts.values())
            # Among labels with max count, pick smallest
            new_label[v] = min(l for l, c in counts.items() if c == max_count)
        label = new_label

    return label


def lcc(adj: dict[int, list[int]]) -> dict[int, float]:
    """Local Clustering Coefficient per vertex.

    IMPORTANT: *adj* must be the **undirected** adjacency list with self-loops
    already removed and edges deduplicated. The caller is responsible for
    passing the cleaned undirected view.

    For vertex *v* with degree *d* and *t* triangles:
    ``lcc(v) = 2t / (d * (d - 1))`` if ``d >= 2``, else ``0``.

    Returns ``{vertex_id: coefficient}``.
    """
    coefficients: dict[int, float] = {}

    # Pre-compute neighbor sets for O(1) membership tests
    neighbor_sets: dict[int, set[int]] = {
        v: set(neighbors) for v, neighbors in adj.items()
    }

    for v in adj:
        neighbors = list(neighbor_sets[v])  # Use deduplicated set
        d = len(neighbors)
        if d < 2:
            coefficients[v] = 0.0
            continue

        # Count triangles: pairs of neighbors that are also connected
        triangles = 0
        for i in range(len(neighbors)):
            u = neighbors[i]
            u_neighbors = neighbor_sets.get(u, set())
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if w in u_neighbors:
                    triangles += 1

        coefficients[v] = (2.0 * triangles) / (d * (d - 1))

    return coefficients


def sssp(
    adj: dict[int, list[int]],
    weights: dict[tuple[int, int], float] | None,
    source: int,
) -> dict[int, float]:
    """Single-source shortest paths via Dijkstra.

    If *weights* is ``None``, all edges have weight 1.  Unreachable vertices
    get ``math.inf``.  Returns ``{vertex_id: distance}``.

    Note: Dijkstra requires non-negative edge weights.
    """
    dist: dict[int, float] = {v: UNREACHABLE_FLOAT for v in adj}
    if source not in adj:
        return dist
    dist[source] = 0.0
    # Priority queue: (distance, vertex)
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in adj.get(u, []):
            w = weights[(u, v)] if weights is not None else 1.0
            nd = d + w
            if nd < dist.get(v, UNREACHABLE_FLOAT):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist
