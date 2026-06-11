"""Junction-tree topology for exact inference on sparse graphical models.

Exact inference on a discrete graphical model --- computing its partition
function, or sampling from it --- costs $O(2^n)$ in general, but far less when
the interaction graph is sparse. The classical route to exploiting sparsity is
the junction tree: group the variables into overlapping clusters (cliques) and
arrange the clusters in a tree, so that dynamic programming over the tree
(sum-product message passing) yields exact answers at a cost exponential only
in the largest cluster size --- the *treewidth* --- rather than in $n$. Chains,
trees, and band-like topologies have small treewidth, so models built on them
remain exactly tractable at sizes where brute force is hopeless.

This module builds that structure. Given an undirected graph on $n$ nodes,
:meth:`JunctionTree.from_edges` triangulates it via the min-fill heuristic,
extracts the maximal cliques of the chordal completion, and builds a
maximum-weight spanning tree of the clique graph. The resulting topology
admits exact sum-product inference in $O(n \\cdot 2^{w+1})$ time for binary
variables, where $w$ is the treewidth (the largest clique size minus one).

Note that the input graph is a *seed*, not a topology the tree preserves:
triangulation may add fill-in edges, and the junction tree describes the
chordal completion with no memory of which edges were original. Models built
on a :class:`JunctionTree` (e.g. ``ChordalBoltzmann``) parameterize every
edge of the completion, fill-ins included --- the module is a chordal-model
*construction* system, not an exact-inference layer for arbitrary input
graphs. Seeds that are already chordal (chains, trees, band graphs) incur no
fill-in, so for them the completion is the input graph.

Mathematically, a junction tree of a chordal graph $G^\\star$ is a tree whose nodes
are the maximal cliques of $G^\\star$ and which satisfies the running-intersection
property: for any two cliques $C_i, C_j$, every clique on the path between them
contains $C_i \\cap C_j$. This is automatic for maximum-weight spanning trees of
the clique graph weighted by separator size.

The result is a frozen, hashable Python-level object whose fields are the tree's
identity; all further topology (separators, message schedules, parameter
ownership) is derived on demand. Run-time inference
(:mod:`~goal.models.base.gaussian.coupling.sum_product`) takes the static
topology and the dynamic parameter vector and computes via sum-product on the
tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import numpy as np

### Construction phases of JunctionTree.from_edges ###


def _triangulate(
    n: int, edges: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, list[int]]:
    """Greedy min-fill elimination: chordal completion and elimination order.

    ``adj`` is the evolving elimination graph (eliminated rows are zeroed, so
    neighbourhoods are always active); ``chordal`` accumulates the original
    edges plus all fill-ins --- the chordal completion.
    """
    adj = np.zeros((n, n), dtype=bool)
    for a, b in edges:
        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"edge ({a}, {b}) out of range for n={n}")
        if a != b:
            adj[a, b] = adj[b, a] = True
    chordal = adj.copy()
    active = np.ones(n, dtype=bool)

    def fill_in(v: int) -> int:
        nbrs = np.flatnonzero(adj[v])
        k = nbrs.size
        return k * (k - 1) // 2 - int(np.count_nonzero(adj[np.ix_(nbrs, nbrs)])) // 2

    fill = np.array([fill_in(v) for v in range(n)])
    peo: list[int] = []
    for _ in range(n):
        cand = np.flatnonzero(active)
        # argmin takes the first minimum --- deterministic lowest-index tie-break.
        v = int(cand[np.argmin(fill[cand])])
        nbrs = np.flatnonzero(adj[v])
        # Connect all pairs of v's neighbours. Fill counts change only for
        # vertices whose neighbourhood gained an internal edge: v's
        # neighbours, plus vertices adjacent to both endpoints of a fill-in.
        missing = ~adj[np.ix_(nbrs, nbrs)]
        touched = set(nbrs.tolist())
        for pa, pb in np.argwhere(np.triu(missing, 1)):
            a, b = int(nbrs[pa]), int(nbrs[pb])
            adj[a, b] = adj[b, a] = True
            chordal[a, b] = chordal[b, a] = True
            touched.update(np.flatnonzero(adj[a] & adj[b]).tolist())
        active[v] = False
        adj[v, :] = adj[:, v] = False
        peo.append(v)
        for u in touched:
            if active[u]:
                fill[u] = fill_in(u)
    return chordal, peo


def _maximal_cliques(
    chordal: np.ndarray, peo: list[int]
) -> tuple[tuple[tuple[int, ...], ...], list[list[int]]]:
    """Maximal cliques of the chordal graph, plus the per-node clique index.

    The candidate at $v$ is $\\{v\\}$ + later neighbours; a non-maximal
    candidate is contained in the (kept) candidate of an earlier-eliminated
    vertex that contains $v$, so containment only needs checking there.
    """
    n = len(peo)
    pos = np.empty(n, dtype=np.int64)
    pos[np.array(peo)] = np.arange(n)
    kept: list[tuple[int, ...]] = []
    kept_sets: list[set[int]] = []
    node_cliques: list[list[int]] = [[] for _ in range(n)]
    for v in peo:
        cset = {v, *np.flatnonzero(chordal[v] & (pos > pos[v])).tolist()}
        if any(cset <= kept_sets[k] for k in node_cliques[v]):
            continue
        for u in cset:
            node_cliques[u].append(len(kept))
        kept.append(tuple(sorted(cset)))
        kept_sets.append(cset)
    return tuple(kept), node_cliques


def _spanning_tree(
    n_cliques: int, node_cliques: list[list[int]]
) -> tuple[tuple[int, int], ...]:
    """Maximum-weight spanning tree of the clique graph, linked across components.

    Separator sizes via co-occurrence counts: each node contributes 1 to the
    weight of every pair of cliques containing it. Kruskal on the weights,
    then zero-weight links join any remaining components (a disconnected
    graph yields a clique forest) into a single tree.
    """
    weights: dict[tuple[int, int], int] = {}
    for ids in node_cliques:
        for x, ci in enumerate(ids):
            for cj in ids[x + 1 :]:
                key = (ci, cj) if ci < cj else (cj, ci)
                weights[key] = weights.get(key, 0) + 1

    parent = list(range(n_cliques))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: list[tuple[int, int]] = []
    for (ci, cj), _w in sorted(weights.items(), key=lambda kv: -kv[1]):
        ri, rj = find(ci), find(cj)
        if ri != rj:
            parent[ri] = rj
            tree.append((ci, cj))
    for c in range(1, n_cliques):
        rc, r0 = find(c), find(0)
        if rc != r0:
            parent[rc] = r0
            tree.append((0, c))
    return tuple(tree)


@dataclass(frozen=True)
class JunctionTree:
    """Static junction-tree topology for sum-product inference on a chordal graph.

    The four fields are the identity of the tree --- what ``__eq__`` / ``__hash__``
    compare and what JIT treats as the static cache key; everything else is derived
    on demand. Construct via :meth:`from_edges`; the raw constructor performs no
    validation.

    Each bias parameter $\\theta_i$ and each chordal-edge coupling $\\theta_{ij}$ is
    attributed to *exactly one* clique that contains it (the first clique in
    canonical order), so each parameter contributes once across the tree.
    """

    # Fields

    n_nodes: int
    """Number of nodes (variables) in the underlying graph."""

    cliques: tuple[tuple[int, ...], ...]
    """Maximal cliques of the chordal completion, each as a sorted tuple of node indices."""

    tree_edges: tuple[tuple[int, int], ...]
    """Junction-tree edges as (clique_i, clique_j) pairs; always a single spanning tree."""

    root: int
    """Index of the root clique used for message passing."""

    # Constructors

    @staticmethod
    def from_edges(
        n: int,
        edges: Sequence[tuple[int, int]],
        max_treewidth: int | None = None,
    ) -> JunctionTree:
        """Build a junction tree from an undirected graph on ``n`` nodes.

        The edge list is a *seed*, not the final topology: min-fill
        triangulation may add fill-in edges, and the resulting tree describes
        the full chordal completion --- downstream models parameterize every
        chordal edge, fill-ins included. The original edge list is not
        retained; consumers that need it must keep their own copy.

        Edges are undirected; self-loops are ignored; out-of-range endpoints
        raise ``ValueError``. The graph may be disconnected: clique-forest
        components are linked by empty-separator tree edges, over which
        sum-product correctly multiplies the per-component partition
        functions. Construction is deterministic (min-fill ties break toward
        the lowest vertex index). If ``max_treewidth`` is given, raise
        ``ValueError`` when the min-fill triangulation exceeds it --- a
        heuristic bound: the graph may still admit a smaller-treewidth
        triangulation that min-fill did not find.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        chordal, peo = _triangulate(n, edges)
        cliques, node_cliques = _maximal_cliques(chordal, peo)

        max_size = max(len(c) for c in cliques)
        if max_treewidth is not None and max_size - 1 > max_treewidth:
            msg = (
                f"min-fill triangulation produced treewidth {max_size - 1} > "
                f"max_treewidth {max_treewidth} (largest clique size {max_size})"
            )
            raise ValueError(msg)

        tree = _spanning_tree(len(cliques), node_cliques)
        return JunctionTree(n_nodes=n, cliques=cliques, tree_edges=tree, root=0)

    # Derived topology

    @property
    def n_cliques(self) -> int:
        return len(self.cliques)

    @property
    def max_clique_size(self) -> int:
        return max(len(c) for c in self.cliques)

    @property
    def treewidth(self) -> int:
        return self.max_clique_size - 1

    @property
    def chordal_edges(self) -> tuple[tuple[int, int], ...]:
        """All edges of the chordal completion as sorted (i, j) pairs with $i < j$.

        The position of an edge in this tuple is its canonical index into the
        off-diagonal parameter vector.
        """
        pairs = {
            (vi, vj) for c in self.cliques for x, vi in enumerate(c) for vj in c[x + 1 :]
        }
        return tuple(sorted(pairs))

    @property
    def n_chordal_edges(self) -> int:
        return len(self.chordal_edges)

    @property
    def n_params(self) -> int:
        """Diagonal entries + chordal off-diagonal entries."""
        return self.n_nodes + self.n_chordal_edges

    @property
    def chordal_edges_arr(self) -> np.ndarray:
        """Chordal edges as an (n_chordal_edges, 2) numpy int array for vectorized indexing."""
        if not self.chordal_edges:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(self.chordal_edges, dtype=np.int32)

    @property
    def pre_order(self) -> tuple[int, ...]:
        """Cliques in depth-first pre-order from the root (parent before children)."""
        adj: list[list[int]] = [[] for _ in range(self.n_cliques)]
        for a, b in self.tree_edges:
            adj[a].append(b)
            adj[b].append(a)
        order: list[int] = []
        seen = {self.root}
        stack = [self.root]
        while stack:
            c = stack.pop()
            order.append(c)
            for u in adj[c]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        return tuple(order)

    @property
    def parent_of(self) -> tuple[int, ...]:
        """parent_of[c] = parent clique index in the rooted tree, or -1 for the root."""
        pos = {c: i for i, c in enumerate(self.pre_order)}
        parent = [-1] * self.n_cliques
        for a, b in self.tree_edges:
            a, b = (a, b) if pos[a] < pos[b] else (b, a)
            parent[b] = a
        return tuple(parent)

    @property
    def collect_order(self) -> tuple[tuple[int, int], ...]:
        """(child, parent) clique pairs in post-order --- leaves before the root."""
        parent = self.parent_of
        return tuple(
            (c, parent[c]) for c in reversed(self.pre_order) if parent[c] >= 0
        )

    @property
    def bias_owner(self) -> tuple[int, ...]:
        """For each node, the first clique (in canonical order) that contains it."""
        owner = [-1] * self.n_nodes
        for ci, c in enumerate(self.cliques):
            for v in c:
                if owner[v] < 0:
                    owner[v] = ci
        return tuple(owner)

    @property
    def edge_owner(self) -> tuple[int, ...]:
        """For each chordal edge (canonical order), the first clique that contains it."""
        index = {e: k for k, e in enumerate(self.chordal_edges)}
        owner = [-1] * len(index)
        for ci, c in enumerate(self.cliques):
            for x, vi in enumerate(c):
                for vj in c[x + 1 :]:
                    k = index[(vi, vj)]
                    if owner[k] < 0:
                        owner[k] = ci
        return tuple(owner)

    # Methods

    def separator(self, child: int, parent: int) -> tuple[int, ...]:
        """Sorted tuple of node indices shared by the (child, parent) cliques."""
        return tuple(sorted(set(self.cliques[child]) & set(self.cliques[parent])))


@dataclass(frozen=True)
class ChainTree(JunctionTree):
    """Junction tree whose cliques form a path --- a *path decomposition*.

    Reifies the precondition of the transfer-matrix inference kernel
    (:func:`~goal.models.base.gaussian.coupling.sum_product.chain_log_partition`):
    every clique has at most two tree neighbours, so messages compose along a
    single sequence. The underlying graphs are interval graphs of bounded
    *pathwidth* --- chains, band graphs, triangulated cycles. Construction
    validates the path shape and raises ``ValueError`` when the clique tree
    branches, so holders of a ``ChainTree`` are guaranteed the fast regime;
    there is no silent fallback in either direction.

    Validation checks the clique tree actually built: some interval graphs
    admit a path arrangement that the spanning-tree construction does not find
    (proper recognition is PQ-tree territory). Chains and bands are always
    recognized.
    """

    def __post_init__(self) -> None:
        # Assumes the field invariant that tree_edges is a single spanning
        # tree (raw construction is unvalidated, as in the base class); for a
        # tree, max degree <= 2 is equivalent to being a path.
        degree = [0] * self.n_cliques
        for a, b in self.tree_edges:
            degree[a] += 1
            degree[b] += 1
        if any(d > 2 for d in degree):
            msg = (
                "Clique tree is not a path (a clique has more than two "
                "neighbours); use JunctionTree / ChordalBoltzmann for "
                "branching topologies"
            )
            raise ValueError(msg)

    # Constructors

    @staticmethod
    @override
    def from_edges(
        n: int,
        edges: Sequence[tuple[int, int]],
        max_treewidth: int | None = None,
    ) -> ChainTree:
        """Build a chain tree from an undirected graph on ``n`` nodes.

        Same construction as :meth:`JunctionTree.from_edges`, plus the
        path-shape validation.
        """
        jt = JunctionTree.from_edges(n, edges, max_treewidth)
        return ChainTree(jt.n_nodes, jt.cliques, jt.tree_edges, jt.root)

    # Derived topology

    @property
    def clique_path(self) -> tuple[int, ...]:
        """Cliques ordered leaf-to-leaf along the path."""
        if self.n_cliques == 1:
            return (0,)
        nbrs: list[list[int]] = [[] for _ in range(self.n_cliques)]
        for a, b in self.tree_edges:
            nbrs[a].append(b)
            nbrs[b].append(a)
        order = [min(c for c in range(self.n_cliques) if len(nbrs[c]) == 1)]
        prev = -1
        while len(order) < self.n_cliques:
            cur = order[-1]
            order.append(next(u for u in nbrs[cur] if u != prev))
            prev = cur
        return tuple(order)
