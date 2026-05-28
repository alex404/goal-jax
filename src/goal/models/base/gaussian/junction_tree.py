"""Junction-tree topology for tractable inference on bounded-treewidth graphical models.

Given an undirected graph on $n$ nodes, a ``JunctionTree`` triangulates it via the
min-fill heuristic, extracts the maximal cliques of the chordal completion, and
builds a maximum-weight spanning tree of the clique graph. The resulting topology
admits exact sum-product inference in $O(n \\cdot 2^{w+1})$ time, where $w$ is the
treewidth (the largest clique size minus one).

Mathematically, a junction tree of a chordal graph $G^\\star$ is a tree whose nodes
are the maximal cliques of $G^\\star$ and which satisfies the running-intersection
property: for any two cliques $C_i, C_j$, every clique on the path between them
contains $C_i \\cap C_j$. This is automatic for maximum-weight spanning trees of
the clique graph weighted by separator size.

The result is a frozen, hashable Python-level object holding all static topology.
Run-time inference (log-partition, sampling) takes the static topology and the
dynamic parameter vector and computes via sum-product on the tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

### Triangulation and clique extraction ###


def _min_fill_triangulation(  # noqa: C901
    n: int, edges: Sequence[tuple[int, int]]
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Triangulate the graph by greedy min-fill elimination.

    At each step pick the unmarked vertex whose elimination adds the fewest fill-in
    edges; record the perfect elimination order. Return the PEO and the full edge
    set of the chordal completion (originals plus fill-ins), as sorted (i, j) pairs
    with $i < j$.
    """
    active_adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        if a == b:
            continue
        active_adj[a].add(b)
        active_adj[b].add(a)
    chordal: set[tuple[int, int]] = set()
    for a, b in edges:
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        chordal.add((lo, hi))

    active = set(range(n))
    peo: list[int] = []

    while active:
        best_v = -1
        best_fill = -1
        for v in active:
            nbrs = sorted(active_adj[v])
            fill = 0
            for i, u in enumerate(nbrs):
                for w in nbrs[i + 1 :]:
                    if w not in active_adj[u]:
                        fill += 1
            if best_fill < 0 or fill < best_fill:
                best_fill = fill
                best_v = v

        v = best_v
        nbrs = sorted(active_adj[v])
        for i, u in enumerate(nbrs):
            for w in nbrs[i + 1 :]:
                if w not in active_adj[u]:
                    active_adj[u].add(w)
                    active_adj[w].add(u)
                    chordal.add((u, w) if u < w else (w, u))
        for u in nbrs:
            active_adj[u].discard(v)
        peo.append(v)
        active.remove(v)

    return tuple(peo), tuple(sorted(chordal))


def _maximal_cliques(
    n: int, peo: Sequence[int], chordal_edges: Sequence[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    """Extract maximal cliques from a chordal graph given its perfect elimination order."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in chordal_edges:
        adj[a].add(b)
        adj[b].add(a)
    pos = {v: i for i, v in enumerate(peo)}

    raw: list[tuple[int, ...]] = []
    for v in peo:
        later = {u for u in adj[v] if pos[u] > pos[v]}
        raw.append(tuple(sorted({v, *later})))

    sets = [frozenset(c) for c in raw]
    keep: list[tuple[int, ...]] = []
    seen: set[frozenset[int]] = set()
    for i, ci in enumerate(sets):
        if ci in seen:
            continue
        if any(j != i and ci < sets[j] for j in range(len(sets))):
            continue
        seen.add(ci)
        keep.append(raw[i])
    return tuple(keep)


def _max_spanning_tree(
    cliques: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, int], ...]:
    """Maximum-weight spanning tree of the clique graph, weighted by separator size.

    Returns tree edges as (i, j) pairs of clique indices.
    """
    nc = len(cliques)
    candidates: list[tuple[int, int, int]] = []
    for i in range(nc):
        for j in range(i + 1, nc):
            shared = len(set(cliques[i]) & set(cliques[j]))
            if shared:
                candidates.append((shared, i, j))
    candidates.sort(key=lambda x: -x[0])

    parent = list(range(nc))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: list[tuple[int, int]] = []
    for _w, i, j in candidates:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            edges.append((i, j))
            if len(edges) == nc - 1:
                break
    return tuple(edges)


def _root_and_collect_order(
    n_cliques: int, tree_edges: Sequence[tuple[int, int]], root: int
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Compute (parent_of, collect_order) for a rooted junction tree.

    parent_of[c] is the parent clique index of clique c (or -1 for the root).
    collect_order is a sequence of (child, parent) edges in a post-order walk
    (leaves processed before parents).
    """
    adj: list[list[int]] = [[] for _ in range(n_cliques)]
    for a, b in tree_edges:
        adj[a].append(b)
        adj[b].append(a)

    parent_of = [-1] * n_cliques
    order: list[int] = []
    visited = [False] * n_cliques
    stack: list[int] = [root]
    visited[root] = True
    while stack:
        v = stack.pop()
        order.append(v)
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                parent_of[u] = v
                stack.append(u)

    collect: list[tuple[int, int]] = [(c, parent_of[c]) for c in reversed(order) if parent_of[c] >= 0]
    return tuple(parent_of), tuple(collect)


### Public dataclass ###


@dataclass(frozen=True)
class JunctionTree:
    """Static topology for sum-product inference on a chordal graph.

    Built once at construction via min-fill triangulation, maximal-clique extraction,
    and Kruskal's maximum-weight spanning tree over the clique graph. The resulting
    object is hashable and freezable into JIT-compiled functions; only the parameter
    vector is array-traced at run time.

    Each bias parameter $\\theta_i$ and each chordal-edge coupling $\\theta_{ij}$ is
    attributed to *exactly one* clique that contains it (the first clique in
    canonical order), so each parameter contributes once across the tree.
    """

    # Fields

    n_neurons: int
    """Number of nodes (binary variables)."""

    cliques: tuple[tuple[int, ...], ...]
    """Maximal cliques of the chordal completion, each as a sorted tuple of node indices."""

    tree_edges: tuple[tuple[int, int], ...]
    """Junction-tree edges as (clique_i, clique_j) pairs."""

    chordal_edges: tuple[tuple[int, int], ...]
    """All edges of the chordal completion as sorted (i, j) pairs with $i < j$."""

    root: int
    """Index of the root clique used for message passing."""

    parent_of: tuple[int, ...]
    """parent_of[c] = parent clique index, or -1 for the root."""

    collect_order: tuple[tuple[int, int], ...]
    """Sequence of (child, parent) clique pairs in post-order — leaves to root."""

    bias_owner: tuple[int, ...]
    """For each node, the clique that owns its bias parameter."""

    edge_owner: tuple[int, ...]
    """For each chordal edge (in canonical order), the clique that owns its coupling."""

    chordal_edge_index: tuple[tuple[tuple[int, int], int], ...] = field(repr=False)
    """((i, j), edge_param_index) pairs, used to look up params for a given chordal edge."""

    # Constructors

    @staticmethod
    def from_edges(  # noqa: C901
        n: int,
        edges: Sequence[tuple[int, int]],
        max_treewidth: int | None = None,
    ) -> JunctionTree:
        """Build a junction tree from a graph on $n$ nodes given an edge list.

        Edges are undirected; self-loops are ignored. If ``max_treewidth`` is given,
        raise ``ValueError`` when the triangulation produces any clique of size
        greater than ``max_treewidth + 1``.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        peo, chordal_edges = _min_fill_triangulation(n, edges)
        cliques = _maximal_cliques(n, peo, chordal_edges)
        if not cliques:
            # Disconnected graph with isolated nodes: each node is its own clique
            cliques = tuple((i,) for i in range(n))
        max_size = max(len(c) for c in cliques)
        tw = max_size - 1
        if max_treewidth is not None and tw > max_treewidth:
            msg = (
                f"Triangulation produced treewidth {tw} > max_treewidth"
                f" {max_treewidth}; largest clique size {max_size}"
            )
            raise ValueError(msg)

        tree_edges = _max_spanning_tree(cliques)
        parent_of, collect_order = _root_and_collect_order(len(cliques), tree_edges, root=0)

        bias_owner: list[int] = [-1] * n
        for idx, c in enumerate(cliques):
            for v in c:
                if bias_owner[v] == -1:
                    bias_owner[v] = idx
        chordal_edge_index = tuple((e, k) for k, e in enumerate(chordal_edges))
        edge_to_idx = {e: k for e, k in chordal_edge_index}
        edge_owner = [-1] * len(chordal_edges)
        for idx, c in enumerate(cliques):
            for i_pos, vi in enumerate(c):
                for vj in c[i_pos + 1 :]:
                    key = (vi, vj)
                    k = edge_to_idx[key]
                    if edge_owner[k] == -1:
                        edge_owner[k] = idx

        return JunctionTree(
            n_neurons=n,
            cliques=cliques,
            tree_edges=tree_edges,
            chordal_edges=chordal_edges,
            root=0,
            parent_of=tuple(parent_of),
            collect_order=collect_order,
            bias_owner=tuple(bias_owner),
            edge_owner=tuple(edge_owner),
            chordal_edge_index=chordal_edge_index,
        )

    # Derived quantities

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
    def n_chordal_edges(self) -> int:
        return len(self.chordal_edges)

    @property
    def n_params(self) -> int:
        """Diagonal entries + chordal off-diagonal entries."""
        return self.n_neurons + self.n_chordal_edges

    def separator(self, child: int, parent: int) -> tuple[int, ...]:
        """Sorted tuple of node indices shared by the (child, parent) cliques."""
        return tuple(sorted(set(self.cliques[child]) & set(self.cliques[parent])))

    @cached_property
    def edge_param_index(self) -> dict[tuple[int, int], int]:
        """Map from chordal edge (i, j) with $i < j$ to its index in the off-diagonal params."""
        return {e: k for e, k in self.chordal_edge_index}

    @cached_property
    def chordal_edges_arr(self) -> np.ndarray:
        """Chordal edges as an (n_chordal_edges, 2) numpy int array, useful for vectorized indexing."""
        if not self.chordal_edges:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(self.chordal_edges, dtype=np.int32)

    # Padded static topology arrays for lax.scan-based inference.
    #
    # These are all shape-fixed per JunctionTree instance, computed once,
    # numpy-typed (so the JT remains hashable and JIT-friendly as static data).
    # Inference passes them in as constants and runs lax.scan over rows of the
    # collect / pre-order arrays.

    @cached_property
    def n_clique_states(self) -> int:
        """$2^{\\text{max\\_clique\\_size}}$ — padded number of states per clique."""
        return 1 << self.max_clique_size

    @cached_property
    def max_separator_size(self) -> int:
        """Max separator size across all tree edges (0 if there are no tree edges)."""
        if not self.collect_order:
            return 0
        return max(len(self.separator(c, p)) for c, p in self.collect_order)

    @cached_property
    def n_separator_states(self) -> int:
        """$2^{\\text{max\\_separator\\_size}}$ — padded number of states per separator."""
        return 1 << self.max_separator_size

    @cached_property
    def bits_table(self) -> np.ndarray:
        """Static bit table ``(n_clique_states, max_clique_size)``.

        Row $s$ is the binary expansion of $s$ (LSB first) over ``max_clique_size`` bits.
        Shared across cliques: only the first $|C|$ bits of each row are interpreted
        per clique, and the ``clique_state_mask`` filters padded states.
        """
        k = self.max_clique_size
        states = np.arange(1 << k, dtype=np.int64)
        return ((states[:, None] >> np.arange(k)) & 1).astype(np.int32)

    @cached_property
    def clique_state_mask(self) -> np.ndarray:
        """``(n_cliques, n_clique_states)`` bool array.

        ``mask[c, s]`` is True iff state $s < 2^{|C_c|}$ — i.e. $s$ is a real
        assignment for clique $c$, not a padded slot.
        """
        sizes = np.array([len(c) for c in self.cliques], dtype=np.int64)  # (n_cliques,)
        s = np.arange(self.n_clique_states, dtype=np.int64)  # (n_clique_states,)
        return s[None, :] < (1 << sizes[:, None])

    @cached_property
    def owned_bias_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-clique padded bias-ownership arrays.

        Returns ``(bias_param_idx, bias_local_pos)`` each of shape
        ``(n_cliques, max_owned_biases)``. Sentinel ``-1`` in ``bias_param_idx``
        marks padded slots. ``max_owned_biases`` equals ``max_clique_size``
        (a clique can own at most as many biases as it has nodes).
        """
        max_obi = self.max_clique_size
        owners: list[list[int]] = [[] for _ in range(self.n_cliques)]
        positions: list[list[int]] = [[] for _ in range(self.n_cliques)]
        for v, c_idx in enumerate(self.bias_owner):
            owners[c_idx].append(v)
            positions[c_idx].append(self.cliques[c_idx].index(v))
        idx = np.full((self.n_cliques, max_obi), -1, dtype=np.int32)
        pos = np.zeros((self.n_cliques, max_obi), dtype=np.int32)
        for c, (owned_v, owned_p) in enumerate(zip(owners, positions)):
            for slot, (v, p) in enumerate(zip(owned_v, owned_p)):
                idx[c, slot] = v
                pos[c, slot] = p
        return idx, pos

    @cached_property
    def owned_edge_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-clique padded edge-ownership arrays.

        Returns ``(edge_param_idx, edge_local_pos_i, edge_local_pos_j)`` each of
        shape ``(n_cliques, max_owned_edges)``. Sentinel ``-1`` in
        ``edge_param_idx`` marks padded slots. ``max_owned_edges`` is the max
        number of chordal edges any single clique owns; an upper bound is
        ``max_clique_size * (max_clique_size - 1) // 2``.
        """
        per_clique: list[list[tuple[int, int, int]]] = [[] for _ in range(self.n_cliques)]
        edge_to_idx = self.edge_param_index
        for k, (vi, vj) in enumerate(self.chordal_edges):
            owner = self.edge_owner[k]
            clique = self.cliques[owner]
            pi = clique.index(vi)
            pj = clique.index(vj)
            per_clique[owner].append((k, pi, pj))
        max_oei = max((len(x) for x in per_clique), default=0)
        if max_oei == 0:
            return (
                np.full((self.n_cliques, 0), -1, dtype=np.int32),
                np.zeros((self.n_cliques, 0), dtype=np.int32),
                np.zeros((self.n_cliques, 0), dtype=np.int32),
            )
        eidx = np.full((self.n_cliques, max_oei), -1, dtype=np.int32)
        ei = np.zeros((self.n_cliques, max_oei), dtype=np.int32)
        ej = np.zeros((self.n_cliques, max_oei), dtype=np.int32)
        for c, edges_in_c in enumerate(per_clique):
            for slot, (k, pi, pj) in enumerate(edges_in_c):
                eidx[c, slot] = k
                ei[c, slot] = pi
                ej[c, slot] = pj
        del edge_to_idx  # only used to assert presence
        return eidx, ei, ej

    @cached_property
    def collect_step_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-tree-edge data for the collect ``lax.scan``.

        Returns ``(child_idx, parent_idx, sep_maps)``:

        - ``child_idx``, ``parent_idx``: shape ``(n_tree_edges,)`` int.
        - ``sep_maps``: shape ``(n_tree_edges, 2, n_clique_states)``. For tree
          edge $e$, ``sep_maps[e, 0, s]`` is the separator state corresponding
          to clique state $s$ in the *child* clique; ``sep_maps[e, 1, s]`` is
          the same for the *parent* clique. Padded clique states are mapped to
          $0$ (an always-valid separator-state index) so that ``segment_max``
          over a real separator state never sees an all-padded segment.
        """
        n_edges = len(self.collect_order)
        child_idx = np.zeros(n_edges, dtype=np.int32)
        parent_idx = np.zeros(n_edges, dtype=np.int32)
        sep_maps = np.zeros((n_edges, 2, self.n_clique_states), dtype=np.int32)
        for e, (child, parent) in enumerate(self.collect_order):
            child_idx[e] = child
            parent_idx[e] = parent
            sep = self.separator(child, parent)
            sep_maps[e, 0] = _build_sep_state_map(
                self.cliques[child], sep, self.max_clique_size
            )
            sep_maps[e, 1] = _build_sep_state_map(
                self.cliques[parent], sep, self.max_clique_size
            )
        return child_idx, parent_idx, sep_maps

    @cached_property
    def pre_order(self) -> tuple[int, ...]:
        """Cliques in pre-order from the root (parent before its children)."""
        children: list[list[int]] = [[] for _ in range(self.n_cliques)]
        for c, p in enumerate(self.parent_of):
            if p >= 0:
                children[p].append(c)
        order: list[int] = []
        stack: list[int] = [self.root]
        while stack:
            v = stack.pop()
            order.append(v)
            for u in children[v]:
                stack.append(u)
        return tuple(order)

    @cached_property
    def walk_step_arrays(self) -> dict[str, np.ndarray]:
        """Per-clique data for the walk-down sampling ``lax.scan``.

        Iterates cliques in :attr:`pre_order`. Each entry packs everything one
        scan step needs: which clique to visit, its parent (or $-1$ for the root),
        the separator map from clique state to parent's separator state, and
        a padded list of global variable indices to decode the sampled state.

        Returns a dict with keys ``clique_idx``, ``sep_map``, ``sep_global_vars``,
        ``sep_local_positions``, ``sep_size``, ``clique_global_vars``,
        ``clique_size``.
        """
        n = self.n_cliques
        m = self.max_clique_size
        clique_idx = np.zeros(n, dtype=np.int32)
        sep_map = np.zeros((n, self.n_clique_states), dtype=np.int32)
        sep_global_vars = np.zeros((n, m), dtype=np.int32)
        sep_size = np.zeros(n, dtype=np.int32)
        clique_global_vars = np.zeros((n, m), dtype=np.int32)
        clique_size = np.zeros(n, dtype=np.int32)
        for step, c in enumerate(self.pre_order):
            clique_idx[step] = c
            cl = self.cliques[c]
            clique_size[step] = len(cl)
            for i, v in enumerate(cl):
                clique_global_vars[step, i] = v
            p = self.parent_of[c]
            if p >= 0:
                sep = self.separator(c, p)
                sep_size[step] = len(sep)
                for i, v in enumerate(sep):
                    sep_global_vars[step, i] = v
                sep_map[step] = _build_sep_state_map(cl, sep, m)
            # Root: sep_size stays 0, sep_map stays zeros, sep_global_vars stays zeros
        return {
            "clique_idx": clique_idx,
            "sep_map": sep_map,
            "sep_global_vars": sep_global_vars,
            "sep_size": sep_size,
            "clique_global_vars": clique_global_vars,
            "clique_size": clique_size,
        }


def _build_sep_state_map(
    clique: tuple[int, ...], separator: tuple[int, ...], max_clique_size: int
) -> np.ndarray:
    """Map each clique state in $[0, 2^{\\text{max\\_clique\\_size}})$ to its separator state.

    For a real state $s < 2^{|C|}$, the separator state is obtained by reading off
    the bits of $s$ at the local positions of separator nodes within ``clique``
    and packing them LSB-first. For padded states $s \\geq 2^{|C|}$, the map
    returns $0$ — an always-valid separator-state index. This guarantees that
    ``segment_logsumexp`` over the real-state values doesn't see an empty
    segment (every separator state has at least one real contribution from the
    consistent clique states, and padded -inf values only get absorbed into
    segments that already contain finite values).
    """
    n_states = 1 << max_clique_size
    real_states = 1 << len(clique)
    result = np.zeros(n_states, dtype=np.int32)
    if not separator:
        # Empty separator: single segment, all real states map to 0.
        return result
    sep_positions = [clique.index(v) for v in separator]
    states = np.arange(real_states, dtype=np.int64)
    sep_state = np.zeros_like(states)
    for i, pos in enumerate(sep_positions):
        bit = (states >> pos) & 1
        sep_state |= bit << i
    result[:real_states] = sep_state.astype(np.int32)
    return result
