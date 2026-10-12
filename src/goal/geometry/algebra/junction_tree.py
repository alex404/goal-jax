"""Junction trees: the cliques of a chordal graph arranged in a tree, with exact sum-product over binary variables.

Exact inference on a discrete graphical model --- computing its partition
function, or sampling from it --- costs $O(2^n)$ in general, but far less when
the interaction graph is sparse. The classical route to exploiting sparsity is
the junction tree: group the variables into overlapping clusters (cliques) and
arrange the clusters in a tree, so that dynamic programming over the tree
(sum-product message passing) yields exact answers at a cost exponential only
in the largest cluster size --- the *treewidth* --- rather than in $n$. Chains,
trees, and band-like topologies have small treewidth, so models built on them
remain exactly tractable at sizes where brute force is hopeless.

:meth:`JunctionTree.from_edges` builds the structure from an undirected graph on $n$
nodes: it triangulates the graph via the min-fill heuristic, extracts the maximal
cliques of the chordal completion, and builds a maximum-weight spanning tree of the
clique graph. The input graph is a *seed*, not a topology the tree preserves:
triangulation may add fill-in edges, and the junction tree describes the chordal
completion with no memory of which edges were original. Seeds that are already chordal
(chains, trees, band graphs) incur no fill-in, so for them the completion is the input
graph.

Inference runs on a table of clique log-potentials ``phi`` of shape
``(n_cliques, 2^m)``, with $m$ the size of the largest clique: row $c$ holds
$\\phi_c(x_C)$ for every binary state of clique $C$, and the distribution is
$p(x) \\propto \\exp \\sum_c \\phi_c(x_C)$. A state is encoded LSB-first: bit $p$ of a
clique state is the value of the clique's $p$-th node, in ascending node order, as in
:func:`binary_states`. Entries at states $s \\geq 2^{\\lvert C \\rvert}$ of a smaller clique are
padding and are ignored. :meth:`JunctionTree.log_partition` and
:meth:`JunctionTree.sample` cost $O(n \\cdot 2^{w+1})$, with $w$ the treewidth;
a :class:`ChainTree` runs the same computations at $O(\\log n)$ depth.

All static index tables (binary state tables, separator-state maps, traversal
arrays) are built with numpy at trace time from the topology --- under ``jax.jit``
this runs once per compilation, so the compilation cache is the only cache. The
potential-dependent arithmetic runs through ``jax.lax.scan`` or
``jax.lax.associative_scan`` over those tables, so the staged program is not unrolled
over cliques.

Mathematically, a junction tree of a chordal graph $G^\\star$ is a tree whose nodes
are the maximal cliques of $G^\\star$ and which satisfies the running-intersection
property: for any two cliques $C_i, C_j$, every clique on the path between them
contains $C_i \\cap C_j$. This is automatic for maximum-weight spanning trees of
the clique graph weighted by separator size.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .clique import Cliques

### State tables ###


def binary_states(n: int) -> np.ndarray:
    """``(2^n, n)`` table of all binary states: row $s$ is the LSB-first binary expansion of $s$."""
    states = np.arange(1 << n, dtype=np.int64)
    return ((states[:, None] >> np.arange(n)) & 1).astype(np.int32)


def _sep_state_map(
    clique: tuple[int, ...], separator: tuple[int, ...], m: int
) -> np.ndarray:
    """Map each (padded) clique state to its separator state.

    For a real state $s < 2^{|C|}$ the separator state packs the bits of $s$ at
    the separator nodes' local positions, LSB-first. Padded states ($s \\geq
    2^{|C|}$) map to separator state $0$, which always has at least one real
    contribution, so segment reductions never see an all-padded segment.
    """
    result = np.zeros(1 << m, dtype=np.int32)
    if not separator:
        return result
    states = np.arange(1 << len(clique), dtype=np.int64)
    sep_state = np.zeros_like(states)
    for i, v in enumerate(separator):
        sep_state |= ((states >> clique.index(v)) & 1) << i
    result[: 1 << len(clique)] = sep_state.astype(np.int32)
    return result


def _state_mask(jt: JunctionTree) -> np.ndarray:
    """``(n_cliques, 2^m)`` bool: True iff the state is a real assignment for the clique."""
    sizes = np.array([len(c) for c in jt.cliques], dtype=np.int64)
    states = np.arange(1 << jt.max_clique_size, dtype=np.int64)
    return states[None, :] < (1 << sizes[:, None])


### Traced computations ###


def _masked(jt: JunctionTree, phi: Array) -> Array:
    """The potentials with padded states set to $-\\infty$, so they fall out of every ``logsumexp``."""
    return jnp.where(jnp.asarray(_state_mask(jt)), phi, -jnp.inf)


def _segment_logsumexp(values: Array, segment_ids: Array, num_segments: int) -> Array:
    """``logsumexp`` over groups defined by ``segment_ids``.

    Numerically stable: subtracts the per-segment max before exponentiating.
    Padded states carry ``-inf`` and are absorbed into segments whose max is
    finite --- guaranteed by the separator-state map construction.
    """
    seg_max = jax.ops.segment_max(values, segment_ids, num_segments=num_segments)
    shifted = values - seg_max[segment_ids]
    seg_sum = jax.ops.segment_sum(
        jnp.exp(shifted), segment_ids, num_segments=num_segments
    )
    return seg_max + jnp.log(seg_sum)


def _collect(jt: JunctionTree, phi: Array) -> Array:
    """Leaves-to-root collect pass via ``lax.scan``.

    Returns per-clique log-totals (own potential plus all incoming child
    messages), shape ``(n_cliques, 2^m)``; padded states carry ``-inf``. Each
    step marginalizes the child's total onto the separator, $m_{C \\to P}(x_S)
    = \\log \\sum_{x_{C \\setminus S}} \\exp(\\phi_C(x_C) + \\text{incoming})$,
    and broadcasts it into the parent.
    """
    totals = _masked(jt, phi)
    order = jt.collect_order
    if not order:
        return totals

    m = jt.max_clique_size
    seps = [jt.separator(c, p) for c, p in order]
    n_sep_states = 1 << max(len(s) for s in seps)
    child_idx = jnp.asarray(np.array([c for c, _ in order], dtype=np.int32))
    parent_idx = jnp.asarray(np.array([p for _, p in order], dtype=np.int32))
    sep_maps = jnp.asarray(
        np.stack(
            [
                np.stack(
                    [
                        _sep_state_map(jt.cliques[c], sep, m),
                        _sep_state_map(jt.cliques[p], sep, m),
                    ]
                )
                for (c, p), sep in zip(order, seps)
            ]
        )
    )

    def step(
        totals: Array, edge_info: tuple[Array, Array, Array]
    ) -> tuple[Array, None]:
        child, parent, sep_pair = edge_info
        msg = _segment_logsumexp(totals[child], sep_pair[0], n_sep_states)
        return totals.at[parent].add(msg[sep_pair[1]]), None

    totals, _ = jax.lax.scan(step, totals, (child_idx, parent_idx, sep_maps))
    return totals


_LOG_ZERO = -1e30
"""Finite stand-in for $-\\infty$ in transfer matrices. Empty separator-state
segments (heterogeneous separator sizes along the path) would otherwise put
true $-\\infty$ rows into the matrices, and autodiff through an all-$-\\infty$
``logsumexp`` produces NaN gradients; clamping keeps values exact (the weight
$e^{-10^{30}}$ is exactly $0$ in floating point) and gradients clean.

This is not a stability floor in the sense the library excludes from core
(cf. CLAUDE.md): it changes no value for any potentials, carries no tunable
policy, and the slots it touches are structurally empty (topology-determined
constants with zero potential dependence, whose true gradient contribution is
zero). It exists to make autodiff agree with the exact mathematical gradient,
which is finite; only the differentiated computation needs it ---
:meth:`ChainTree.sample` builds the same matrices unclamped."""


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


### Junction trees ###


@dataclass(frozen=True)
class JunctionTree(Cliques):
    """The maximal cliques of a chordal graph, arranged in a tree rooted at one of them.

    The fields are the identity of the tree --- what ``__eq__`` / ``__hash__``
    compare and what JIT treats as the static cache key; everything else is derived
    on demand. Construct via :meth:`from_edges`; the raw constructor performs no
    validation.

    Its edges (:attr:`~goal.geometry.algebra.clique.Cliques.edges`) are those of the
    chordal completion, fill-ins included.
    """

    # Fields

    _cliques: tuple[tuple[int, ...], ...]
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
        the full chordal completion. The original edge list is not retained;
        consumers that need it must keep their own copy.

        Edges are undirected; self-loops are ignored; out-of-range endpoints
        raise ``ValueError``. The graph may be disconnected: clique-forest
        components are linked by empty-separator tree edges, over which
        sum-product correctly multiplies the per-component partition
        functions, and a node without edges is a clique of its own.
        Construction is deterministic (min-fill ties break toward the lowest
        vertex index). If ``max_treewidth`` is given, raise ``ValueError`` when
        the min-fill triangulation exceeds it --- a heuristic bound: the graph
        may still admit a smaller-treewidth triangulation that min-fill did
        not find.
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
        return JunctionTree(cliques, tree, 0)

    # Overrides

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        return self._cliques

    # Methods

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
        return tuple((c, parent[c]) for c in reversed(self.pre_order) if parent[c] >= 0)

    @property
    def node_owner(self) -> tuple[int, ...]:
        """For each node, the first clique (in storage order) that contains it."""
        owner = [-1] * self.n_nodes
        for ci, c in enumerate(self.cliques):
            for v in c:
                if owner[v] < 0:
                    owner[v] = ci
        return tuple(owner)

    @property
    def edge_owner(self) -> tuple[int, ...]:
        """For each edge, in the order of :attr:`edges`, the first clique that contains it."""
        index = {e: k for k, e in enumerate(self.edges)}
        owner = [-1] * len(index)
        for ci, c in enumerate(self.cliques):
            for x, vi in enumerate(c):
                for vj in c[x + 1 :]:
                    k = index[(vi, vj)]
                    if owner[k] < 0:
                        owner[k] = ci
        return tuple(owner)

    def separator(self, child: int, parent: int) -> tuple[int, ...]:
        """Sorted tuple of node indices shared by the (child, parent) cliques."""
        return tuple(sorted(set(self.cliques[child]) & set(self.cliques[parent])))

    def log_partition(self, phi: Array) -> Array:
        """$\\log \\sum_x \\exp \\sum_c \\phi_c(x_C)$ at the clique log-potentials ``phi``, by a leaves-to-root collect."""
        totals = _collect(self, phi)
        return jax.scipy.special.logsumexp(totals[self.root])

    def sample(self, key: Array, phi: Array) -> Array:
        """Draw one exact sample of all $n$ nodes at the clique log-potentials ``phi``.

        Collect pass for per-clique log-totals, then a pre-order walk down the
        rooted tree: the root clique is sampled from its full categorical, and each
        child clique from its total masked to states consistent with the values
        already drawn for the separator with its parent (the running-intersection
        property makes that conditioning sufficient).

        The same property makes the write pattern static: when the walk visits a
        clique, its already-assigned nodes are exactly that parent separator (any
        node shared with an earlier-visited clique lies on the tree path to it,
        which passes through the parent). Each step therefore writes the
        statically known complement, and every node is written exactly once --- by
        the first pre-order clique that contains it.

        The collect pass does not depend on ``key``, so under ``vmap`` over keys it
        is computed once across draws.
        """
        totals = _collect(self, phi)

        n, m = self.n_nodes, self.max_clique_size
        pre = self.pre_order
        parent = self.parent_of
        nc = self.n_cliques

        clique_idx = np.zeros(nc, dtype=np.int32)
        sep_map = np.zeros((nc, 1 << m), dtype=np.int32)
        sep_vars = np.zeros((nc, m), dtype=np.int32)
        sep_size = np.zeros(nc, dtype=np.int32)
        # Nodes this step writes (clique minus parent separator): node index, with
        # sentinel n for padded slots (dropped after the scan), and the node's
        # clique-local position for bit extraction.
        new_vars = np.full((nc, m), n, dtype=np.int32)
        new_pos = np.zeros((nc, m), dtype=np.int32)
        for step_i, c in enumerate(pre):
            cl = self.cliques[c]
            clique_idx[step_i] = c
            sep: tuple[int, ...] = ()
            if parent[c] >= 0:
                sep = self.separator(c, parent[c])
                sep_size[step_i] = len(sep)
                sep_vars[step_i, : len(sep)] = sep
                sep_map[step_i] = _sep_state_map(cl, sep, m)
            slot = 0
            for p, v in enumerate(cl):
                if v not in sep:
                    new_vars[step_i, slot] = v
                    new_pos[step_i, slot] = p
                    slot += 1

        bits = jnp.asarray(binary_states(m))
        m_range = jnp.arange(m, dtype=jnp.int32)

        def step(
            carry: tuple[Array, Array],
            info: tuple[Array, Array, Array, Array, Array, Array],
        ) -> tuple[tuple[Array, Array], None]:
            x, key = carry
            c_idx, smap, svars, ssize, nvars, npos = info
            # Mask to states consistent with already-sampled separator values
            # (no-op for the root, where ssize == 0 and target stays 0). Padded
            # states already carry -inf in the totals.
            valid_sep = m_range < ssize
            sep_vals = jnp.where(valid_sep, x[svars].astype(jnp.int32), 0)
            target = jnp.sum(sep_vals << m_range, dtype=jnp.int32)
            consistent = (smap == target) | (ssize == 0)
            logits = jnp.where(consistent, totals[c_idx], -jnp.inf)

            key, sub = jax.random.split(key)
            sampled = bits[jax.random.categorical(sub, logits)]
            x = x.at[nvars].set(sampled[npos].astype(x.dtype))
            return (x, key), None

        init = (jnp.zeros(n + 1, dtype=float), key)
        (x, _), _ = jax.lax.scan(
            step,
            init,
            (
                jnp.asarray(clique_idx),
                jnp.asarray(sep_map),
                jnp.asarray(sep_vars),
                jnp.asarray(sep_size),
                jnp.asarray(new_vars),
                jnp.asarray(new_pos),
            ),
        )
        return x[:n]


@dataclass(frozen=True)
class ChainTree(JunctionTree):
    """Junction tree whose cliques form a path --- a *path decomposition*.

    Every clique has at most two tree neighbours, so messages compose along a
    single sequence, and :meth:`log_partition` and :meth:`sample` run at
    $O(\\log n)$ depth through ``associative_scan``: up to two orders of magnitude
    faster on accelerators for long chains with small separators (measured ~100x for
    the gradient of $\\log Z$ at $n \\approx 1000$). The underlying graphs are interval
    graphs of bounded *pathwidth* --- chains, band graphs, triangulated cycles.
    Construction validates the path shape and raises ``ValueError`` when the clique
    tree branches, so holders of a ``ChainTree`` are guaranteed the fast regime;
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
                "neighbours); use a JunctionTree for branching topologies"
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
        return ChainTree(jt.cliques, jt.tree_edges, jt.root)

    # Overrides

    @override
    def log_partition(self, phi: Array) -> Array:
        """$\\log Z$ along the path via ``associative_scan``.

        Mathematically, the collect recurrence along a path is a product of
        transfer matrices in the log-semiring: with $T_k[t_{\\text{out}},
        t_{\\text{in}}] = \\log \\sum_{s : \\text{out}(s) = t_{\\text{out}},\\,
        \\text{in}(s) = t_{\\text{in}}} \\exp \\phi_k(s)$, the message recurrence is
        the ``logsumexp``-matvec $m_k = T_k \\otimes m_{k-1}$. Matrix composition
        is associative, so the product runs in $O(\\log K)$ depth instead of the
        $O(K)$ sequential scan --- the win on accelerators, where the sequential
        collect is latency-bound.

        Each combine costs $(2^{\\lvert S \\rvert})^3$ versus the sequential
        scan's $2^{\\lvert S \\rvert + 1}$ per step, so the log-depth trade pays
        off for separator sizes up to roughly 6; beyond that prefer a
        :class:`JunctionTree`.
        """
        order = self.clique_path
        phi = _masked(self, phi)
        k_total = len(order)
        if k_total == 1:
            return jax.scipy.special.logsumexp(phi[order[0]])

        m = self.max_clique_size
        cliques = self.cliques
        seps = [self.separator(a, b) for a, b in pairwise(order)]
        n_sep_states = 1 << max(len(s) for s in seps)
        out_maps = [
            _sep_state_map(cliques[order[k]], seps[k], m) for k in range(k_total - 1)
        ]
        in_maps = [
            _sep_state_map(cliques[order[k]], seps[k - 1], m) for k in range(1, k_total)
        ]

        msg = _segment_logsumexp(phi[order[0]], jnp.asarray(out_maps[0]), n_sep_states)
        msg = jnp.maximum(msg, _LOG_ZERO)

        if k_total > 2:
            # T_k as a 2D segment reduction: segment id = out * S + in.
            ids = np.stack(
                [
                    out_maps[k] * n_sep_states + in_maps[k - 1]
                    for k in range(1, k_total - 1)
                ]
            )
            phi_mid = phi[jnp.asarray(np.array(order[1:-1], dtype=np.int32))]
            transfer = jax.vmap(
                lambda p, i: _segment_logsumexp(
                    p, i, n_sep_states * n_sep_states
                ).reshape(n_sep_states, n_sep_states)
            )(phi_mid, jnp.asarray(ids))
            transfer = jnp.maximum(transfer, _LOG_ZERO)

            def logmm(left: Array, right: Array) -> Array:
                # Earlier matrix `left`, later `right`; composed = right after left.
                return jax.scipy.special.logsumexp(
                    right[..., :, :, None] + left[..., None, :, :], axis=-2
                )

            prod = jax.lax.associative_scan(logmm, transfer)[-1]
            msg = jax.scipy.special.logsumexp(prod + msg[None, :], axis=-1)
            msg = jnp.maximum(msg, _LOG_ZERO)

        return jax.scipy.special.logsumexp(
            phi[order[-1]] + msg[jnp.asarray(in_maps[-1])]
        )

    @override
    def sample(self, key: Array, phi: Array) -> Array:
        """Draw one exact sample at $O(\\log n)$ depth.

        Parallel forward-filtering backward-sampling: two ``associative_scan`` passes
        replace the sequential collect and walk-down of :meth:`JunctionTree.sample`.

        Mathematically, the backward messages $\\beta_k(t)$ (log-mass downstream of
        clique $k$ given separator state $t$) are suffix products of the transfer
        matrices. The ancestral pass then exploits that each clique's conditional
        draw depends on the past only through the incoming separator state: a
        shared Gumbel vector per clique resolves the draw for *every possible*
        incoming state at once, defining a random map $F_k : s_{k-1} \\mapsto s_k$
        over the $2^{\\lvert S \\rvert}$ separator states. The Gumbels are
        independent across cliques and the realized chain consumes each map at a
        point independent of that clique's Gumbels, so composing the maps --- an
        associative operation --- realizes the joint sample exactly.

        Only the Gumbels and final gathers depend on ``key``: under ``vmap`` over
        keys the messages and conditional logits are computed once across draws.
        """
        order = self.clique_path
        phi = _masked(self, phi)
        k_total = len(order)
        m = self.max_clique_size
        n = self.n_nodes
        bits = jnp.asarray(binary_states(m))

        # Scatter table: clique-local position -> node index; padded slots hit the
        # sentinel row n and are dropped. Overlapping writes agree by construction
        # (shared nodes are separator-consistent), so write order is irrelevant.
        var_idx = np.full((k_total, m), n, dtype=np.int32)
        for k, c in enumerate(order):
            var_idx[k, : len(self.cliques[c])] = self.cliques[c]

        phi_path = phi[jnp.asarray(np.array(order, dtype=np.int32))]  # (K, 2^m)

        if k_total == 1:
            vals = bits[jax.random.categorical(key, phi_path[0])].astype(float)
            x = jnp.zeros(n + 1, dtype=float).at[jnp.asarray(var_idx[0])].set(vals)
            return x[:n]

        cliques = self.cliques
        seps = [self.separator(a, b) for a, b in pairwise(order)]
        n_sep_states = 1 << max(len(s) for s in seps)
        out_maps = np.stack(
            [_sep_state_map(cliques[order[k]], seps[k], m) for k in range(k_total - 1)]
        )
        in_maps = np.stack(
            [
                _sep_state_map(cliques[order[k]], seps[k - 1], m)
                for k in range(1, k_total)
            ]
        )

        # Backward messages beta_k (k = 0..K-2) via suffix products of the
        # transfer matrices M_k[t_in, t_out].
        last_vec = _segment_logsumexp(
            phi_path[-1], jnp.asarray(in_maps[-1]), n_sep_states
        )
        if k_total > 2:
            ids = np.stack(
                [
                    in_maps[j] * n_sep_states + out_maps[j + 1]
                    for j in range(k_total - 2)
                ]
            )
            transfer = jax.vmap(
                lambda p, i: _segment_logsumexp(
                    p, i, n_sep_states * n_sep_states
                ).reshape(n_sep_states, n_sep_states)
            )(phi_path[1:-1], jnp.asarray(ids))

            def logmm(acc: Array, elem: Array) -> Array:
                # Log-semiring product elem . acc. Under reverse=True the first
                # argument accumulates the higher-index product, so this yields
                # suffix[j] = M_j . M_{j+1} . ... --- lowest index leftmost.
                return jax.scipy.special.logsumexp(
                    elem[..., :, :, None] + acc[..., None, :, :], axis=-2
                )

            suffix = jax.lax.associative_scan(logmm, transfer, reverse=True)
            betas_front = jax.scipy.special.logsumexp(
                suffix + last_vec[None, None, :], axis=-1
            )
            betas = jnp.concatenate([betas_front, last_vec[None, :]])  # (K-1, S)
        else:
            betas = last_vec[None, :]

        # Conditional logits for every clique k and every incoming separator state
        # t: phi_k(s) + beta_k(out_k(s)), masked to in_k(s) == t. Clique 0 has no
        # incoming separator: its in-map row is zero, so conditioning on t = 0
        # recovers the unconditional draw; the last clique has no beta term.
        in_ext = jnp.asarray(
            np.concatenate([np.zeros((1, 1 << m), dtype=np.int32), in_maps])
        )
        out_ext = jnp.asarray(
            np.concatenate([out_maps, np.zeros((1, 1 << m), dtype=np.int32)])
        )
        betas_ext = jnp.concatenate([betas, jnp.zeros((1, n_sep_states))])
        beta_term = jnp.take_along_axis(betas_ext, out_ext, axis=1)  # (K, 2^m)
        t_range = jnp.arange(n_sep_states)
        logits = jnp.where(
            in_ext[:, None, :] == t_range[None, :, None],
            (phi_path + beta_term)[:, None, :],
            -jnp.inf,
        )  # (K, S, 2^m)

        # Shared Gumbels resolve the draw for all hypothetical separator states at
        # once; sampled[k, t] is the clique state drawn given incoming state t, and
        # maps[k, t] the separator state it hands to clique k + 1.
        gumbels = jax.random.gumbel(key, (k_total, 1 << m))
        sampled = jnp.argmax(logits + gumbels[:, None, :], axis=-1)  # (K, S)
        maps = jnp.take_along_axis(out_ext, sampled, axis=1)  # (K, S)

        def compose(first: Array, second: Array) -> Array:
            # (second after first)[t] = second[first[t]]
            return jnp.take_along_axis(second, first, axis=-1)

        # Realized separator sequence: t_{-1} = 0, t_k = F_k(t_{k-1}).
        prefix = jax.lax.associative_scan(compose, maps[:-1])  # (K-1, S)
        t_prev = jnp.concatenate([jnp.zeros(1, dtype=prefix.dtype), prefix[:, 0]])
        states = jnp.take_along_axis(sampled, t_prev[:, None], axis=1)[:, 0]  # (K,)

        vals = bits[states].astype(float).reshape(-1)
        x = jnp.zeros(n + 1, dtype=float).at[jnp.asarray(var_idx).reshape(-1)].set(vals)
        return x[:n]

    # Methods

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
