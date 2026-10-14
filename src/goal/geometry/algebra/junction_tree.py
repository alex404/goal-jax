"""Junction trees: the cliques of a chordal graph arranged in a tree, with exact sum-product over binary variables.

:meth:`JunctionTree.from_edges` builds a junction tree from an undirected graph. The graph is a
seed: the tree describes its chordal completion, fill-in edges included.
:meth:`JunctionTree.log_partition` and :meth:`JunctionTree.sample` run sum-product on a table of
clique log-potentials, at a cost exponential in the treewidth rather than in the number of nodes,
and a :class:`ChainTree` runs the same computations at logarithmic depth when the cliques form a
path. The main user is :class:`~goal.models.ChordalBoltzmann`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .clique import Cliques

### Static tables ###


def binary_states(n: int) -> np.ndarray:
    """``(2^n, n)`` table of all binary states: row $s$ is the LSB-first binary expansion of $s$."""
    states = np.arange(1 << n, dtype=np.int64)
    return ((states[:, None] >> np.arange(n)) & 1).astype(np.int32)


def _incidence(cliques: tuple[tuple[int, ...], ...], n: int) -> np.ndarray:
    """``(n_cliques, n)`` bool: True where the clique contains the node."""
    incidence = np.zeros((len(cliques), n), dtype=bool)
    for c, clique in enumerate(cliques):
        incidence[c, list(clique)] = True
    return incidence


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
    positions = np.searchsorted(clique, separator)
    weights = 1 << np.arange(len(separator))
    result[: 1 << len(clique)] = binary_states(len(clique))[:, positions] @ weights
    return result


def _tree_tables(jt: JunctionTree) -> tuple[Array, Array, Array, int]:
    """The tree edges in pre-order, parent before child, for a tree of at least two cliques.

    Returns ``(parents, children, sep_maps, n_sep_states)``: ``sep_maps[k]`` stacks the
    separator-state maps (:func:`_sep_state_map`) of the child and of the parent of edge
    $k$, over the separator they share, and ``n_sep_states`` is the number of states of
    the largest separator.
    """
    parent_of = jt.parent_of
    children = jt.pre_order[1:]
    parents = [parent_of[c] for c in children]
    seps = [jt.separator(c, p) for c, p in zip(children, parents)]
    m = jt.max_clique_size
    sep_maps = np.stack(
        [
            [
                _sep_state_map(jt.cliques[c], sep, m),
                _sep_state_map(jt.cliques[p], sep, m),
            ]
            for c, p, sep in zip(children, parents, seps)
        ]
    )
    return (
        jnp.asarray(parents, dtype=jnp.int32),
        jnp.asarray(children, dtype=jnp.int32),
        jnp.asarray(sep_maps),
        1 << max(len(s) for s in seps),
    )


### Traced computations ###


def _masked(jt: JunctionTree, phi: Array) -> Array:
    """The potentials with padded states set to $-\\infty$, so they fall out of every ``logsumexp``."""
    sizes = np.array([len(c) for c in jt.cliques])
    real = np.arange(1 << jt.max_clique_size)[None, :] < (1 << sizes[:, None])
    return jnp.where(jnp.asarray(real), phi, -jnp.inf)


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
    and broadcasts it into the parent. The edges run in reverse pre-order, so every
    child has received its own messages before it sends.
    """
    totals = _masked(jt, phi)
    if jt.n_cliques == 1:
        return totals
    parents, children, sep_maps, n_sep_states = _tree_tables(jt)

    def step(totals: Array, edge: tuple[Array, Array, Array]) -> tuple[Array, None]:
        parent, child, maps = edge
        msg = _segment_logsumexp(totals[child], maps[0], n_sep_states)
        return totals.at[parent].add(msg[maps[1]]), None

    totals, _ = jax.lax.scan(
        step, totals, (parents[::-1], children[::-1], sep_maps[::-1])
    )
    return totals


def _read_nodes(jt: JunctionTree, states: Array) -> Array:
    """The value of every node, read from the drawn state of the first clique that contains it.

    The cliques agree on the nodes they share, so any containing clique would do.
    """
    owner = np.asarray(jt.node_owner)
    positions = jt.local_positions[owner, np.arange(jt.n_nodes)]
    bits = jnp.asarray(binary_states(jt.max_clique_size))
    return bits[states[owner], positions].astype(float)


def _logmm(left: Array, right: Array) -> Array:
    """Batched matrix product ``right @ left`` in the log-semiring."""
    return jax.scipy.special.logsumexp(
        right[..., :, :, None] + left[..., None, :, :], axis=-2
    )


def _transfer_matrices(
    phi: Array, rows: Array, cols: Array, n_sep_states: int
) -> Array:
    """Per-clique transfer matrices $T[t_r, t_c] = \\log \\sum_{s : r(s) = t_r, c(s) = t_c} \\exp \\phi(s)$.

    ``rows`` and ``cols`` map each clique's states to the separator states that index
    the rows and columns of its matrix.
    """
    return jax.vmap(
        lambda p, i: _segment_logsumexp(p, i, n_sep_states**2).reshape(
            n_sep_states, n_sep_states
        )
    )(phi, rows * n_sep_states + cols)


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
        """The number of pairs of neighbours of $v$ that are not adjacent."""
        nbrs = np.flatnonzero(adj[v])
        k = nbrs.size
        return k * (k - 1) // 2 - int(
            np.count_nonzero(np.triu(adj[np.ix_(nbrs, nbrs)], 1))
        )

    fill = np.array([fill_in(v) for v in range(n)])
    peo: list[int] = []
    for _ in range(n):
        # argmin takes the first minimum --- deterministic lowest-index tie-break.
        v = int(np.argmin(np.where(active, fill, n * n)))
        nbrs = np.flatnonzero(adj[v])
        # Fill-in: v's neighbours become pairwise adjacent.
        fa, fb = np.nonzero(np.triu(~adj[np.ix_(nbrs, nbrs)], 1))
        adj[np.ix_(nbrs, nbrs)] = True
        adj[nbrs, nbrs] = False
        chordal |= adj
        # Fill counts change only for vertices whose neighbourhood changed: v's
        # neighbours, plus vertices adjacent to both endpoints of a fill-in.
        touched = np.zeros(n, dtype=bool)
        touched[nbrs] = True
        touched |= (adj[nbrs[fa]] & adj[nbrs[fb]]).any(axis=0)
        active[v] = False
        adj[v, :] = adj[:, v] = False
        peo.append(v)
        for u in np.flatnonzero(touched & active).tolist():
            fill[u] = fill_in(u)
    return chordal, peo


def _maximal_cliques(
    chordal: np.ndarray, peo: list[int]
) -> tuple[tuple[int, ...], ...]:
    """Maximal cliques of the chordal graph, in elimination order.

    Every maximal clique is the candidate $\\{v\\} \\cup$ (later neighbours of $v$) of
    some $v$, and a candidate is maximal when no other candidate contains it. The
    candidates are distinct (each contains its own vertex but no earlier one), so
    containment is strict.
    """
    n = len(peo)
    pos = np.empty(n, dtype=np.int64)
    pos[np.array(peo)] = np.arange(n)
    candidates = (chordal & (pos[None, :] > pos[:, None])) | np.eye(n, dtype=bool)
    # outside[v, u] counts the members of candidate v missing from candidate u.
    outside = candidates.astype(np.float32) @ (~candidates).T.astype(np.float32)
    contained = outside == 0
    np.fill_diagonal(contained, False)
    return tuple(
        tuple(np.flatnonzero(candidates[v]).tolist())
        for v in peo
        if not contained[v].any()
    )


def _spanning_tree(incidence: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Maximum-weight spanning tree of the clique graph, linked across components.

    The weight of two cliques is the size of their intersection. Kruskal on the
    weights, then zero-weight links join any remaining components (a disconnected
    graph yields a clique forest) into a single tree.
    """
    n_cliques = incidence.shape[0]
    overlap = incidence.astype(np.float32) @ incidence.T.astype(np.float32)
    rows, cols = np.nonzero(np.triu(overlap, 1))
    heaviest = np.argsort(-overlap[rows, cols], kind="stable")

    parent = list(range(n_cliques))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: list[tuple[int, int]] = []
    for ci, cj in zip(rows[heaviest].tolist(), cols[heaviest].tolist()):
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


# The index tables of a tree are built with numpy from its topology at trace time, so under
# ``jax.jit`` the compilation cache is the only cache.
@dataclass(frozen=True)
class JunctionTree(Cliques):
    """The maximal cliques of a chordal graph, arranged in a tree rooted at one of them.

    The fields are the identity of the tree --- what ``__eq__`` / ``__hash__``
    compare and what JIT treats as the static cache key; everything else is derived
    on demand. Construct via :meth:`from_edges`; the raw constructor performs no
    validation.

    Its edges (:attr:`~goal.geometry.algebra.clique.Cliques.edges`) are those of the
    chordal completion, fill-ins included.

    Inference runs on a table of clique log-potentials ``phi`` of shape ``(n_cliques, 2^m)``,
    with $m$ the size of the largest clique: row $c$ holds $\\phi_c(x_C)$ at every binary state
    of clique $C$, encoded LSB-first in ascending node order (:func:`binary_states`), and entries
    at states $s \\geq 2^{\\lvert C \\rvert}$ are padding. :meth:`log_partition` and
    :meth:`sample` cost $O(n 2^{w+1})$, with $w$ the treewidth.

    Mathematically, the tree satisfies the running-intersection property: for any two cliques
    $C_i, C_j$, every clique on the path between them contains $C_i \\cap C_j$. A maximum-weight
    spanning tree of the clique graph, weighted by separator size, has it.
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
        cliques = _maximal_cliques(chordal, peo)

        max_size = max(len(c) for c in cliques)
        if max_treewidth is not None and max_size - 1 > max_treewidth:
            msg = (
                f"min-fill triangulation produced treewidth {max_size - 1} > "
                f"max_treewidth {max_treewidth} (largest clique size {max_size})"
            )
            raise ValueError(msg)

        return JunctionTree(cliques, _spanning_tree(_incidence(cliques, n)), 0)

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
    def incidence(self) -> np.ndarray:
        """``(n_cliques, n_nodes)`` bool: True where the clique contains the node."""
        return _incidence(self.cliques, self.n_nodes)

    @property
    def local_positions(self) -> np.ndarray:
        """``(n_cliques, n_nodes)``: the position of each node within each clique that contains it."""
        return np.cumsum(self.incidence, axis=1) - 1

    @property
    def pre_order(self) -> tuple[int, ...]:
        """Cliques in depth-first pre-order from the root (parent before children)."""
        return self._traversal[0]

    @property
    def parent_of(self) -> tuple[int, ...]:
        """parent_of[c] = parent clique index in the rooted tree, or -1 for the root."""
        return self._traversal[1]

    @property
    def node_owner(self) -> tuple[int, ...]:
        """For each node, the first clique (in storage order) that contains it."""
        return tuple(self.incidence.argmax(axis=0).tolist())

    @property
    def edge_owner(self) -> tuple[int, ...]:
        """For each edge, in the order of :attr:`edges`, the first clique that contains it."""
        incidence = self.incidence
        edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        both = incidence[:, edges[:, 0]] & incidence[:, edges[:, 1]]
        return tuple(both.argmax(axis=0).tolist())

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
        rooted tree: the root clique is drawn from its total, and each child clique
        from its total restricted to the states that agree with its parent's drawn
        state on their separator. Given the separator, a child's subtree is
        independent of the rest of the tree, so this is the exact conditional, and by
        the running-intersection property the drawn cliques agree on every node they
        share.

        The collect pass does not depend on ``key``, so under ``vmap`` over keys it
        is computed once across draws.
        """
        totals = _collect(self, phi)
        root_key, key = jax.random.split(key)
        root_state = jax.random.categorical(root_key, totals[self.root])
        states = (
            jnp.zeros(self.n_cliques, dtype=root_state.dtype)
            .at[self.root]
            .set(root_state)
        )
        if self.n_cliques == 1:
            return _read_nodes(self, states)
        parents, children, sep_maps, _ = _tree_tables(self)

        def step(
            states: Array, edge: tuple[Array, Array, Array, Array]
        ) -> tuple[Array, None]:
            parent, child, maps, k = edge
            consistent = maps[0] == maps[1][states[parent]]
            logits = jnp.where(consistent, totals[child], -jnp.inf)
            return states.at[child].set(jax.random.categorical(k, logits)), None

        keys = jax.random.split(key, children.shape[0])
        states, _ = jax.lax.scan(step, states, (parents, children, sep_maps, keys))
        return _read_nodes(self, states)

    @property
    def _traversal(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Depth-first traversal from the root: the cliques in pre-order, and the parent of each."""
        nbrs: list[list[int]] = [[] for _ in range(self.n_cliques)]
        for a, b in self.tree_edges:
            nbrs[a].append(b)
            nbrs[b].append(a)
        order: list[int] = []
        parent = [-1] * self.n_cliques
        stack = [self.root]
        while stack:
            c = stack.pop()
            order.append(c)
            for u in nbrs[c]:
                if u != parent[c]:
                    parent[u] = c
                    stack.append(u)
        return tuple(order), tuple(parent)


@dataclass(frozen=True)
class ChainTree(JunctionTree):
    """Junction tree whose cliques form a path, rooted at one end --- a *path decomposition*.

    Every clique has at most two tree neighbours, so messages compose along a
    single sequence, the pre-order, and :meth:`log_partition` and :meth:`sample` run at
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
        degree = _degrees(self.tree_edges, self.n_cliques)
        if degree.max() > 2:
            msg = (
                "Clique tree is not a path (a clique has more than two "
                "neighbours); use a JunctionTree for branching topologies"
            )
            raise ValueError(msg)
        if degree[self.root] > 1:
            raise ValueError("The root of a ChainTree must be an end of the path")

    # Constructors

    @staticmethod
    @override
    def from_edges(
        n: int,
        edges: Sequence[tuple[int, int]],
        max_treewidth: int | None = None,
    ) -> ChainTree:
        """Build a chain tree from an undirected graph on ``n`` nodes.

        Same construction as :meth:`JunctionTree.from_edges`, rooted at the first end of
        the path, plus the path-shape validation.
        """
        jt = JunctionTree.from_edges(n, edges, max_treewidth)
        end = int(np.argmin(_degrees(jt.tree_edges, jt.n_cliques)))
        return ChainTree(jt.cliques, jt.tree_edges, end)

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
        phi = _masked(self, phi)[jnp.asarray(self.pre_order)]
        if self.n_cliques == 1:
            return jax.scipy.special.logsumexp(phi[0])
        # Separator k joins path cliques k and k + 1; in_maps[k] is clique k + 1's
        # map onto it and out_maps[k] clique k's.
        _, _, sep_maps, n_sep_states = _tree_tables(self)
        in_maps, out_maps = sep_maps[:, 0], sep_maps[:, 1]

        msg = _segment_logsumexp(phi[0], out_maps[0], n_sep_states)
        msg = jnp.maximum(msg, _LOG_ZERO)
        if self.n_cliques > 2:
            # T_k[out, in] for the inner cliques k = 1..K-2.
            transfer = _transfer_matrices(
                phi[1:-1], out_maps[1:], in_maps[:-1], n_sep_states
            )
            transfer = jnp.maximum(transfer, _LOG_ZERO)
            prod = jax.lax.associative_scan(_logmm, transfer)[-1]
            msg = jax.scipy.special.logsumexp(prod + msg[None, :], axis=-1)
            msg = jnp.maximum(msg, _LOG_ZERO)
        return jax.scipy.special.logsumexp(phi[-1] + msg[in_maps[-1]])

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
        path = jnp.asarray(self.pre_order)
        phi = _masked(self, phi)[path]
        k_total = self.n_cliques
        if k_total == 1:
            state = jax.random.categorical(key, phi[0])
            return _read_nodes(self, jnp.atleast_1d(state))
        _, _, sep_maps, n_sep_states = _tree_tables(self)
        in_maps, out_maps = sep_maps[:, 0], sep_maps[:, 1]

        # Backward messages beta_k on separator k (k = 0..K-2) via suffix products
        # of the transfer matrices M_k[t_in, t_out] of the inner cliques.
        last_vec = _segment_logsumexp(phi[-1], in_maps[-1], n_sep_states)
        if k_total > 2:
            transfer = _transfer_matrices(
                phi[1:-1], in_maps[:-1], out_maps[1:], n_sep_states
            )

            # Under reverse=True the first argument accumulates the higher-index
            # product, so this yields suffix[j] = M_j . M_{j+1} . ... --- lowest
            # index leftmost.
            suffix = jax.lax.associative_scan(_logmm, transfer, reverse=True)
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
        zeros = jnp.zeros((1, phi.shape[1]), dtype=in_maps.dtype)
        in_ext = jnp.concatenate([zeros, in_maps])
        out_ext = jnp.concatenate([out_maps, zeros])
        betas_ext = jnp.concatenate([betas, jnp.zeros((1, n_sep_states))])
        beta_term = jnp.take_along_axis(betas_ext, out_ext, axis=1)  # (K, 2^m)
        t_range = jnp.arange(n_sep_states)
        logits = jnp.where(
            in_ext[:, None, :] == t_range[None, :, None],
            (phi + beta_term)[:, None, :],
            -jnp.inf,
        )  # (K, S, 2^m)

        # Shared Gumbels resolve the draw for all hypothetical separator states at
        # once; sampled[k, t] is the clique state drawn given incoming state t, and
        # maps[k, t] the separator state it hands to clique k + 1.
        gumbels = jax.random.gumbel(key, (k_total, phi.shape[1]))
        sampled = jnp.argmax(logits + gumbels[:, None, :], axis=-1)  # (K, S)
        maps = jnp.take_along_axis(out_ext, sampled, axis=1)  # (K, S)

        def compose(first: Array, second: Array) -> Array:
            # (second after first)[t] = second[first[t]]
            return jnp.take_along_axis(second, first, axis=-1)

        # Realized separator sequence: t_{-1} = 0, t_k = F_k(t_{k-1}).
        prefix = jax.lax.associative_scan(compose, maps[:-1])  # (K-1, S)
        t_prev = jnp.concatenate([jnp.zeros(1, dtype=prefix.dtype), prefix[:, 0]])
        states = jnp.take_along_axis(sampled, t_prev[:, None], axis=1)[:, 0]  # (K,)
        return _read_nodes(
            self, jnp.zeros(k_total, dtype=states.dtype).at[path].set(states)
        )


def _degrees(tree_edges: tuple[tuple[int, int], ...], n_cliques: int) -> np.ndarray:
    """The number of tree neighbours of each clique."""
    ends = np.asarray(tree_edges, dtype=np.int64).reshape(-1)
    return np.bincount(ends, minlength=n_cliques)
