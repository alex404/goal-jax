"""Cliques: the combinatorial skeleton a parameter layout is indexed by.

A ``Cliques`` records which nodes there are, which of them are roots, and which groups of
nodes interact. ``LinearCliques`` in ``exponential_family/clique.py`` turns one into a parameter layout;
this module is purely about the combinatorics, over integer indices, with no JAX.

**The strategy, and why the class is not called ``Graph``.** A ``Cliques`` *presents* a graph
$G = (V, E)$ --- as a **cover of $V$ by complete subgraphs**, together with a distinguished
root set $R = \\{0, \\ldots, n_r - 1\\} \\subseteq V$. The cover is what is stored, because
one clique is one parameter block one level up, and a three-way interaction has to be a
single object rather than three pairwise ones. The graph is what is *derived*: two nodes are
adjacent exactly when some clique contains both, so $E$ falls out of the cover and never
needs declaring. :attr:`Cliques.graph` is that derivation, and everything about
connectivity --- levels, boundary, depth --- reads from it. So the name says what is stored
and the property says what it means.

A node's *level* is its distance from the root set, $\\ell(v) = \\min_{r \\in R} d(v, r)$,
with $d$ the number of edges on a shortest path. The fibres of $\\ell$ are the *level
sets*, so depth is read off the cover rather than declared --- and because edges are
derived, a single step crosses a whole clique. The path $0 - 1 - 2$ rooted at $\\{0\\}$
has three level sets of sizes $(1, 1, 1)$; adding the clique $\\{0, 1, 2\\}$ makes $0$
adjacent to $2$ and collapses them to $(1, 2)$; the path $0 - 2 - 1$ rooted at
$\\{0, 1\\}$ gives $(2, 1)$.

Adjacency implies a level gap of at most one, since a shorter route would otherwise exist,
and because cliques are complete this lifts from edges to cliques: every clique lies within
one level or crosses two consecutive ones. Neither is a condition to check --- no cover can
describe a graph where it fails.

So the cliques fall into three groups --- wholly root, crossing, and wholly above --- and
those groups partition the cover. That partition is what makes
:attr:`Cliques.canonical_cliques` a layout listing every clique exactly once.

**Nodes are numbered by level**, roots first: $\\ell$ is non-decreasing in the index. This is
a numbering convention rather than a restriction --- sorting the nodes by level achieves it
for any graph --- and it is what makes layout order *readable* rather than computed.
Dropping the root level becomes a shift by $n_r$ instead of a permutation, so the cliques
one level up occupy a contiguous suffix at contiguous indices, and a manifold can splice its
deep span in by renumbering rather than reordering.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection
from dataclasses import dataclass


@dataclass(frozen=True)
class Cliques:
    """A rooted graph as $(V, R, C)$: its nodes, its root set, and its clique cover.

    Each clique is the index tuple of the nodes it couples, with no manifold attached ---
    :class:`~goal.geometry.exponential_family.clique.LinearCliques` is what hangs parameters off
    one. Every node must be reachable from the root set, so that its level is defined.
    Singleton cliques are *not* required: whether a node carries a bias of its own is a
    fact about the family occupying it, not about the graph. A lone edge ``((0, 1),)`` is
    a legitimate cover, and is what an interaction on its own looks like.

    Members are sorted within each clique and the cliques among themselves at construction,
    so two equivalent descriptions compare and hash equal. That storage order is not layout
    order --- :attr:`canonical_cliques` is.

    Node indices must ascend with level: roots first, then the boundary, then everything
    deeper. Renumber if a graph does not already come that way.
    """

    # Fields

    n_nodes: int
    """$|V|$. Root nodes are ``0`` to ``n_roots - 1``."""

    n_roots: int
    """$|R|$. The root set is the origin that levels are measured from."""

    cliques: tuple[tuple[int, ...], ...]
    """$C$, the cover. Singletons, where present, are the per-node blocks."""

    def __post_init__(self) -> None:
        for clique in self.cliques:
            if not clique:
                raise ValueError("empty clique: a clique must name at least one node")
            if len(set(clique)) != len(clique):
                raise ValueError(f"clique {clique} repeats a node")
        normalized = tuple(sorted(tuple(sorted(c)) for c in self.cliques))
        object.__setattr__(self, "cliques", normalized)
        self._validate()

    # Properties

    @property
    def graph(self) -> tuple[tuple[int, ...], ...]:
        """The graph this cover presents: node ``i``'s neighbours, ascending, are ``graph[i]``.

        Two nodes are adjacent exactly when some clique contains both, so this is
        *derived* and never declared --- see the module docstring for why the cover is
        the primitive. Everything about connectivity reads from here:
        :attr:`node_levels` traverses it, and :attr:`edges` is the same relation as a
        sorted pair list.
        """
        neighbours: list[set[int]] = [set() for _ in range(self.n_nodes)]
        for clique in self.cliques:
            for i in clique:
                neighbours[i].update(j for j in clique if j != i)
        return tuple(tuple(sorted(n)) for n in neighbours)

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """:attr:`graph` as a sorted list of pairs, each once."""
        return tuple(
            sorted((i, j) for i, near in enumerate(self.graph) for j in near if i < j)
        )

    @property
    def node_levels(self) -> tuple[int, ...]:
        """The distance of each node from the root set, $\\min_{r \\in R} d(v, r)$.

        Computed by multi-source breadth-first search: seeding every root at zero means a
        node's first visit is along a shortest path from the nearest root. Unreachable
        nodes come back as ``-1``, which construction rejects, so a constructed clique
        set never holds one.
        """
        neighbours = self.graph
        levels = [-1] * self.n_nodes
        queue = deque(range(self.n_roots))
        for i in queue:
            levels[i] = 0
        while queue:
            i = queue.popleft()
            for j in neighbours[i]:
                if levels[j] < 0:
                    levels[j] = levels[i] + 1
                    queue.append(j)
        return tuple(levels)

    @property
    def level_sets(self) -> tuple[tuple[int, ...], ...]:
        """Nodes grouped by distance from the root set --- the fibres of :attr:`node_levels`.

        Level ``0`` is exactly the root nodes; level ``k`` is the nodes at distance $k$.
        The length of this tuple is the depth of the graph.
        """
        node_levels = self.node_levels
        depth = max(node_levels) + 1
        return tuple(
            tuple(i for i, lvl in enumerate(node_levels) if lvl == level)
            for level in range(depth)
        )

    @property
    def root_nodes(self) -> frozenset[int]:
        """The root set $R$, as a set of node indices.

        Node indices ascend with level, so this is $\\{0, \\ldots, n_r - 1\\}$ and testing
        membership of it is the same as testing ``node_levels[i] == 0``.
        """
        return frozenset(range(self.n_roots))

    @property
    def root_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques lying wholly within the root set.

        This and the two properties below are the three groups of
        :func:`partition` at the root set, read back as cliques --- the split a layout is
        stored by. That the underlying operation yields *positions* is what lets a manifold
        slice its coordinate vector by the same split.
        """
        inside, _, _ = self.partition(self.root_nodes)
        return tuple(self.cliques[i] for i in inside)

    @property
    def cross_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques holding both a root node and a non-root node.

        Mathematically, the cut-set of the partition $(R, V \\setminus R)$ --- the cliques
        carrying interactions between the root set and everything else.
        """
        _, crossing, _ = self.partition(self.root_nodes)
        return tuple(self.cliques[i] for i in crossing)

    @property
    def deep_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques lying wholly outside the root set, in this graph's indices.

        Because node indices ascend with level, "outside the root set" is "index at least
        :attr:`n_roots`". These are exactly the cliques that survive :meth:`ascend_level`,
        which renumbers them by $-n_r$ and nothing else.
        """
        _, _, outside = self.partition(self.root_nodes)
        return tuple(self.cliques[i] for i in outside)

    @property
    def boundary(self) -> tuple[int, ...]:
        """The non-root nodes adjacent to the root set --- the root set one level up.

        Mathematically, the vertex boundary $\\partial R$ of the root set $R$. Cliques are
        complete, so these are exactly the non-root members of :attr:`cross_cliques`, and
        the level gap bound puts every one of them at level 1 --- which is what this
        computes. Empty exactly when the graph has depth one.
        """
        node_levels = self.node_levels
        return tuple(i for i in range(self.n_nodes) if node_levels[i] == 1)

    @property
    def canonical_cliques(self) -> tuple[tuple[int, ...], ...]:
        """The cliques in parameter-layout order.

        Root cliques first, then cross cliques, then the cliques one level up shifted
        back into this graph's indices --- the same rule applied recursively. Those three
        groups partition the cliques, so every clique appears exactly once, and the order
        nests: the ascended graph's cliques form a contiguous suffix, in precisely the
        order that graph would put them in on its own.

        Because node indices ascend with level, relabelling is $+ n_r$ and nothing is
        reordered --- which is what lets a manifold splice its deep span in unchanged.
        """
        head = self.root_cliques + self.cross_cliques
        if self.n_roots == self.n_nodes:  # depth one: nothing above to recurse into
            return head
        return head + tuple(
            tuple(i + self.n_roots for i in c)
            for c in self.ascend_level().canonical_cliques
        )

    # Methods

    def partition(
        self, nodes: Collection[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Split the cover by how each clique meets ``nodes``, checking they are nodes.

        The graph-side entry to :func:`partition`, which is where the operation and its
        meaning are described. A layout calls that one directly, on its own storage order.

        Raises:
            ValueError: if ``nodes`` names a node this graph does not have, which would
                otherwise sort every clique into ``outside`` without complaint.
        """
        stray = sorted(i for i in nodes if not 0 <= i < self.n_nodes)
        if stray:
            msg = f"nodes {stray} are not among the graph's {self.n_nodes}"
            raise ValueError(msg)
        return partition(self.cliques, nodes)

    def ascend_level(self) -> Cliques:
        """Drop the root level and reroot at the boundary, reindexed from zero.

        The boundary becomes the new root set, so the resulting graph's levels are this
        graph's shifted down by one, and ascending repeatedly climbs the graph one level
        at a time. Node indices ascend with level here and one level up, so this is a
        shift by :attr:`n_roots` --- node $i$ above is node $i + n_r$ below.
        """
        offset = self.n_roots
        cliques = tuple(tuple(i - offset for i in c) for c in self.deep_cliques)
        return Cliques(self.n_nodes - offset, len(self.boundary), cliques)

    # Private

    def _validate(self) -> None:
        """Reject the covers that would be read as some *other* cover.

        Every check here rules out an input that goes on to describe a different graph
        without failing: an empty clique is vacuously both root and deep, a repeated or
        negative index silently renames a node, a duplicate clique doubles a parameter
        block, and an unreachable node has no level and so vanishes from
        :meth:`ascend_level`. Nodes numbered out of level order make the cliques one level
        up a *permutation* of an index range rather than a shift of it, which no manifold
        laying out ``[root | cross | deep]`` can honour. Nothing here guards against an
        error that would raise on its own.
        """
        if not 1 <= self.n_roots <= self.n_nodes:
            raise ValueError(
                f"n_roots must be in 1..{self.n_nodes}, got {self.n_roots}"
            )
        for clique in self.cliques:
            for i in clique:
                if not 0 <= i < self.n_nodes:
                    raise ValueError(
                        f"clique {clique} references out-of-range node {i}"
                    )
        if len(set(self.cliques)) != len(self.cliques):
            duplicates = sorted({c for c in self.cliques if self.cliques.count(c) > 1})
            raise ValueError(f"duplicate cliques: {duplicates}")
        node_levels = self.node_levels
        for i, level in enumerate(node_levels):
            if level < 0:
                raise ValueError(f"node {i} is not reachable from the root nodes")
        for i in range(1, self.n_nodes):
            if node_levels[i] < node_levels[i - 1]:
                msg = f"node {i} is at level {node_levels[i]}"
                msg += f" but node {i - 1} is at level {node_levels[i - 1]}"
                raise ValueError(f"{msg}: number the nodes by level, roots first")


### Partitions ###


def partition(
    cliques: tuple[tuple[int, ...], ...], nodes: Collection[int]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Sort a clique cover by how each of its cliques meets ``nodes``.

    Returns three groups --- the cliques lying wholly inside $S$, those crossing its
    boundary, and those lying wholly outside --- as **positions** in the cover rather than
    as cliques. Positions are what a coordinate split consumes, and the cliques are always
    recoverable by indexing ``cliques``. ``nodes`` is taken as a set: order and repetition
    in it mean nothing.

    Mathematically, this is the partition of the cover induced by the vertex cut
    $(S, V \\setminus S)$, whose middle group is the cut-set. Both of the library's
    structural splits are instances. At $S = R$ it is the ``[root | cross | deep]``
    grouping a clique manifold stores its coordinates in; at $S = \\{k\\}$ it is the
    re-rooting that reads a mixture of harmoniums off a harmonium's parameters. Being one
    operation rather than two is why a layout only ever needs one splitting rule.

    The split is total --- every cover admits it. :func:`crossing_rows`, which the cut view
    additionally needs, is not.

    Free-standing because the order that positions count in is the caller's: a graph
    splits its own :attr:`Cliques.cliques`, while a parameter layout splits its blocks in
    *storage* order, so that the positions it computes and the dimensions it slices come
    from one list --- and those two orders genuinely differ, since a graph normalizes its
    cover at construction. :meth:`Cliques.partition` is the graph-side entry point.
    """
    node_set = frozenset(nodes)
    inside: list[int] = []
    crossing: list[int] = []
    outside: list[int] = []
    for pos, clique in enumerate(cliques):
        members = set(clique)
        if members <= node_set:
            inside.append(pos)
        elif members & node_set:
            crossing.append(pos)
        else:
            outside.append(pos)
    return tuple(inside), tuple(crossing), tuple(outside)


def crossing_rows(
    cliques: tuple[tuple[int, ...], ...], nodes: Collection[int]
) -> tuple[int, ...]:
    """For each crossing clique, the position among the outside ones of $c \\setminus S$.

    What turns the crossing group of :func:`partition` into a matrix: each crossing clique
    becomes the row block sitting above the outside clique it reduces to once $S$ is
    deleted, so the rows follow the outside group's order. Only a cut view needs this ---
    a split that just slices ``[inside | crossing | outside]`` into three spans does not,
    which is why this is a second pass over the cover and not part of the first.

    Injective only when $|S| = 1$: two crossing cliques with the same near part $P$ would
    both be $P \\cup S$, hence the same clique. At $|S| > 1$ they can collide, as a
    two-root graph's two crossing cliques do when both reach the same node above.

    Raises:
        ValueError: if some crossing clique's near part is not itself in the cover, so
            there is no block for it to be a row of. A bare edge ``((0, 1),)`` split at
            ``{0}`` is the smallest case: the graph is legitimate and its crossing group is
            well defined, but node ``1`` carries no block.
    """
    node_set = frozenset(nodes)
    _, crossing, outside = partition(cliques, node_set)
    outside_at = {cliques[i]: pos for pos, i in enumerate(outside)}
    rows: list[int] = []
    for i in crossing:
        near_part = tuple(j for j in cliques[i] if j not in node_set)
        if near_part not in outside_at:
            msg = f"clique {cliques[i]} crosses the cut, near part {near_part}"
            raise ValueError(f"{msg} is not itself a clique, so it has no row block")
        rows.append(outside_at[near_part])
    return tuple(rows)


### Cuts ###


@dataclass(frozen=True)
class CliqueCut:
    """Where each of a layout's blocks lands when it is re-viewed across a one-node cut.

    Integers only, so a cut is a block permutation and nothing else --- which is why one
    applies to any coordinate system alike, natural and mean coordinates included. Built by
    :func:`cut_indices`, where the two conditions a cut has to satisfy are checked, so
    holding one of these is holding a cut that is known to be consistent with the layout's
    dimensions.

    The array operations that read a cut ---
    :func:`~goal.geometry.exponential_family.clique.project_cut` and
    :func:`~goal.geometry.exponential_family.clique.join_cut` --- live one layer up, because
    this module holds no JAX. That is a dependency boundary, not a conceptual one.
    """

    # Fields

    clique_dims: tuple[int, ...]
    """Block sizes of the layout being re-viewed, in its own order."""

    near_idx: tuple[int, ...]
    """Positions of the blocks lying wholly on the near side."""

    cross_idx: tuple[int, ...]
    """Positions of the blocks crossing the cut."""

    far_idx: tuple[int, ...]
    """Positions of the blocks lying wholly on the far side."""

    cross_rows: tuple[int, ...]
    """For each crossing block, the position *within* :attr:`near_idx` of its near part."""

    n_cols: int
    """Width of the crossing matrix: the far side's total dimension."""


def cut_indices(
    cliques: tuple[tuple[int, ...], ...],
    dims: tuple[int, ...],
    far_node: int,
    n_nodes: int,
) -> CliqueCut:
    """Re-group a layout's blocks with ``far_node`` split off instead of the root nodes.

    The re-rooting isomorphism, as indices: the same coordinates read as
    ``(near | crossing | far)`` where the near side is everything not touching
    ``far_node``, the far side is that node's own block, and the crossing blocks become one
    matrix whose rows follow the near side's order. For a mixture of harmoniums cut at its
    category node, the near side is exactly the base harmonium's parameter vector, which is
    what makes the mixture view and the graph view two readings of one array.

    ``cliques`` and ``dims`` are parallel and in the layout's *storage* order, so the
    positions computed here and the dimensions they select come from one list. This is
    :func:`partition` at a single node plus the two conditions a matrix view needs, so it
    is not a generalization of a level split --- a level's cross span couples the root set
    to the *boundary* and may address only a subspace of it.

    Cutting **one** node is what makes the row assignment total: two crossing cliques
    sharing a near part $P$ would both be $P \\cup \\{far\\}$, hence the same clique, so
    distinct crossing cliques always land on distinct rows.

    Raises:
        ValueError: if ``far_node`` is not a node of the graph, or is its only node; if it
            carries no block of its own, so the cut has no columns; if a crossing clique's
            near part is not itself a clique, so it has no row block; or if a crossing
            clique does not couple the whole far side, so the crossing blocks do not share
            one column axis.
    """
    if not 0 <= far_node < n_nodes:
        msg = f"far_node {far_node} is not one of the graph's {n_nodes} nodes"
        raise ValueError(msg)
    if n_nodes == 1:
        raise ValueError("cutting the only node off leaves no near side")

    far_idx, cross_idx, near_idx = partition(cliques, (far_node,))
    if not far_idx:
        msg = f"far_node {far_node} has no block of its own"
        raise ValueError(f"{msg}: the cut would have no columns")

    n_cols = sum(dims[i] for i in far_idx)
    cross_rows = crossing_rows(cliques, (far_node,))
    for pos, i in enumerate(cross_idx):
        height = dims[near_idx[cross_rows[pos]]]
        if dims[i] != height * n_cols:
            msg = f"clique {cliques[i]} has dimension {dims[i]}"
            msg += f", not {height} x {n_cols}"
            raise ValueError(f"{msg}: it does not couple the whole far side")

    return CliqueCut(
        clique_dims=dims,
        near_idx=near_idx,
        cross_idx=cross_idx,
        far_idx=far_idx,
        cross_rows=cross_rows,
        n_cols=n_cols,
    )
