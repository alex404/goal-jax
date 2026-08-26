"""Clique sets: the combinatorial skeleton of a graph.

A ``CliqueSet`` says which nodes a graph has, which of them are roots, and which groups of
nodes form cliques. ``LinearCliques`` in ``manifold/graphical.py`` turns a clique set
into a parameter layout; this module is purely about the combinatorics, over integer
indices, with no JAX.

Mathematically, a clique set describes an undirected graph $G = (V, E)$ together with a
cover of $V$ by complete subgraphs, and a distinguished root set $R = \\{0, \\ldots, n_r -
1\\} \\subseteq V$. Edges are *derived*: two nodes are adjacent exactly when some clique
contains both, so the cover is the primitive and $E$ falls out of it.

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
:attr:`CliqueSet.canonical_cliques` a layout listing every clique exactly once.

**Nodes are numbered by level**, roots first: $\\ell$ is non-decreasing in the index. This is
a numbering convention rather than a restriction --- sorting the nodes by level achieves it
for any graph --- and it is what makes layout order *readable* rather than computed.
Dropping the root level becomes a shift by $n_r$ instead of a permutation, so the cliques
one level up occupy a contiguous suffix at contiguous indices, and a manifold can splice its
deep span in by renumbering rather than reordering.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CliqueSet:
    """A graph as $(V, R, C)$: its nodes, its root set, and its clique cover.

    Each clique is the index tuple of the nodes it couples, with no manifold attached ---
    :class:`~goal.geometry.manifold.graphical.LinearCliques` is what hangs parameters off
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
    def edges(self) -> tuple[tuple[int, int], ...]:
        """The derived adjacency: every pair of nodes sharing a clique, sorted."""
        pairs = {
            (i, j)
            for clique in self.cliques
            for k, i in enumerate(clique)
            for j in clique[k + 1 :]
        }
        return tuple(sorted(pairs))

    @property
    def node_levels(self) -> tuple[int, ...]:
        """The distance of each node from the root set, $\\min_{r \\in R} d(v, r)$.

        Computed by multi-source breadth-first search: seeding every root at zero means a
        node's first visit is along a shortest path from the nearest root. Unreachable
        nodes come back as ``-1``, which construction rejects, so a constructed clique
        set never holds one.
        """
        neighbours: list[set[int]] = [set() for _ in range(self.n_nodes)]
        for i, j in self.edges:
            neighbours[i].add(j)
            neighbours[j].add(i)

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
    def root_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques lying wholly within the root set."""
        node_levels = self.node_levels
        return tuple(c for c in self.cliques if all(node_levels[i] == 0 for i in c))

    @property
    def cross_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques holding both a root node and a non-root node.

        Mathematically, the cut-set of the partition $(R, V \\setminus R)$ --- the cliques
        carrying interactions between the root set and everything else.
        """
        node_levels = self.node_levels
        return tuple(
            c
            for c in self.cliques
            if any(node_levels[i] == 0 for i in c)
            and any(node_levels[i] > 0 for i in c)
        )

    @property
    def deep_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques lying wholly outside the root set, in this graph's indices.

        Because node indices ascend with level, "outside the root set" is "index at least
        :attr:`n_roots`". These are exactly the cliques that survive :meth:`ascend_level`,
        which renumbers them by $-n_r$ and nothing else.
        """
        node_levels = self.node_levels
        return tuple(c for c in self.cliques if all(node_levels[i] > 0 for i in c))

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

    def ascend_level(self) -> CliqueSet:
        """Drop the root level and reroot at the boundary, reindexed from zero.

        The boundary becomes the new root set, so the resulting graph's levels are this
        graph's shifted down by one, and ascending repeatedly climbs the graph one level
        at a time. Node indices ascend with level here and one level up, so this is a
        shift by :attr:`n_roots` --- node $i$ above is node $i + n_r$ below.
        """
        offset = self.n_roots
        cliques = tuple(tuple(i - offset for i in c) for c in self.deep_cliques)
        return CliqueSet(self.n_nodes - offset, len(self.boundary), cliques)

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
