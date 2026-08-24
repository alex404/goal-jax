"""Clique sets: the combinatorial skeleton of a graph.

A ``CliqueSet`` says which nodes a graph has, which of them are roots, and which groups of
nodes form cliques. ``CliqueManifold`` in ``manifold/combinators.py`` turns a clique set
into a parameter layout; this module is purely about the combinatorics, over integer
indices, with no JAX.

Mathematically, a clique set describes an undirected graph $G = (V, E)$ together with a
cover of $V$ by complete subgraphs, and a distinguished root set $R = \\{0, \\ldots, n_r -
1\\} \\subseteq V$. Edges are *derived*: two nodes are adjacent exactly when some clique
contains both, so the cover is the primitive and $E$ falls out of it.

Levels are the distance partition of $G$ rooted at $R$, that is $\\ell(v) = \\min_{r \\in R}
d(v, r)$, computed by breadth-first search. Depth is therefore read off the cover rather
than declared: the path $0 - 1 - 2$ rooted at $\\{0\\}$ has levels $(1, 1, 1)$; adding the
clique $\\{0, 1, 2\\}$ makes $0$ adjacent to $2$ and collapses those to $(1, 2)$; the path
$0 - 2 - 1$ rooted at $\\{0, 1\\}$ gives $(2, 1)$.

Adjacency implies a level gap of at most one, since a shorter route would otherwise exist.
Because cliques are complete this lifts from edges to cliques, so every clique
automatically lies within a single level or crosses two consecutive ones; neither
condition needs checking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CliqueSet:
    """The nodes, root set, and cliques of a graph.

    Cliques are index tuples. Every node must carry a singleton clique, so that the cover
    addresses each node on its own, and every node must be reachable from the root set, so
    that its level is defined.

    Members are sorted within each clique and cliques are sorted among themselves at
    construction, so two equivalent descriptions compare and hash equal. Storage order is
    not layout order: :attr:`canonical_cliques` computes the latter independently.
    """

    # Fields

    n_nodes: int
    """Total number of nodes. Root nodes are ``0`` to ``n_roots - 1``."""

    n_roots: int
    """Number of root nodes --- the set the level partition is measured from."""

    cliques: tuple[tuple[int, ...], ...]
    """The interacting groups of nodes, including one singleton per node."""

    def __post_init__(self) -> None:
        normalized = tuple(sorted(tuple(sorted(set(c))) for c in self.cliques))
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
        """The breadth-first distance of each node from the root set."""
        return tuple(self._bfs_levels())

    @property
    def levels(self) -> tuple[tuple[int, ...], ...]:
        """Nodes grouped by breadth-first distance from the root set.

        Level ``0`` is exactly the root nodes; level ``k`` is the nodes first reached
        after $k$ steps. The length of this tuple is the depth of the graph.
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

        Mathematically, the cut-set of the partition $(R, V \\setminus R)$: the cliques
        carrying interactions between the root set and everything else. Together with
        :attr:`root_cliques` and :attr:`deep_cliques` these partition the cliques, which
        is what makes :attr:`canonical_cliques` a layout of every block exactly once.
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
        """Cliques holding no root node at all, in this graph's indices.

        The third group of the partition, alongside :attr:`root_cliques` and
        :attr:`cross_cliques`. These are exactly the cliques that survive
        :meth:`ascend_level`, before relabelling; :attr:`canonical_cliques` is what puts
        them in the ascended graph's own order.
        """
        node_levels = self.node_levels
        return tuple(c for c in self.cliques if all(node_levels[i] > 0 for i in c))

    @property
    def boundary(self) -> tuple[int, ...]:
        """The non-root nodes adjacent to the root set.

        Mathematically, the vertex boundary $\\partial R$ of the root set $R$. Cliques are
        complete, so these are exactly the non-root members of :attr:`cross_cliques`,
        and the level gap bound puts them all at level 1.
        """
        node_levels = self.node_levels
        return tuple(
            sorted({i for c in self.cross_cliques for i in c if node_levels[i] > 0})
        )

    @property
    def deep_nodes(self) -> tuple[int, ...]:
        """The non-root nodes, by original index, ordered as :meth:`ascend_level` reindexes them.

        The boundary comes first so that it lands as the new root set, which is why this
        is not plain index order: it doubles as the map from an index one level up back
        to the node it came from.
        """
        node_levels = self.node_levels
        level_1 = [i for i in range(self.n_nodes) if node_levels[i] == 1]
        deep = [i for i in range(self.n_nodes) if node_levels[i] > 1]
        return tuple(level_1 + deep)

    @property
    def canonical_cliques(self) -> tuple[tuple[int, ...], ...]:
        """The cliques in parameter-layout order.

        Root cliques first, then cross cliques, then the cliques surviving
        :meth:`ascend_level` relabelled back to this graph's indices --- with the same
        rule applied recursively one level up. Those three groups partition the cliques,
        so every clique appears exactly once, and the order nests: the ascended graph's
        cliques form a contiguous suffix, in precisely the order that graph would put them
        in on its own.
        """
        head = self.root_cliques + self.cross_cliques
        if len(self.levels) == 1:
            return head
        deep_nodes = self.deep_nodes
        above = self.ascend_level()
        return head + tuple(
            tuple(sorted(deep_nodes[i] for i in c)) for c in above.canonical_cliques
        )

    # Methods

    def ascend_level(self) -> CliqueSet:
        """Drop the root level and reroot at the boundary, reindexed from zero.

        The boundary becomes the new root set, so the resulting graph's levels are this
        graph's shifted down by one, and ascending repeatedly climbs the graph one level
        at a time.
        """
        node_levels = self.node_levels
        deep_nodes = self.deep_nodes
        index = {node: i for i, node in enumerate(deep_nodes)}
        cliques = tuple(
            tuple(index[i] for i in c)
            for c in self.cliques
            if all(node_levels[i] >= 1 for i in c)
        )
        n_roots = sum(1 for i in deep_nodes if node_levels[i] == 1)
        return CliqueSet(len(deep_nodes), n_roots, cliques)

    # Private

    def _bfs_levels(self) -> list[int]:
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
        return levels

    def _validate(self) -> None:
        if self.n_nodes < 1:
            raise ValueError(f"n_nodes must be at least 1, got {self.n_nodes}")
        if not 1 <= self.n_roots <= self.n_nodes:
            raise ValueError(
                f"n_roots must be in 1..{self.n_nodes}, got {self.n_roots}"
            )
        self._validate_cliques()
        self._validate_nodes()

    def _validate_cliques(self) -> None:
        for clique in self.cliques:
            for i in clique:
                if not 0 <= i < self.n_nodes:
                    raise ValueError(
                        f"clique {clique} references out-of-range node {i}"
                    )
        if len(set(self.cliques)) != len(self.cliques):
            duplicates = sorted({c for c in self.cliques if self.cliques.count(c) > 1})
            raise ValueError(f"duplicate cliques: {duplicates}")

    def _validate_nodes(self) -> None:
        for i in range(self.n_nodes):
            if (i,) not in self.cliques:
                raise ValueError(f"node {i} has no singleton clique")
        for i, level in enumerate(self._bfs_levels()):
            if level < 0:
                raise ValueError(f"node {i} is not reachable from the root nodes")
