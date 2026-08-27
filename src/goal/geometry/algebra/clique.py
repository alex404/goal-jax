"""Rooted graphs presented by a clique cover.

A ``Cliques`` is a graph $G = (V, E)$ given as a **cover of $V$ by complete subgraphs**,
together with a distinguished root set $R \\subseteq V$. The cover is what is stored; the
nodes and the edges are both derived, $V$ being every index some clique names and two
nodes adjacent exactly when some clique contains both. Hence the name: a three-way
interaction is one clique, and that is lost the moment a graph is stored as $E$.

A node's *level* is its distance from the root set, $\\ell(v) = \\min_{r \\in R} d(v, r)$,
with $d$ the number of edges on a shortest path; the fibres of $\\ell$ are the *level sets*,
and their number is the depth. Because edges are derived, one step crosses a whole clique:
the path $0 - 1 - 2$ rooted at $\\{0\\}$ has level sets of sizes $(1, 1, 1)$, and adding
the clique $\\{0, 1, 2\\}$ makes $0$ adjacent to $2$ and collapses them to $(1, 2)$.

Adjacent nodes differ by at most one level, and because cliques are complete this lifts
from edges to cliques: every clique lies within one level or crosses two consecutive ones.
No cover can describe a graph where that fails, so it is a fact rather than a condition to
check. Against the root set the cliques therefore fall into three groups --- wholly inside,
crossing, wholly outside --- and those groups partition the cover.

**Node indices are labels, nothing more.** They need not start at zero, run contiguously,
or ascend with level; nothing here reads meaning into their order or their arithmetic.
Membership of the root set is membership, and a graph one level up is a relabelling.

Integer labels throughout. Nothing here knows what occupies a node, and there is no JAX.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque


class Cliques(ABC):
    """A rooted graph as $(V, R, C)$: its nodes, its root set, and its clique cover.

    Subclasses state the cover and the root set. The nodes, adjacency, levels, the
    boundary, the split against the root set, and canonical order are all derived from
    those two.

    Each clique is the ascending label tuple of the nodes it joins. Singletons are optional
    --- a lone edge ``((0, 1),)`` is a legitimate cover.
    """

    # Contract

    @property
    @abstractmethod
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        """$C$, the cover, each clique ascending.

        The order is the subclass's own and is never rearranged here: every group this
        class computes is a tuple of positions into this list.
        """

    @property
    @abstractmethod
    def root_nodes(self) -> frozenset[int]:
        """$R$, the labels the graph is rooted at. Levels are distances from this set."""

    # Properties

    @property
    def nodes(self) -> tuple[int, ...]:
        """$V$, ascending: every label the cover names."""
        return tuple(sorted({i for clique in self.cliques for i in clique}))

    @property
    def n_nodes(self) -> int:
        """$|V|$."""
        return len(self.nodes)

    @property
    def graph(self) -> dict[int, tuple[int, ...]]:
        """The graph the cover presents: node ``i``'s neighbours, ascending, are ``graph[i]``.

        Two nodes are adjacent exactly when some clique contains both. Everything about
        connectivity reads from here.
        """
        neighbours: dict[int, set[int]] = {i: set() for i in self.nodes}
        for clique in self.cliques:
            for i in clique:
                neighbours[i].update(j for j in clique if j != i)
        return {i: tuple(sorted(near)) for i, near in neighbours.items()}

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """:attr:`graph` as a sorted list of pairs, each once."""
        return tuple(
            sorted((i, j) for i, near in self.graph.items() for j in near if i < j)
        )

    @property
    def node_levels(self) -> dict[int, int]:
        """The distance of each node from the root set, $\\min_{r \\in R} d(v, r)$.

        Multi-source breadth-first search: seeding every root at zero means a node's first
        visit is along a shortest path from the nearest root. A node the root set cannot
        reach is absent from the result, as is a root the cover never names.
        """
        neighbours = self.graph
        levels: dict[int, int] = {}
        queue = deque(i for i in self.nodes if i in self.root_nodes)
        for i in queue:
            levels[i] = 0
        while queue:
            i = queue.popleft()
            for j in neighbours[i]:
                if j not in levels:
                    levels[j] = levels[i] + 1
                    queue.append(j)
        return levels

    @property
    def level_sets(self) -> tuple[tuple[int, ...], ...]:
        """Nodes grouped by distance from the root set --- the fibres of :attr:`node_levels`.

        Level ``0`` is the root nodes, level $k$ the nodes at distance $k$, and the length
        of this tuple is the depth.
        """
        node_levels = self.node_levels
        depth = max(node_levels.values()) + 1
        return tuple(
            tuple(i for i in self.nodes if node_levels.get(i) == level)
            for level in range(depth)
        )

    @property
    def root_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques lying wholly within the root set."""
        root, _, _ = self.level_split()
        return tuple(self.cliques[i] for i in root)

    @property
    def cross_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques holding both a root node and a non-root node.

        Mathematically, the cut-set of the partition $(R, V \\setminus R)$.
        """
        _, cross, _ = self.level_split()
        return tuple(self.cliques[i] for i in cross)

    @property
    def deep_cliques(self) -> tuple[tuple[int, ...], ...]:
        """Cliques lying wholly outside the root set, in this graph's labels.

        Relabelling them gives the cover of the graph one level up, whose root set is
        :attr:`boundary`.
        """
        _, _, deep = self.level_split()
        return tuple(self.cliques[i] for i in deep)

    @property
    def boundary(self) -> tuple[int, ...]:
        """The vertex boundary $\\partial R$: the non-root nodes adjacent to the root set.

        These are the root set of the graph one level up, and by the level gap bound they
        are exactly the level-1 nodes. Empty exactly when the graph has depth one.
        """
        node_levels = self.node_levels
        return tuple(i for i in self.nodes if node_levels.get(i) == 1)

    @property
    def canonical_cliques(self) -> tuple[tuple[int, ...], ...]:
        """The cliques ordered by level, the cover's own order kept within each group.

        Level by level from the root set: the cliques lying wholly within level $k$, then
        those crossing from level $k$ to level $k + 1$, then the same rule at level
        $k + 1$. Since a clique lies within one level or crosses two consecutive ones and
        there is no third case, (lowest level touched, whether it crosses) names its group
        and the whole order is one sort.

        The order nests: the cliques above the root set form a contiguous suffix, in
        precisely the order the graph one level up would put them in on its own.
        """
        node_levels = self.node_levels

        def group(indexed: tuple[int, tuple[int, ...]]) -> tuple[int, bool, int]:
            pos, clique = indexed
            levels = [node_levels[i] for i in clique]
            return min(levels), max(levels) > min(levels), pos

        return tuple(c for _, c in sorted(enumerate(self.cliques), key=group))

    # Methods

    def level_split(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """The three groups of the cover, as **positions** in :attr:`cliques`.

        The cliques lying wholly within $R$, those crossing its boundary, and those lying
        wholly outside it. :attr:`root_cliques` and its two siblings read these back as
        cliques; positions are what a caller needs to index anything held alongside the
        cover.
        """
        roots = self.root_nodes
        root: list[int] = []
        cross: list[int] = []
        deep: list[int] = []
        for pos, clique in enumerate(self.cliques):
            inside = [i in roots for i in clique]
            if all(inside):
                root.append(pos)
            elif any(inside):
                cross.append(pos)
            else:
                deep.append(pos)
        return tuple(root), tuple(cross), tuple(deep)

    def same_graph(self, other: Cliques) -> bool:
        """Whether ``other`` presents the same rooted graph: same cover, same root set.

        Compares the two descriptions, not the objects, so subclasses of different types
        can be equal by it. Cover order counts: two covers listing one graph's cliques in
        different orders are not the same graph by this test.
        """
        return self.root_nodes == other.root_nodes and self.cliques == other.cliques
