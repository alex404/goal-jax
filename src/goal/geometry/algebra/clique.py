"""Rooted graphs given by their cliques.

A ``RecursiveCliques`` is a graph described by two things a subclass states: its **cliques**,
each a set of node labels, and a nonempty **root set** $R$ of labels that cliques name.
The list need not hold every clique of the graph: it is whichever cliques the subclass
carries, so singletons, or the smaller cliques inside a larger one, may be absent. The
nodes $V$ are every label some clique names, and two nodes are adjacent exactly when some
clique contains both.

**The root set orients the graph.** A node's *level* is its distance from $R$,
$\\ell(v) = \\min_{r \\in R} d(v, r)$, with $d$ the number of edges on a shortest path.
Grouping the nodes by level gives the *level sets*, and their number is the depth.
Adjacent nodes differ by at most one level, so every clique lies within one level or
spans two consecutive ones.

**Representation.** The cliques as a subclass writes them are free-form: their order, the
order of labels within one, repeats, and empty cliques change nothing. Two members read
them. :attr:`RecursiveCliques.level_sets` asks only which labels share a clique, which the writing
cannot change, and :attr:`RecursiveCliques.canonical_cliques` is the one normalized form, each
clique once and grouped by level. Every other member reads these two, so a component with
no root is left out everywhere. :meth:`RecursiveCliques.level_split` coarsens the canonical form to
what a layout stores: the cliques inside $R$, those crossing out of it, and the rest,
which are the cliques of the graph one level up, rooted at the level-1 nodes. Node indices
are labels, and results are sorted tuples, which is only a fixed way of writing a set
down.

**Unchecked requirements.** $R$ is nonempty and contained in $V$. The members fail or
disagree when it does not hold.

Integer labels throughout. Nothing here knows what occupies a node, and there is no JAX.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain


@dataclass(frozen=True)
class RecursiveCliques(ABC):
    """A rooted graph as $(V, R, C)$: its nodes, its root set, and its cliques.

    Subclasses state the cliques and the root set; everything else is derived.
    """

    # Contract

    @property
    @abstractmethod
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        """$C$ as the subclass writes it, in any order and spelling."""

    @property
    @abstractmethod
    def root_nodes(self) -> frozenset[int]:
        """A nonempty $R \\subseteq V$, the labels the graph is rooted at.

        Levels are distances from this set.
        """

    # Properties

    @property
    def nodes(self) -> tuple[int, ...]:
        """$V$: every label in :attr:`level_sets`."""
        return tuple(sorted(chain.from_iterable(self.level_sets)))

    @property
    def n_nodes(self) -> int:
        """$|V|$."""
        return len(self.nodes)

    @property
    def graph(self) -> dict[int, tuple[int, ...]]:
        """The graph the cliques present, as adjacency lists.

        A dict with one key per node of :attr:`nodes`, mapping it to the sorted tuple of its
        neighbours; a node that shares no clique with another maps to ``()``. Two nodes are
        adjacent exactly when some clique of :attr:`canonical_cliques` contains both.
        """
        neighbours: dict[int, set[int]] = {i: set() for i in self.nodes}
        for clique in chain.from_iterable(self.canonical_cliques):
            for i in clique:
                neighbours[i].update(j for j in clique if j != i)
        return {i: tuple(sorted(near)) for i, near in neighbours.items()}

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """$E$, the same adjacency as :attr:`graph` written as pairs.

        Each edge appears once, as $(i, j)$ with $i < j$, and the pairs are sorted. Both
        are only a fixed way of writing the set $E$ down.
        """
        pairs: list[tuple[int, int]] = []
        for i, near in self.graph.items():
            pairs.extend((i, j) for j in near if i < j)
        return tuple(sorted(pairs))

    @property
    def level_sets(self) -> tuple[tuple[int, ...], ...]:
        """Nodes grouped by their distance from the root set.

        Level $0$ is $R$, and level $k + 1$ is every label sharing a clique with level $k$
        that is not already in an earlier level. The length of this tuple is the depth.
        """
        seen = frontier = self.root_nodes
        levels: list[tuple[int, ...]] = []
        while frontier:
            levels.append(tuple(sorted(frontier)))
            overlapping = [c for c in self.cliques if not frontier.isdisjoint(c)]
            frontier = frozenset().union(*overlapping) - seen
            seen |= frontier
        return tuple(levels)

    @property
    def canonical_cliques(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """The cliques in normalized form, grouped by level.

        Each nonempty clique appears once, as its sorted labels; a clique in a component
        with no root has no level and is left out. The groups run within level 0, crossing
        0 to 1, within level 1, and so on: a clique lies within one level or crosses two
        consecutive ones, so (lowest level, whether it crosses) names its group. A graph of
        depth $n$ has $2n - 1$ groups, any of which may be empty. Dropping the first two
        gives the canonical cliques of the graph one level up.
        """
        node_to_level: dict[int, int] = {}
        for k, level in enumerate(self.level_sets):
            node_to_level.update({i: k for i in level})
        clean_cliques = {tuple(sorted(set(c))) for c in self.cliques if c}
        rooted_cliques = sorted(c for c in clean_cliques if c[0] in node_to_level)
        within: list[list[tuple[int, ...]]] = [[] for _ in self.level_sets]
        crossing: list[list[tuple[int, ...]]] = [[] for _ in self.level_sets[1:]]
        for clique in rooted_cliques:
            low = min(node_to_level[i] for i in clique)
            if any(node_to_level[i] > low for i in clique):
                crossing[low].append(clique)
            else:
                within[low].append(clique)
        groups = [within[0]]
        for cross, inner in zip(crossing, within[1:], strict=True):
            groups += [cross, inner]
        return tuple(tuple(group) for group in groups)

    # Methods

    def level_split(
        self,
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
    ]:
        """:attr:`canonical_cliques` coarsened to the root set: inside, crossing, the rest.

        The rest are the cliques of the graph one level up.
        """
        root, *above = self.canonical_cliques
        cross = above[0] if above else ()
        return root, cross, tuple(chain.from_iterable(above[1:]))

    def same_graph(self, other: RecursiveCliques) -> bool:
        """Whether ``other`` has the same root set and :attr:`canonical_cliques`.

        Compares the two descriptions, not the objects, so subclasses of different types
        can be equal by it. One graph described by different cliques --- a triangle as one
        clique or as three edges --- is not the same by this test.
        """
        return (
            self.root_nodes == other.root_nodes
            and self.canonical_cliques == other.canonical_cliques
        )
