"""Graphs given by their cliques.

A ``Cliques`` is a graph given by its **cliques**, each a tuple of node labels, in a fixed
order. The list need not hold every clique of the graph: it is whichever cliques the
subclass carries, so singletons, or the smaller cliques inside a larger one, may be absent.
The nodes $V$ are every label some clique names, and two nodes are adjacent exactly when
some clique contains both.

Integer labels throughout. Nothing here knows what occupies a node, and there is no JAX.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain


@dataclass(frozen=True)
class Cliques(ABC):
    """A graph as $(V, C)$: its nodes and its cliques, in a fixed order."""

    # Contract

    @property
    @abstractmethod
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        """$C$, each clique an ascending label tuple, each once, in the order they are stored.

        The labels are exactly $0, \\ldots, n - 1$: offsetting a part by its node count, as
        composites do, relies on it. Single-node families and composites satisfy this by
        construction; a class that declares its cliques directly must keep it.
        """

    # Methods

    @property
    def nodes(self) -> tuple[int, ...]:
        """$V$: every label some clique names."""
        return tuple(sorted(set(chain.from_iterable(self.cliques))))

    @property
    def n_nodes(self) -> int:
        """$|V|$."""
        return len(self.nodes)

    @property
    def graph(self) -> dict[int, tuple[int, ...]]:
        """The graph the cliques present, as adjacency lists.

        A dict with one key per node of :attr:`nodes`, mapping it to the sorted tuple of its
        neighbours; a node that shares no clique with another maps to ``()``. Two nodes are
        adjacent exactly when some clique contains both.
        """
        neighbours: dict[int, set[int]] = {i: set() for i in self.nodes}
        for clique in self.cliques:
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


def shift_clique(clique: tuple[int, ...], offset: int) -> tuple[int, ...]:
    """The clique's labels in a graph that numbers ``offset`` nodes before it.

    A composite stores its parts one after another and numbers their nodes in the same
    order, so a part's labels are its own raised by the node count of the parts stored
    before it.
    """
    return tuple(i + offset for i in clique)
