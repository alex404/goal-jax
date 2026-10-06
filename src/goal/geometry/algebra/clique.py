"""Graphs given by their cliques.

A ``Cliques`` is a graph given by its **cliques**, each a tuple of node labels, in a fixed
order. The list need not contain every clique of the graph; for example, cliques contained
in a larger one may be absent. The nodes $V$ are the labels that appear in some clique, and
two nodes are adjacent exactly when some clique contains both.

Labels are integers. The module does not depend on JAX or on what a node represents.
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
        """$C$: each clique an ascending tuple of labels, listed once, in storage order.

        The labels must be exactly $0, \\ldots, n - 1$, since composites offset the labels of
        a part by the node count of the parts before it. A class that declares its cliques
        directly must ensure this.
        """

    # Methods

    @property
    def nodes(self) -> tuple[int, ...]:
        """$V$: the labels that appear in some clique."""
        return tuple(sorted(set(chain.from_iterable(self.cliques))))

    @property
    def n_nodes(self) -> int:
        """$|V|$."""
        return len(self.nodes)

    @property
    def graph(self) -> dict[int, tuple[int, ...]]:
        """The graph as adjacency lists: each node mapped to the sorted tuple of its neighbours.

        Two nodes are adjacent exactly when some clique contains both. A node with no
        neighbours maps to ``()``.
        """
        neighbours: dict[int, set[int]] = {i: set() for i in self.nodes}
        for clique in self.cliques:
            for i in clique:
                neighbours[i].update(j for j in clique if j != i)
        return {i: tuple(sorted(near)) for i, near in neighbours.items()}

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """$E$: the edges of :attr:`graph`, each once as $(i, j)$ with $i < j$, in sorted order."""
        pairs: list[tuple[int, int]] = []
        for i, near in self.graph.items():
            pairs.extend((i, j) for j in near if i < j)
        return tuple(sorted(pairs))


def shift_clique(clique: tuple[int, ...], offset: int) -> tuple[int, ...]:
    """The clique's labels in a graph that numbers ``offset`` nodes before it.

    A composite numbers the nodes of its parts in storage order, so the labels of a part
    are offset by the node count of the parts before it.
    """
    return tuple(i + offset for i in clique)
