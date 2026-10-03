"""Tests for geometry/algebra/clique.py.

Verifies the nodes, adjacency and edges a ``Cliques`` presents. ``Cliques`` is pure Python, so
this file imports no JAX and needs no platform configuration.

``Cliques`` is an ABC and the library ships no instance of it carrying no parameters, so the
concrete graph lives here, as ``Graph``.
"""

from dataclasses import dataclass
from typing import override

import pytest

from goal.geometry import Cliques


@dataclass(frozen=True)
class Graph(Cliques):
    """A graph given by its cliques, stated outright."""

    _cliques: tuple[tuple[int, ...], ...]

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        return self._cliques


# The three model shapes, in their composed numbering.

HMOG = Graph(((0,), (0, 1), (1,), (1, 2), (2,)))
"""x --- y --- k: hierarchical mixture of Gaussians."""

MFA = Graph(((0,), (0, 1), (0, 1, 2), (0, 2), (1,), (1, 2), (2,)))
"""Mixture of factor analyzers: the three-clique makes x --- k an edge."""

CCA = Graph(((0,), (1,), (0, 2), (1, 2), (2,)))
"""Canonical correlation analysis: two observable nodes, one latent."""


class TestGraph:
    """The nodes and adjacency are derived from the cliques."""

    @pytest.mark.parametrize("clique_set", [HMOG, MFA, CCA])
    def test_nodes(self, clique_set: Graph) -> None:
        assert clique_set.nodes == (0, 1, 2)
        assert clique_set.n_nodes == 3

    def test_graph_is_derived_from_the_cliques(self) -> None:
        assert HMOG.graph == {0: (1,), 1: (0, 2), 2: (1,)}
        # The three-clique makes x and k adjacent even though no pair (0, 2) was declared.
        assert MFA.graph == {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        assert CCA.graph == {0: (2,), 1: (2,), 2: (0, 1)}

    def test_edges_are_the_graph_as_pairs(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            pairs = {
                (min(i, j), max(i, j))
                for i, near in clique_set.graph.items()
                for j in near
            }
            assert clique_set.edges == tuple(sorted(pairs))
        assert HMOG.edges == ((0, 1), (1, 2))
        assert MFA.edges == ((0, 1), (0, 2), (1, 2))
        assert CCA.edges == ((0, 2), (1, 2))

    def test_a_lone_node_has_no_neighbours(self) -> None:
        graph = Graph(((0,), (1, 2)))
        assert graph.graph == {0: (), 1: (2,), 2: (1,)}
        assert graph.edges == ((1, 2),)

    def test_singletons_are_optional(self) -> None:
        """An edge alone names both its nodes; their singleton cliques add nothing."""
        bare, full = Graph(((0, 1),)), Graph(((0,), (0, 1), (1,)))
        assert bare.nodes == full.nodes
        assert bare.graph == full.graph

    def test_nodes_are_whatever_the_cliques_name(self) -> None:
        graph = Graph(((5,), (5, 9)))
        assert graph.nodes == (5, 9)
        assert graph.graph == {5: (9,), 9: (5,)}
