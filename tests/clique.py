"""Tests for geometry/algebra/clique.py.

Verifies levels as distance from the root set, the split a cover induces against its root
set, canonical clique ordering, and level ascent. ``Cliques`` is pure Python, so this file
imports no JAX and needs no platform configuration --- which is why the cut, whose
operations are array work, is tested in ``graphical.py`` instead.

Node indices are labels: the tests use ascending contiguous ones because they are easy to
read, and ``TestRelabelling`` checks that nothing depends on that.

``Cliques`` is an ABC and the library ships no instance of it carrying no parameters: a
cover with nothing laid out on it is a thing to test with, not a thing to model with. So
the concrete cover and the level ascent both live here, as ``Cover`` and ``ascend``.
"""

from dataclasses import dataclass
from typing import ClassVar, override

import pytest

from goal.geometry import Cliques


@dataclass(frozen=True)
class Cover(Cliques):
    """A cover and a root set, both stated outright.

    The test's instance of the ABC. Nothing normalizes or checks the cover, here or in the
    library, so these tuples are exactly what every property reads.
    """

    _cliques: tuple[tuple[int, ...], ...]
    _root_nodes: frozenset[int]

    @property
    @override
    def cliques(self) -> tuple[tuple[int, ...], ...]:
        return self._cliques

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return self._root_nodes


def ascend(clique_set: Cliques) -> Cover:
    """Drop the root level and reroot at the boundary, keeping the labels.

    The boundary becomes the new root set, so the resulting graph's levels are this
    graph's shifted down by one. Labels carry over untouched: a graph one level up is the
    same nodes minus the root level, and nothing needs renumbering to say so.

    A manifold has no use for this: its deep partition is already the graph one level up, as a
    manifold. It is the levels *below* a cover that the library reads, and this is how the
    tests check that reading against the cover one level up.
    """
    return Cover(clique_set.deep_cliques, frozenset(clique_set.boundary))


def path(n_nodes: int) -> Cover:
    """The path ``0 --- 1 --- ... --- (n_nodes - 1)`` rooted at node 0, depth ``n_nodes``.

    Test scaffolding for parametrizing over depth. Deliberately not a ``Cliques``
    classmethod: level structure does not determine a cover, so no factory keyed on
    hierarchy can exist, and paths are only one of the shapes this module has to serve.
    """
    singletons = tuple((i,) for i in range(n_nodes))
    links = tuple((i, i + 1) for i in range(n_nodes - 1))
    return Cover(singletons + links, frozenset({0}))


# The three model shapes this design has to cover.

HMOG = path(3)
"""x --- y --- k: hierarchical mixture of Gaussians, levels (1, 1, 1)."""

MFA = Cover(((0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)), frozenset({0}))
"""Mixture of factor analyzers: the three-clique makes x --- k an edge, levels (1, 2)."""

CCA = Cover(((0,), (1,), (2,), (0, 2), (1, 2)), frozenset({0, 1}))
"""Canonical correlation analysis: two root nodes, one deep, levels (2, 1)."""


class TestLevels:
    """Levels are derived from the cliques, never declared."""

    @pytest.mark.parametrize(
        ("clique_set", "expected"),
        [
            (path(2), ((0,), (1,))),
            (HMOG, ((0,), (1,), (2,))),
            (MFA, ((0,), (1, 2))),
            (CCA, ((0, 1), (2,))),
        ],
    )
    def test_levels(
        self, clique_set: Cliques, expected: tuple[tuple[int, ...], ...]
    ) -> None:
        assert clique_set.level_sets == expected

    @pytest.mark.parametrize(
        ("clique_set", "expected"),
        [(HMOG, (1, 1, 1)), (MFA, (1, 2)), (CCA, (2, 1))],
    )
    def test_level_sizes(self, clique_set: Cliques, expected: tuple[int, ...]) -> None:
        assert tuple(len(level) for level in clique_set.level_sets) == expected

    def test_graph_is_derived_from_the_cover(self) -> None:
        """The cover is stored; the graph it presents is computed from it."""
        assert HMOG.graph == {0: (1,), 1: (0, 2), 2: (1,)}
        # The three-clique makes x and k adjacent even though no pair (0, 2) was declared.
        assert MFA.graph == {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        assert CCA.graph == {0: (2,), 1: (2,), 2: (0, 1)}

    def test_edges_are_the_graph_as_pairs(self) -> None:
        for clique_set in (path(4), HMOG, MFA, CCA):
            pairs = {
                (min(i, j), max(i, j))
                for i, near in clique_set.graph.items()
                for j in near
            }
            assert clique_set.edges == tuple(sorted(pairs))
        assert HMOG.edges == ((0, 1), (1, 2))
        assert MFA.edges == ((0, 1), (0, 2), (1, 2))
        assert CCA.edges == ((0, 2), (1, 2))

    def test_cliques_span_at_most_two_levels(self) -> None:
        for clique_set in (path(4), HMOG, MFA, CCA):
            node_levels = clique_set.node_levels
            for clique in clique_set.cliques:
                spanned = {node_levels[i] for i in clique}
                assert max(spanned) - min(spanned) <= 1

    def test_boundary(self) -> None:
        assert HMOG.boundary == (1,)
        assert MFA.boundary == (1, 2)
        assert CCA.boundary == (2,)


class TestLevelSplit:
    """The split a clique manifold stores its coordinates by.

    ``level_split`` sorts a cover against the root set into positions: the cliques wholly
    within it, those crossing its boundary, and those wholly above. The three
    ``*_cliques`` properties are readbacks of it.
    """

    @pytest.mark.parametrize("name", ["hmog", "mfa", "cca"])
    def test_it_is_the_three_groups(self, name: str) -> None:
        clq = {"hmog": HMOG, "mfa": MFA, "cca": CCA}[name]
        root, cross, deep = clq.level_split()
        assert tuple(clq.cliques[i] for i in root) == clq.root_cliques
        assert tuple(clq.cliques[i] for i in cross) == clq.cross_cliques
        assert tuple(clq.cliques[i] for i in deep) == clq.deep_cliques

    @pytest.mark.parametrize("name", ["hmog", "mfa", "cca"])
    def test_the_groups_partition_the_cover(self, name: str) -> None:
        clq = {"hmog": HMOG, "mfa": MFA, "cca": CCA}[name]
        root, cross, deep = clq.level_split()
        assert sorted(root + cross + deep) == list(range(len(clq.cliques)))

    @pytest.mark.parametrize("name", ["hmog", "mfa", "cca"])
    def test_the_root_set_is_exactly_level_zero(self, name: str) -> None:
        """``level_split`` asks membership of the root set; levels agree with it."""
        clq = {"hmog": HMOG, "mfa": MFA, "cca": CCA}[name]
        levels = clq.node_levels
        for i in clq.nodes:
            assert (i in clq.root_nodes) == (levels[i] == 0)

    def test_a_bare_edge_has_no_root_or_deep_clique(self) -> None:
        """An interaction with no bias on either node: one crossing clique and nothing else.

        The graph is legitimate --- ``Cliques`` does not require singletons --- so the
        split has to stay total on it.
        """
        edge = Cover(((0, 1),), frozenset({0}))
        root, cross, deep = edge.level_split()
        assert (root, cross, deep) == ((), (0,), ())
        assert edge.cross_cliques == ((0, 1),)

    def test_depth_one_puts_everything_in_root(self) -> None:
        every = Cover(((0,), (0, 1), (1,)), frozenset({0, 1}))
        root, cross, deep = every.level_split()
        assert root == (0, 1, 2)
        assert cross == () and deep == ()


class TestCanonicalOrder:
    """Root cliques, then cross cliques, then the same rule one level up."""

    def test_pair_reproduces_harmonium_layout(self) -> None:
        assert path(2).canonical_cliques == ((0,), (0, 1), (1,))

    def test_chain_nests(self) -> None:
        assert HMOG.canonical_cliques == ((0,), (0, 1), (1,), (1, 2), (2,))

    def test_chain_deep_span_matches_ascended_model(self) -> None:
        # The deep partition of the full layout is byte-identical to the layout the model one
        # level up produces on its own. This is what lets pst_man be a concrete
        # model with no translation.
        deep_span = HMOG.canonical_cliques[
            len(HMOG.root_cliques + HMOG.cross_cliques) :
        ]
        assert deep_span == ascend(HMOG).canonical_cliques

    def test_mfa_order(self) -> None:
        """Level 0, then the two cliques crossing into level 1, then level 1.

        ``MFA`` is written here in an order no model would store it in, so this also pins
        what canonical order does *not* do: within the level-1 group it keeps the cover's
        own order, ``(1,) (2,) (1, 2)``, rather than sorting it.
        """
        assert MFA.canonical_cliques == ((0,), (0, 1), (0, 1, 2), (1,), (2,), (1, 2))

    def test_cca_order(self) -> None:
        assert CCA.canonical_cliques == ((0,), (1,), (0, 2), (1, 2), (2,))

    @pytest.mark.parametrize("clique_set", [path(2), HMOG, MFA, CCA])
    def test_canonical_order_is_a_permutation(self, clique_set: Cliques) -> None:
        assert sorted(clique_set.canonical_cliques) == sorted(clique_set.cliques)

    def test_clique_groups_partition(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            head = clique_set.root_cliques + clique_set.cross_cliques
            deep = ascend(clique_set).cliques
            assert sorted(head + deep) == sorted(clique_set.cliques)


class TestTail:
    """Ascending a level drops the root level, the boundary becoming the new root set."""

    def test_chain_ascends_to_a_shorter_chain(self) -> None:
        assert ascend(path(3)) == Cover(((1,), (2,), (1, 2)), frozenset({1}))
        assert ascend(ascend(path(4))) == Cover(((2,), (3,), (2, 3)), frozenset({2}))

    def test_ascent_keeps_the_labels(self) -> None:
        """One level up is the same nodes minus the root level, under the same names.

        Nothing renumbers, so the graph's clique order and a manifold's clique order
        cannot come apart between levels by disagreeing about which node is which.
        """
        for clique_set in (HMOG, MFA, CCA):
            above = ascend(clique_set)
            assert above.cliques == clique_set.deep_cliques
            assert set(above.nodes) <= set(clique_set.nodes)
            assert above.n_nodes == clique_set.n_nodes - len(clique_set.root_nodes)

    def test_ascended_levels_shift_down(self) -> None:
        assert ascend(MFA).level_sets == ((1, 2),)
        assert ascend(CCA) == Cover(((2,),), frozenset({2}))

    def test_ascended_roots_are_the_boundary(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            assert ascend(clique_set).root_nodes == frozenset(clique_set.boundary)

    @pytest.mark.parametrize("clique_set", [path(2), path(4), HMOG, MFA, CCA])
    def test_ascended_levels_are_this_graphs_levels_minus_one(
        self, clique_set: Cliques
    ) -> None:
        """Ascending shifts every remaining node down exactly one level.

        This is what lets ``LevelCliques.split_level`` be applied repeatedly. It holds
        because a clique joining level $k$ to level $k - 1$ for $k \\geq 2$ cannot also
        contain a level-0 node --- that node would be adjacent to a level-$k$ one --- so
        the connectivity that set the level survives the ascent.
        """
        above = ascend(clique_set)
        node_levels = clique_set.node_levels
        for i in above.nodes:
            assert above.node_levels[i] == node_levels[i] - 1


class TestAtomicShapes:
    """The two irreducible covers: a lone node, and a lone edge.

    These are the base cases of a self-similar clique manifold. The edge is why a
    singleton clique cannot be required --- an interaction carries a coupling and no
    bias, so its cover is one clique with no singletons at all.
    """

    def test_node_atom(self) -> None:
        node = Cover(((0,),), frozenset({0}))
        assert node.level_sets == ((0,),)
        assert node.edges == ()
        assert node.canonical_cliques == ((0,),)

    def test_edge_atom(self) -> None:
        edge = Cover(((0, 1),), frozenset({0}))
        assert edge.level_sets == ((0,), (1,))
        assert edge.edges == ((0, 1),)
        assert edge.canonical_cliques == ((0, 1),)
        assert edge.root_cliques == ()
        assert edge.cross_cliques == ((0, 1),)
        assert edge.deep_cliques == ()

    def test_multi_clique_edge_atom(self) -> None:
        """A block map over three nodes: the cover MFA's interaction needs."""
        edge = Cover(((0, 1), (0, 1, 2), (0, 2)), frozenset({0}))
        assert edge.level_sets == ((0,), (1, 2))
        assert edge.cross_cliques == ((0, 1), (0, 1, 2), (0, 2))

    def test_an_unreachable_component_simply_has_no_level(self) -> None:
        """Two disjoint edges: the second pair has no path to the root, so no level.

        Nothing rejects it --- there is no validation pass --- and the nodes that *are*
        reachable are unaffected.
        """
        split = Cover(((0, 1), (2, 3)), frozenset({0}))
        assert split.nodes == (0, 1, 2, 3)
        assert split.node_levels == {0: 0, 1: 1}
        assert split.level_sets == ((0,), (1,))

    def test_singleton_is_optional_not_forbidden(self) -> None:
        """Dropping the requirement must not make a node-only cover invalid."""
        mixed = Cover(((0,), (0, 1), (1, 2)), frozenset({0}))
        assert mixed.level_sets == ((0,), (1,), (2,))


class TestRelabelling:
    """Node indices are labels: nothing reads meaning into their order or arithmetic.

    Each test takes a graph the rest of this file writes with contiguous, level-ordered
    labels, renames every node through a map that is neither, and checks the derived
    structure is the same graph under the same renaming. Contiguity and level order are
    conveniences for reading the tests, not conditions the code relies on.
    """

    # x --- y --- k with labels that skip, start high, and run against level order.
    RENAME: ClassVar[dict[int, int]] = {0: 30, 1: 7, 2: 19}

    @classmethod
    def _renamed(cls) -> Cover:
        cliques = tuple(tuple(sorted(cls.RENAME[i] for i in c)) for c in HMOG.cliques)
        return Cover(cliques, frozenset({cls.RENAME[0]}))

    def test_nodes_are_whatever_the_cover_names(self) -> None:
        odd = self._renamed()
        assert odd.nodes == (7, 19, 30)
        assert odd.n_nodes == 3

    def test_levels_follow_the_graph_not_the_labels(self) -> None:
        odd = self._renamed()
        assert odd.node_levels == {30: 0, 7: 1, 19: 2}
        assert odd.level_sets == ((30,), (7,), (19,))
        assert odd.boundary == (7,)

    def test_the_split_is_the_same_split(self) -> None:
        odd = self._renamed()
        assert odd.level_split() == HMOG.level_split()

    def test_canonical_order_is_the_same_order(self) -> None:
        odd = self._renamed()
        expected = tuple(
            tuple(sorted(self.RENAME[i] for i in c)) for c in HMOG.canonical_cliques
        )
        assert odd.canonical_cliques == expected

    def test_ascent_still_reroots_at_the_boundary(self) -> None:
        above = ascend(self._renamed())
        assert above.root_nodes == frozenset({7})
        assert above.nodes == (7, 19)
        assert above.node_levels == {7: 0, 19: 1}

    def test_a_root_set_that_is_not_the_lowest_labels(self) -> None:
        """Rooting at the *highest* label: level order and index order run opposite."""
        rooted_high = Cover(HMOG.cliques, frozenset({2}))
        assert rooted_high.node_levels == {2: 0, 1: 1, 0: 2}
        assert rooted_high.level_sets == ((2,), (1,), (0,))
        assert rooted_high.canonical_cliques == ((2,), (1, 2), (1,), (0, 1), (0,))
