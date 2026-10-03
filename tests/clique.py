"""Tests for geometry/algebra/clique.py.

Verifies levels as distance from the root set, the split a cover induces against its root
set, canonical clique ordering, and level ascent. ``RecursiveCliques`` is pure Python, so this file
imports no JAX and needs no platform configuration.

Node indices are labels: the tests use ascending contiguous ones because they are easy to
read, and ``TestRelabelling`` checks that nothing depends on that.

``RecursiveCliques`` is an ABC and the library ships no instance of it carrying no parameters: a
cover with nothing laid out on it is a thing to test with, not a thing to model with. So
the concrete cover and the level ascent both live here, as ``Cover`` and ``ascend``.
"""

from dataclasses import dataclass
from typing import ClassVar, override

import pytest

from goal.geometry import RecursiveCliques


@dataclass(frozen=True)
class Cover(RecursiveCliques):
    """A cover and a root set, both stated outright.

    The test's instance of the ABC. Nothing normalizes or checks the cover, here or in the
    library, so these tuples are exactly what every property reads.
    """

    _raw_cliques: tuple[tuple[int, ...], ...]
    _root_nodes: frozenset[int]

    @property
    @override
    def raw_cliques(self) -> tuple[tuple[int, ...], ...]:
        return self._raw_cliques

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return self._root_nodes


def ascend(clique_set: RecursiveCliques) -> Cover:
    """Drop the root level and reroot at the level-1 nodes, keeping the labels.

    The level-1 nodes become the new root set, so the resulting graph's levels are this
    graph's shifted down by one. Labels carry over untouched: a graph one level up is the
    same nodes minus the root level, and nothing needs renumbering to say so.

    A manifold has no use for this: its deep partition is already the graph one level up, as a
    manifold. It is the levels *below* a cover that the library reads, and this is how the
    tests check that reading against the cover one level up.
    """
    return Cover(clique_set.level_split()[2], frozenset(clique_set.level_sets[1]))


def path(n_nodes: int) -> Cover:
    """The path ``0 --- 1 --- ... --- (n_nodes - 1)`` rooted at node 0, depth ``n_nodes``.

    Test scaffolding for parametrizing over depth. Deliberately not a ``RecursiveCliques``
    classmethod: level structure does not determine a cover, so no factory keyed on
    hierarchy can exist, and paths are only one of the shapes this module has to serve.
    """
    singletons = tuple((i,) for i in range(n_nodes))
    links = tuple((i, i + 1) for i in range(n_nodes - 1))
    return Cover(singletons + links, frozenset({0}))


def levels_of(clique_set: RecursiveCliques) -> dict[int, int]:
    """Each node's level, read off ``level_sets``, for tests that look nodes up one by one."""
    return {i: k for k, level in enumerate(clique_set.level_sets) for i in level}


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
        self, clique_set: RecursiveCliques, expected: tuple[tuple[int, ...], ...]
    ) -> None:
        assert clique_set.level_sets == expected

    @pytest.mark.parametrize(
        ("clique_set", "expected"),
        [(HMOG, (1, 1, 1)), (MFA, (1, 2)), (CCA, (2, 1))],
    )
    def test_level_sizes(
        self, clique_set: RecursiveCliques, expected: tuple[int, ...]
    ) -> None:
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
            levels = levels_of(clique_set)
            for clique in clique_set.raw_cliques:
                spanned = {levels[i] for i in clique}
                assert max(spanned) - min(spanned) <= 1


class TestLevelSplit:
    """The cliques split against the root set: inside it, crossing, and outside."""

    @pytest.mark.parametrize("name", ["hmog", "mfa", "cca"])
    def test_the_groups_partition_the_cliques(self, name: str) -> None:
        clq = {"hmog": HMOG, "mfa": MFA, "cca": CCA}[name]
        root, cross, deep = clq.level_split()
        flat = [c for group in clq.level_cliques for c in group]
        assert sorted(root + cross + deep) == sorted(flat)

    @pytest.mark.parametrize("name", ["hmog", "mfa", "cca"])
    def test_the_root_set_is_exactly_level_zero(self, name: str) -> None:
        """``level_split`` asks membership of the root set; levels agree with it."""
        clq = {"hmog": HMOG, "mfa": MFA, "cca": CCA}[name]
        levels = levels_of(clq)
        for i in clq.nodes:
            assert (i in clq.root_nodes) == (levels[i] == 0)

    def test_mfa_split(self) -> None:
        assert MFA.level_split() == (
            ((0,),),
            ((0, 1), (0, 1, 2)),
            ((1,), (1, 2), (2,)),
        )

    def test_a_bare_edge_has_no_root_or_deep_clique(self) -> None:
        """An interaction with no bias on either node: one crossing clique and nothing else.

        The graph is legitimate --- ``RecursiveCliques`` does not require singletons --- so the
        split has to stay total on it.
        """
        edge = Cover(((0, 1),), frozenset({0}))
        assert edge.level_split() == ((), ((0, 1),), ())

    def test_depth_one_puts_everything_in_root(self) -> None:
        every = Cover(((0,), (0, 1), (1,)), frozenset({0, 1}))
        assert every.level_split() == (((0,), (0, 1), (1,)), (), ())


class TestCanonicalOrder:
    """Within level 0, crossing 0 to 1, within level 1, and so on up the graph."""

    def test_pair_reproduces_harmonium_layout(self) -> None:
        assert path(2).level_cliques == (((0,),), ((0, 1),), ((1,),))

    def test_chain_nests(self) -> None:
        assert HMOG.level_cliques == (
            ((0,),),
            ((0, 1),),
            ((1,),),
            ((1, 2),),
            ((2,),),
        )

    @pytest.mark.parametrize("clique_set", [path(2), path(4), HMOG, MFA, CCA])
    def test_the_groups_above_the_root_are_the_ascended_graphs(
        self, clique_set: RecursiveCliques
    ) -> None:
        """The deep partition of a layout is the layout one level up, on its own."""
        above = ascend(clique_set).level_cliques
        assert clique_set.level_cliques[2:] == above

    def test_mfa_order(self) -> None:
        """A fork: both crossing cliques couple into level 1, which holds y and k together."""
        assert MFA.level_cliques == (
            ((0,),),
            ((0, 1), (0, 1, 2)),
            ((1,), (1, 2), (2,)),
        )

    def test_cca_order(self) -> None:
        assert CCA.level_cliques == (((0,), (1,)), ((0, 2), (1, 2)), ((2,),))

    def test_depth_one_has_one_group(self) -> None:
        every = Cover(((0,), (0, 1), (1,)), frozenset({0, 1}))
        assert every.level_cliques == (((0,), (0, 1), (1,)),)

    def test_an_empty_group_is_kept(self) -> None:
        """No clique lies within level 1 of a bare chain, but its slot stays."""
        chain = Cover(((0, 1), (1, 2)), frozenset({0}))
        assert chain.level_cliques == ((), ((0, 1),), (), ((1, 2),), ())

    @pytest.mark.parametrize("clique_set", [path(2), HMOG, MFA, CCA])
    def test_each_clique_appears_once(self, clique_set: RecursiveCliques) -> None:
        flat = [c for group in clique_set.level_cliques for c in group]
        assert sorted(flat) == sorted(set(clique_set.raw_cliques))

    def test_cliques_are_the_level_cliques_flattened(self) -> None:
        """The stored order: CCA's root cliques come before both crossing cliques."""
        assert CCA.cliques == ((0,), (1,), (0, 2), (1, 2), (2,))


class TestTail:
    """Ascending a level drops the root level, the level-1 nodes becoming the new roots."""

    def test_chain_ascends_to_a_shorter_chain(self) -> None:
        assert ascend(path(3)).same_graph(Cover(((1,), (2,), (1, 2)), frozenset({1})))
        above = ascend(ascend(path(4)))
        assert above.same_graph(Cover(((2,), (3,), (2, 3)), frozenset({2})))

    def test_ascent_keeps_the_labels(self) -> None:
        """One level up is the same nodes minus the root level, under the same names."""
        for clique_set in (HMOG, MFA, CCA):
            above = ascend(clique_set)
            assert sorted(above.raw_cliques) == sorted(clique_set.level_split()[2])
            assert set(above.nodes) <= set(clique_set.nodes)
            assert above.n_nodes == clique_set.n_nodes - len(clique_set.root_nodes)

    def test_ascended_levels_shift_down(self) -> None:
        assert ascend(MFA).level_sets == ((1, 2),)
        assert ascend(CCA).same_graph(Cover(((2,),), frozenset({2})))

    def test_ascended_roots_are_the_level_one_nodes(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            assert ascend(clique_set).root_nodes == frozenset(clique_set.level_sets[1])

    @pytest.mark.parametrize("clique_set", [path(2), path(4), HMOG, MFA, CCA])
    def test_ascended_levels_are_this_graphs_levels_minus_one(
        self, clique_set: RecursiveCliques
    ) -> None:
        """Ascending shifts every remaining node down exactly one level.

        This is what lets ``RecursiveLinearCliques.split_level`` be applied repeatedly. It holds
        because a clique joining level $k$ to level $k - 1$ for $k \\geq 2$ cannot also
        contain a level-0 node --- that node would be adjacent to a level-$k$ one --- so
        the connectivity that set the level survives the ascent.
        """
        above = ascend(clique_set)
        levels, levels_above = levels_of(clique_set), levels_of(above)
        for i in above.nodes:
            assert levels_above[i] == levels[i] - 1


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
        assert node.level_cliques == (((0,),),)

    def test_edge_atom(self) -> None:
        edge = Cover(((0, 1),), frozenset({0}))
        assert edge.level_sets == ((0,), (1,))
        assert edge.edges == ((0, 1),)
        assert edge.level_cliques == ((), ((0, 1),), ())
        assert edge.level_split() == ((), ((0, 1),), ())

    def test_multi_clique_edge_atom(self) -> None:
        """A block map over three nodes: the cover MFA's interaction needs."""
        edge = Cover(((0, 1), (0, 1, 2), (0, 2)), frozenset({0}))
        assert edge.level_sets == ((0,), (1, 2))
        assert edge.level_split() == ((), ((0, 1), (0, 1, 2), (0, 2)), ())

    def test_a_component_without_a_root_is_left_out_everywhere(self) -> None:
        """Two disjoint edges, rooted in one: the other is in no member."""
        split = Cover(((0, 1), (2, 3)), frozenset({0}))
        assert split.level_sets == ((0,), (1,))
        assert split.nodes == (0, 1)
        assert split.graph == {0: (1,), 1: (0,)}
        assert split.level_cliques == ((), ((0, 1),), ())

    def test_components_are_fine_when_each_has_a_root(self) -> None:
        """Two disjoint edges, one root in each: levels are measured from either root."""
        split = Cover(((0, 1), (2, 3)), frozenset({0, 2}))
        assert split.level_sets == ((0, 2), (1, 3))
        assert split.level_cliques == ((), ((0, 1), (2, 3)), ())

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
        cliques = tuple(
            tuple(sorted(cls.RENAME[i] for i in c)) for c in HMOG.raw_cliques
        )
        return Cover(cliques, frozenset({cls.RENAME[0]}))

    def test_nodes_are_whatever_the_cover_names(self) -> None:
        odd = self._renamed()
        assert odd.nodes == (7, 19, 30)
        assert odd.n_nodes == 3

    def test_levels_follow_the_graph_not_the_labels(self) -> None:
        odd = self._renamed()
        assert odd.level_sets == ((30,), (7,), (19,))

    @classmethod
    def _rename(cls, cliques: tuple[tuple[int, ...], ...]) -> set[frozenset[int]]:
        return {frozenset(cls.RENAME[i] for i in c) for c in cliques}

    def test_the_split_is_the_same_split(self) -> None:
        odd = self._renamed()
        for mine, theirs in zip(odd.level_split(), HMOG.level_split(), strict=True):
            assert {frozenset(c) for c in mine} == self._rename(theirs)

    def test_canonical_groups_are_the_same_groups(self) -> None:
        odd = self._renamed()
        pairs = zip(odd.level_cliques, HMOG.level_cliques, strict=True)
        for mine, theirs in pairs:
            assert {frozenset(c) for c in mine} == self._rename(theirs)

    def test_ascent_still_reroots_at_the_level_one_nodes(self) -> None:
        above = ascend(self._renamed())
        assert above.root_nodes == frozenset({7})
        assert above.nodes == (7, 19)
        assert above.level_sets == ((7,), (19,))

    def test_a_root_set_that_is_not_the_lowest_labels(self) -> None:
        """Rooting at the *highest* label: level order and index order run opposite."""
        rooted_high = Cover(HMOG.raw_cliques, frozenset({2}))
        assert rooted_high.level_sets == ((2,), (1,), (0,))
        assert rooted_high.level_cliques == (
            ((2,),),
            ((1, 2),),
            ((1,),),
            ((0, 1),),
            ((0,),),
        )


class TestSpelling:
    """How the raw cliques are written changes nothing but :attr:`~RecursiveCliques.raw_cliques` itself.

    Clique order, label order within a clique, repeated labels, repeated cliques, and empty
    cliques all normalize away in :attr:`~RecursiveCliques.level_cliques`.
    """

    PLAIN: ClassVar[Cover] = Cover(((0,), (0, 1), (1,)), frozenset({0}))
    RESPELT: ClassVar[Cover] = Cover(
        ((1,), (1, 0), (0,), (), (0, 0, 1), (1,)), frozenset({0})
    )

    def test_the_cliques_normalize(self) -> None:
        assert self.RESPELT.level_cliques == (((0,),), ((0, 1),), ((1,),))

    def test_every_derived_member_agrees(self) -> None:
        for name in (
            "nodes",
            "graph",
            "edges",
            "level_sets",
            "level_cliques",
            "cliques",
        ):
            assert getattr(self.RESPELT, name) == getattr(self.PLAIN, name), name
        assert self.RESPELT.level_split() == self.PLAIN.level_split()

    def test_same_graph_ignores_spelling(self) -> None:
        assert self.PLAIN.same_graph(self.RESPELT)

    def test_same_graph_distinguishes_a_triangle_from_its_edges(self) -> None:
        triangle = Cover(((0, 1, 2),), frozenset({0}))
        edges = Cover(((0, 1), (0, 2), (1, 2)), frozenset({0}))
        assert triangle.graph == edges.graph
        assert not triangle.same_graph(edges)

    def test_same_graph_compares_roots(self) -> None:
        assert not self.PLAIN.same_graph(Cover(self.PLAIN.raw_cliques, frozenset({1})))
