"""Tests for geometry/algebra/clique.py.

Verifies level derivation by breadth-first search, canonical clique ordering, level
ascent and reindexing, normalization/hashing, and each validation failure. ``CliqueSet`` is pure
Python, so this file imports no JAX and needs no platform configuration.
"""

import pytest

from goal.geometry import CliqueSet


def path(n_nodes: int) -> CliqueSet:
    """The path ``0 --- 1 --- ... --- (n_nodes - 1)`` rooted at node 0, depth ``n_nodes``.

    Test scaffolding for parametrizing over depth. Deliberately not a ``CliqueSet``
    classmethod: level structure does not determine a cover, so no factory keyed on
    hierarchy can exist, and paths are only one of the shapes this module has to serve.
    """
    singletons = tuple((i,) for i in range(n_nodes))
    links = tuple((i, i + 1) for i in range(n_nodes - 1))
    return CliqueSet(n_nodes, 1, singletons + links)


# The three model shapes this design has to cover.

HMOG = path(3)
"""x --- y --- k: hierarchical mixture of Gaussians, levels (1, 1, 1)."""

MFA = CliqueSet(
    n_nodes=3,
    n_roots=1,
    cliques=((0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)),
)
"""Mixture of factor analyzers: the three-clique makes x --- k an edge, levels (1, 2)."""

CCA = CliqueSet(
    n_nodes=3,
    n_roots=2,
    cliques=((0,), (1,), (2,), (0, 2), (1, 2)),
)
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
        self, clique_set: CliqueSet, expected: tuple[tuple[int, ...], ...]
    ) -> None:
        assert clique_set.levels == expected

    @pytest.mark.parametrize(
        ("clique_set", "expected"),
        [(HMOG, (1, 1, 1)), (MFA, (1, 2)), (CCA, (2, 1))],
    )
    def test_level_sizes(
        self, clique_set: CliqueSet, expected: tuple[int, ...]
    ) -> None:
        assert tuple(len(level) for level in clique_set.levels) == expected

    def test_edges_are_derived_from_cliques(self) -> None:
        assert HMOG.edges == ((0, 1), (1, 2))
        # The three-clique induces x --- k even though no pair (0, 2) was declared.
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


class TestCanonicalOrder:
    """Root cliques, then cross cliques, then deep cliques --- recursively."""

    def test_pair_reproduces_harmonium_layout(self) -> None:
        assert path(2).canonical_cliques == ((0,), (0, 1), (1,))

    def test_chain_nests(self) -> None:
        assert HMOG.canonical_cliques == ((0,), (0, 1), (1,), (1, 2), (2,))

    def test_chain_deep_span_matches_ascended_model(self) -> None:
        # The deep span of the full layout is byte-identical to the layout the model one
        # level up produces on its own. This is what lets pst_man be a concrete
        # model with no translation.
        deep_nodes = HMOG.deep_nodes
        deep_span = HMOG.canonical_cliques[
            len(HMOG.root_cliques + HMOG.cross_cliques) :
        ]
        relabelled = tuple(
            tuple(sorted(deep_nodes[i] for i in c))
            for c in HMOG.ascend_level().canonical_cliques
        )
        assert deep_span == relabelled

    def test_mfa_order(self) -> None:
        assert MFA.canonical_cliques == ((0,), (0, 1), (0, 1, 2), (1,), (1, 2), (2,))

    def test_cca_order(self) -> None:
        assert CCA.canonical_cliques == ((0,), (1,), (0, 2), (1, 2), (2,))

    @pytest.mark.parametrize("clique_set", [path(2), HMOG, MFA, CCA])
    def test_canonical_order_is_a_permutation(self, clique_set: CliqueSet) -> None:
        assert sorted(clique_set.canonical_cliques) == sorted(clique_set.cliques)

    def test_clique_groups_partition(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            head = clique_set.root_cliques + clique_set.cross_cliques
            deep_nodes = clique_set.deep_nodes
            deep = tuple(
                tuple(sorted(deep_nodes[i] for i in c))
                for c in clique_set.ascend_level().cliques
            )
            assert sorted(head + deep) == sorted(clique_set.cliques)


class TestTail:
    """Ascending a level reindexes from zero with the boundary becoming the new root set."""

    def test_chain_ascends_to_a_shorter_chain(self) -> None:
        assert path(3).ascend_level() == path(2)
        assert path(4).ascend_level().ascend_level() == path(2)

    def test_deep_nodes_map_back(self) -> None:
        assert HMOG.deep_nodes == (1, 2)
        assert CCA.deep_nodes == (2,)

    def test_ascended_levels_shift_down(self) -> None:
        assert MFA.ascend_level().levels == ((0, 1),)
        assert CCA.ascend_level() == CliqueSet(1, 1, ((0,),))

    def test_ascended_roots_are_the_boundary_level(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            assert clique_set.ascend_level().n_roots == len(clique_set.levels[1])

    @pytest.mark.parametrize("clique_set", [path(2), path(4), HMOG, MFA, CCA])
    def test_ascended_levels_are_this_graphs_levels_minus_one(
        self, clique_set: CliqueSet
    ) -> None:
        """Ascending shifts every remaining node down exactly one level.

        This is what lets ``CliqueManifold.split_level`` be applied repeatedly. It holds
        because a clique joining level $k$ to level $k - 1$ for $k \\geq 2$ cannot also
        contain a level-0 node --- that node would be adjacent to a level-$k$ one --- so
        the connectivity that set the level survives the ascent.
        """
        above = clique_set.ascend_level()
        node_levels = clique_set.node_levels
        for i, node in enumerate(clique_set.deep_nodes):
            assert above.node_levels[i] == node_levels[node] - 1


class TestNormalization:
    """Equivalent descriptions compare and hash equal, and models stay jit-static."""

    def test_member_and_clique_order_do_not_matter(self) -> None:
        scrambled = CliqueSet(3, 1, ((2, 1), (0,), (1, 0), (2,), (1,)))
        assert scrambled == HMOG

    def test_hashable(self) -> None:
        assert hash(path(3)) == hash(HMOG)
        assert {HMOG: "hmog"}[path(3)] == "hmog"

    def test_fields_are_tuples(self) -> None:
        assert isinstance(HMOG.cliques, tuple)
        assert all(isinstance(c, tuple) for c in HMOG.cliques)


class TestValidation:
    """Every failure names the offending node or clique."""

    def test_out_of_range_node(self) -> None:
        with pytest.raises(ValueError, match="out-of-range node 3"):
            CliqueSet(3, 1, ((0,), (1,), (2,), (0, 3)))

    def test_missing_singleton(self) -> None:
        with pytest.raises(ValueError, match="node 2 has no singleton clique"):
            CliqueSet(3, 1, ((0,), (1,), (0, 1), (1, 2)))

    def test_unreachable_node(self) -> None:
        with pytest.raises(ValueError, match="node 1 is not reachable"):
            CliqueSet(3, 1, ((0,), (1,), (2,), (1, 2)))

    def test_duplicate_cliques(self) -> None:
        with pytest.raises(ValueError, match="duplicate cliques"):
            CliqueSet(2, 1, ((0,), (1,), (0, 1), (1, 0)))

    @pytest.mark.parametrize("n_roots", [0, 3])
    def test_bad_root_count(self, n_roots: int) -> None:
        with pytest.raises(ValueError, match=r"n_roots must be in 1\.\.2"):
            CliqueSet(2, n_roots, ((0,), (1,), (0, 1)))

    def test_empty_model(self) -> None:
        with pytest.raises(ValueError, match="n_nodes must be at least 1"):
            CliqueSet(0, 1, ())
