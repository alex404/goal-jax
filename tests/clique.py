"""Tests for geometry/algebra/clique.py.

Verifies level derivation by breadth-first search, canonical clique ordering, tail
reindexing, normalization/hashing, and each validation failure. ``CliqueSet`` is pure
Python, so this file imports no JAX and needs no platform configuration.
"""

import pytest

from goal.geometry import CliqueSet

# The three model shapes this design has to cover.

HMOG = CliqueSet.chain(3)
"""x --- y --- k: hierarchical mixture of Gaussians, levels (1, 1, 1)."""

MFA = CliqueSet(
    n_nodes=3,
    n_observable=1,
    cliques=((0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)),
)
"""Mixture of factor analyzers: the three-clique makes x --- k an edge, levels (1, 2)."""

CCA = CliqueSet(
    n_nodes=3,
    n_observable=2,
    cliques=((0,), (1,), (2,), (0, 2), (1, 2)),
)
"""Canonical correlation analysis: two observable nodes, one latent, levels (2, 1)."""


class TestLevels:
    """Levels are derived from the cliques, never declared."""

    @pytest.mark.parametrize(
        ("clique_set", "expected"),
        [
            (CliqueSet.chain(2), ((0,), (1,))),
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
        for clique_set in (CliqueSet.chain(4), HMOG, MFA, CCA):
            node_levels = clique_set.node_levels
            for clique in clique_set.cliques:
                spanned = {node_levels[i] for i in clique}
                assert max(spanned) - min(spanned) <= 1

    def test_boundary(self) -> None:
        assert HMOG.boundary == (1,)
        assert MFA.boundary == (1, 2)
        assert CCA.boundary == (2,)

    def test_latent_nodes(self) -> None:
        assert HMOG.latent_nodes == (1, 2)
        assert CCA.latent_nodes == (2,)


class TestCanonicalOrder:
    """Level-0 cliques, then crossing cliques, then the tail --- recursively."""

    def test_pair_reproduces_harmonium_layout(self) -> None:
        assert CliqueSet.chain(2).canonical_cliques == ((0,), (0, 1), (1,))

    def test_chain_nests(self) -> None:
        assert HMOG.canonical_cliques == ((0,), (0, 1), (1,), (1, 2), (2,))

    def test_chain_tail_span_matches_tail_model(self) -> None:
        # The tail span of the full layout is byte-identical to the layout the model on
        # the tail would produce on its own. This is what lets pst_man be a concrete
        # model with no translation.
        tail_nodes = HMOG.tail_nodes
        tail_span = HMOG.canonical_cliques[
            len(HMOG.observable_cliques + HMOG.crossing_cliques) :
        ]
        relabelled = tuple(
            tuple(sorted(tail_nodes[i] for i in c))
            for c in HMOG.tail().canonical_cliques
        )
        assert tail_span == relabelled

    def test_mfa_order(self) -> None:
        assert MFA.canonical_cliques == ((0,), (0, 1), (0, 1, 2), (1,), (1, 2), (2,))

    def test_cca_order(self) -> None:
        assert CCA.canonical_cliques == ((0,), (1,), (0, 2), (1, 2), (2,))

    @pytest.mark.parametrize("clique_set", [CliqueSet.chain(2), HMOG, MFA, CCA])
    def test_canonical_order_is_a_permutation(self, clique_set: CliqueSet) -> None:
        assert sorted(clique_set.canonical_cliques) == sorted(clique_set.cliques)

    def test_clique_groups_partition(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            head = clique_set.observable_cliques + clique_set.crossing_cliques
            tail_nodes = clique_set.tail_nodes
            tail = tuple(
                tuple(sorted(tail_nodes[i] for i in c))
                for c in clique_set.tail().cliques
            )
            assert sorted(head + tail) == sorted(clique_set.cliques)


class TestTail:
    """Peeling a level reindexes from zero with level 1 becoming observable."""

    def test_chain_tail_is_a_shorter_chain(self) -> None:
        assert CliqueSet.chain(3).tail() == CliqueSet.chain(2)
        assert CliqueSet.chain(4).tail().tail() == CliqueSet.chain(2)

    def test_tail_nodes_map_back(self) -> None:
        assert HMOG.tail_nodes == (1, 2)
        assert CCA.tail_nodes == (2,)

    def test_tail_levels_shift_down(self) -> None:
        assert MFA.tail().levels == ((0, 1),)
        assert CCA.tail() == CliqueSet(1, 1, ((0,),))

    def test_tail_observables_are_the_boundary_level(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            assert clique_set.tail().n_observable == len(clique_set.levels[1])


class TestNormalization:
    """Equivalent descriptions compare and hash equal, and models stay jit-static."""

    def test_member_and_clique_order_do_not_matter(self) -> None:
        scrambled = CliqueSet(3, 1, ((2, 1), (0,), (1, 0), (2,), (1,)))
        assert scrambled == HMOG

    def test_hashable(self) -> None:
        assert hash(CliqueSet.chain(3)) == hash(HMOG)
        assert {HMOG: "hmog"}[CliqueSet.chain(3)] == "hmog"

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

    @pytest.mark.parametrize("n_observable", [0, 3])
    def test_bad_observable_count(self, n_observable: int) -> None:
        with pytest.raises(ValueError, match=r"n_observable must be in 1\.\.2"):
            CliqueSet(2, n_observable, ((0,), (1,), (0, 1)))

    def test_empty_model(self) -> None:
        with pytest.raises(ValueError, match="n_nodes must be at least 1"):
            CliqueSet(0, 1, ())
