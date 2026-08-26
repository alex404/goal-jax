"""Tests for geometry/algebra/clique.py.

Verifies levels as distance from the root set, the partition a cover induces against a
set of nodes, the cut indices a one-node split yields, canonical clique ordering, level ascent and reindexing,
normalization/hashing, and each validation failure. ``Cliques`` is pure
Python, so this file imports no JAX and needs no platform configuration.
"""

import pytest

from goal.geometry import Cliques
from goal.geometry.algebra.clique import crossing_rows, cut_indices, partition


def path(n_nodes: int) -> Cliques:
    """The path ``0 --- 1 --- ... --- (n_nodes - 1)`` rooted at node 0, depth ``n_nodes``.

    Test scaffolding for parametrizing over depth. Deliberately not a ``Cliques``
    classmethod: level structure does not determine a cover, so no factory keyed on
    hierarchy can exist, and paths are only one of the shapes this module has to serve.
    """
    singletons = tuple((i,) for i in range(n_nodes))
    links = tuple((i, i + 1) for i in range(n_nodes - 1))
    return Cliques(n_nodes, 1, singletons + links)


# The three model shapes this design has to cover.

HMOG = path(3)
"""x --- y --- k: hierarchical mixture of Gaussians, levels (1, 1, 1)."""

MFA = Cliques(
    n_nodes=3,
    n_roots=1,
    cliques=((0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)),
)
"""Mixture of factor analyzers: the three-clique makes x --- k an edge, levels (1, 2)."""

CCA = Cliques(
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
        assert HMOG.graph == ((1,), (0, 2), (1,))
        # The three-clique makes x and k adjacent even though no pair (0, 2) was declared.
        assert MFA.graph == ((1, 2), (0, 2), (0, 1))
        assert CCA.graph == ((2,), (2,), (0, 1))

    def test_edges_are_the_graph_as_pairs(self) -> None:
        for clique_set in (path(4), HMOG, MFA, CCA):
            pairs = {
                (min(i, j), max(i, j))
                for i, near in enumerate(clique_set.graph)
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


class TestPartition:
    """One primitive under both structural splits.

    ``partition`` sorts a cover into the cliques inside $S$, those crossing its boundary,
    and those outside. At the root set it is the split a layout is stored by; at a single
    node it is the re-rooting cut. These check that the two really are one operation, and
    that the split stays total where ``crossing_rows`` does not.
    """

    @pytest.mark.parametrize("name", ["hmog", "mfa", "cca"])
    def test_at_the_root_set_it_is_the_three_groups(self, name: str) -> None:
        clq = {"hmog": HMOG, "mfa": MFA, "cca": CCA}[name]
        inside, crossing, outside = clq.partition(clq.root_nodes)
        assert tuple(clq.cliques[i] for i in inside) == clq.root_cliques
        assert tuple(clq.cliques[i] for i in crossing) == clq.cross_cliques
        assert tuple(clq.cliques[i] for i in outside) == clq.deep_cliques

    @pytest.mark.parametrize("name", ["hmog", "mfa", "cca"])
    def test_the_groups_partition_the_cover(self, name: str) -> None:
        clq = {"hmog": HMOG, "mfa": MFA, "cca": CCA}[name]
        inside, crossing, outside = clq.partition(clq.root_nodes)
        assert sorted(inside + crossing + outside) == list(range(len(clq.cliques)))

    def test_cut_at_one_node_gives_the_mixture_view(self) -> None:
        """MFA split at its category node: the outside group is the base harmonium."""
        inside, crossing, outside = MFA.partition({2})
        assert tuple(MFA.cliques[i] for i in inside) == ((2,),)
        assert tuple(MFA.cliques[i] for i in crossing) == ((0, 1, 2), (1, 2))
        assert tuple(MFA.cliques[i] for i in outside) == ((0,), (0, 1), (1,))
        assert crossing_rows(MFA.cliques, {2}) == (1, 2)

    def test_rows_follow_the_outside_order(self) -> None:
        """Each crossing clique is the row above the clique it reduces to."""
        _, crossing, outside = MFA.partition({2})
        rows = crossing_rows(MFA.cliques, {2})
        for pos, row in zip(crossing, rows, strict=True):
            near_part = tuple(j for j in MFA.cliques[pos] if j != 2)
            assert MFA.cliques[outside[row]] == near_part

    def test_rows_can_collide_above_one_node(self) -> None:
        """At $|S| > 1$ two crossing cliques can share a row --- CCA's two roots do."""
        assert crossing_rows(CCA.cliques, CCA.root_nodes) == (0, 0)

    def test_a_bare_edge_splits_but_has_no_rows(self) -> None:
        """The split is total where the row assignment is not.

        Node 1 carries no block, so the crossing clique has nothing to be a row of ---
        yet the graph is legitimate and its crossing group is well defined.
        """
        edge = Cliques(n_nodes=2, n_roots=1, cliques=((0, 1),))
        assert edge.cross_cliques == ((0, 1),)
        with pytest.raises(ValueError, match="has no row block"):
            crossing_rows(edge.cliques, edge.root_nodes)

    def test_the_empty_and_full_sets_are_degenerate(self) -> None:
        every = range(MFA.n_nodes)
        assert MFA.partition(())[2] == tuple(range(len(MFA.cliques)))
        assert MFA.partition(every)[0] == tuple(range(len(MFA.cliques)))

    def test_a_stray_node_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"nodes \[7\] are not among"):
            MFA.partition({7})

    def test_positions_count_in_the_order_given(self) -> None:
        """The free function splits whatever order it is handed, not the graph's.

        A layout counts positions in *storage* order, so that the positions it computes
        and the dimensions it slices come from one list. The two orders really do differ:
        a two-root graph lays its cover out root-then-cross while the graph itself stores
        it sorted, so the same $S$ yields different positions on each.
        """
        reversed_cover = tuple(reversed(MFA.cliques))
        inside, crossing, outside = partition(reversed_cover, {2})
        assert tuple(reversed_cover[i] for i in inside) == ((2,),)
        assert tuple(reversed_cover[i] for i in crossing) == ((1, 2), (0, 1, 2))
        assert tuple(reversed_cover[i] for i in outside) == ((1,), (0, 1), (0,))
        assert crossing_rows(reversed_cover, {2}) == (0, 1)


class TestCliqueCut:
    """The re-rooting cut, as indices over a cover and its block sizes.

    ``cut_indices`` is :func:`partition` at a single node plus the two conditions a matrix
    view needs. The cover is passed in *storage* order, so these build it by hand rather
    than through ``Cliques`` --- which is the case the free function exists for.
    """

    # x = 0 (dim 4), y = 1 (dim 2), k = 2 (dim 2): MFA's layout.
    COVER: tuple[tuple[int, ...], ...] = ((0,), (0, 1), (0, 1, 2), (1,), (1, 2), (2,))
    DIMS: tuple[int, ...] = (4, 8, 16, 2, 4, 2)

    def test_the_mixture_view_of_mfa(self) -> None:
        cut = cut_indices(self.COVER, self.DIMS, far_node=2, n_nodes=3)
        assert cut.near_idx == (0, 1, 3)
        assert cut.cross_idx == (2, 4)
        assert cut.far_idx == (5,)
        assert cut.cross_rows == (1, 2)
        assert cut.n_cols == 2

    def test_the_near_side_is_the_base_harmonium(self) -> None:
        """Near blocks sum to the factor analyzer's own parameter vector."""
        cut = cut_indices(self.COVER, self.DIMS, far_node=2, n_nodes=3)
        assert sum(self.DIMS[i] for i in cut.near_idx) == 4 + 8 + 2

    def test_every_crossing_block_is_a_full_row_band(self) -> None:
        cut = cut_indices(self.COVER, self.DIMS, far_node=2, n_nodes=3)
        for pos, i in enumerate(cut.cross_idx):
            height = self.DIMS[cut.near_idx[cut.cross_rows[pos]]]
            assert self.DIMS[i] == height * cut.n_cols

    def test_rows_are_distinct_at_a_single_node(self) -> None:
        """What makes the row assignment total, and the matrix well formed."""
        cut = cut_indices(self.COVER, self.DIMS, far_node=2, n_nodes=3)
        assert len(set(cut.cross_rows)) == len(cut.cross_rows)

    def test_a_stray_far_node_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="far_node 7 is not one of the graph's 3"):
            cut_indices(self.COVER, self.DIMS, far_node=7, n_nodes=3)

    def test_cutting_the_only_node_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="leaves no near side"):
            cut_indices(((0,),), (3,), far_node=0, n_nodes=1)

    def test_a_far_node_without_a_block_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no block of its own"):
            cut_indices(((0,), (0, 1)), (4, 8), far_node=1, n_nodes=2)

    def test_a_crossing_clique_without_a_row_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no row block"):
            cut_indices(((0, 1), (1,)), (8, 2), far_node=1, n_nodes=2)

    def test_a_partial_coupling_is_rejected(self) -> None:
        """A linear Gaussian model has no cut at its latent node.

        Combinatorially admissible --- both ``(0,)`` and ``(1,)`` are in the cover --- but
        the interaction reaches only a subspace of node 1, so the crossing blocks do not
        share one column axis. This is the condition that keeps a cut from being a mere
        generalization of a level split.
        """
        with pytest.raises(ValueError, match="does not couple the whole far side"):
            cut_indices(((0,), (0, 1), (1,)), (4, 4, 2), far_node=1, n_nodes=2)


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
        deep_span = HMOG.canonical_cliques[
            len(HMOG.root_cliques + HMOG.cross_cliques) :
        ]
        relabelled = tuple(
            tuple(i + HMOG.n_roots for i in c)
            for c in HMOG.ascend_level().canonical_cliques
        )
        assert deep_span == relabelled

    def test_mfa_order(self) -> None:
        assert MFA.canonical_cliques == ((0,), (0, 1), (0, 1, 2), (1,), (1, 2), (2,))

    def test_cca_order(self) -> None:
        assert CCA.canonical_cliques == ((0,), (1,), (0, 2), (1, 2), (2,))

    @pytest.mark.parametrize("clique_set", [path(2), HMOG, MFA, CCA])
    def test_canonical_order_is_a_permutation(self, clique_set: Cliques) -> None:
        assert sorted(clique_set.canonical_cliques) == sorted(clique_set.cliques)

    def test_clique_groups_partition(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            head = clique_set.root_cliques + clique_set.cross_cliques
            deep = tuple(
                tuple(i + clique_set.n_roots for i in c)
                for c in clique_set.ascend_level().cliques
            )
            assert sorted(head + deep) == sorted(clique_set.cliques)


class TestTail:
    """Ascending a level reindexes from zero with the boundary becoming the new root set."""

    def test_chain_ascends_to_a_shorter_chain(self) -> None:
        assert path(3).ascend_level() == path(2)
        assert path(4).ascend_level().ascend_level() == path(2)

    def test_ascent_is_a_shift_not_a_permutation(self) -> None:
        """Node $i$ one level up is node $i + n_r$ here, because levels ascend with index.

        This is what lets ``LevelCliques`` splice its deep span in by renumbering rather
        than reordering, so the graph's clique order and the manifold's block order cannot
        come apart between levels.
        """
        for clique_set in (HMOG, MFA, CCA):
            above = clique_set.ascend_level()
            offset = clique_set.n_roots
            assert above.n_nodes == clique_set.n_nodes - offset
            assert (
                tuple(tuple(i + offset for i in c) for c in above.cliques)
                == clique_set.deep_cliques
            )

    def test_ascended_levels_shift_down(self) -> None:
        assert MFA.ascend_level().level_sets == ((0, 1),)
        assert CCA.ascend_level() == Cliques(1, 1, ((0,),))

    def test_ascended_roots_are_the_boundary_level(self) -> None:
        for clique_set in (HMOG, MFA, CCA):
            assert clique_set.ascend_level().n_roots == len(clique_set.level_sets[1])

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
        above = clique_set.ascend_level()
        node_levels = clique_set.node_levels
        for i in range(above.n_nodes):
            assert above.node_levels[i] == node_levels[i + clique_set.n_roots] - 1


class TestAtomicShapes:
    """The two irreducible covers: a lone node, and a lone edge.

    These are the base cases of a self-similar clique manifold. The edge is why a
    singleton clique cannot be required --- an interaction carries a coupling and no
    bias, so its cover is one clique with no singletons at all.
    """

    def test_node_atom(self) -> None:
        node = Cliques(n_nodes=1, n_roots=1, cliques=((0,),))
        assert node.level_sets == ((0,),)
        assert node.edges == ()
        assert node.canonical_cliques == ((0,),)

    def test_edge_atom(self) -> None:
        edge = Cliques(n_nodes=2, n_roots=1, cliques=((0, 1),))
        assert edge.level_sets == ((0,), (1,))
        assert edge.edges == ((0, 1),)
        assert edge.canonical_cliques == ((0, 1),)
        assert edge.root_cliques == ()
        assert edge.cross_cliques == ((0, 1),)
        assert edge.deep_cliques == ()

    def test_multi_clique_edge_atom(self) -> None:
        """A block map over three nodes: the cover MFA's interaction needs."""
        edge = Cliques(n_nodes=3, n_roots=1, cliques=((0, 1), (0, 1, 2), (0, 2)))
        assert edge.level_sets == ((0,), (1, 2))
        assert edge.cross_cliques == ((0, 1), (0, 1, 2), (0, 2))

    def test_reachability_still_holds_without_singletons(self) -> None:
        with pytest.raises(ValueError, match="node 2 is not reachable"):
            Cliques(3, 1, ((0, 1),))

    def test_singleton_is_optional_not_forbidden(self) -> None:
        """Dropping the requirement must not make a node-only cover invalid."""
        mixed = Cliques(3, 1, ((0,), (0, 1), (1, 2)))
        assert mixed.level_sets == ((0,), (1,), (2,))


class TestNormalization:
    """Equivalent descriptions compare and hash equal, and models stay jit-static."""

    def test_member_and_clique_order_do_not_matter(self) -> None:
        scrambled = Cliques(3, 1, ((2, 1), (0,), (1, 0), (2,), (1,)))
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
            Cliques(3, 1, ((0,), (1,), (2,), (0, 3)))

    def test_unreachable_node(self) -> None:
        with pytest.raises(ValueError, match="node 1 is not reachable"):
            Cliques(3, 1, ((0,), (1,), (2,), (1, 2)))

    def test_nodes_numbered_out_of_level_order(self) -> None:
        """The path $0 - 2 - 1$ rooted at $\\{0\\}$: node 1 is at level 2, node 2 at level 1.

        Levels then descend with the index, so the cliques one level up would need a
        *permutation* back into this graph's indices rather than a shift. No manifold
        laying out ``[root | cross | deep]`` can honour that, which is why the graph is
        rejected rather than the mismatch tolerated. Renumbering the two nodes fixes it.
        """
        with pytest.raises(ValueError, match="number the nodes by level"):
            Cliques(3, 1, ((0,), (1,), (2,), (0, 2), (1, 2)))
        assert Cliques(3, 1, ((0,), (1,), (2,), (0, 1), (1, 2))).node_levels == (
            0,
            1,
            2,
        )

    def test_duplicate_cliques(self) -> None:
        with pytest.raises(ValueError, match="duplicate cliques"):
            Cliques(2, 1, ((0,), (1,), (0, 1), (1, 0)))

    @pytest.mark.parametrize("n_roots", [0, 3])
    def test_bad_root_count(self, n_roots: int) -> None:
        with pytest.raises(ValueError, match=r"n_roots must be in 1\.\.2"):
            Cliques(2, n_roots, ((0,), (1,), (0, 1)))

    def test_empty_model(self) -> None:
        """``n_roots >= 1`` and ``n_roots <= n_nodes`` already forbid an empty node set."""
        with pytest.raises(ValueError, match=r"n_roots must be in 1\.\.0"):
            Cliques(0, 1, ())


class TestMemberValidation:
    """A clique names a non-empty set of distinct nodes.

    Both checks run before normalization: sorting and deduplication would otherwise
    destroy the evidence and report a clique the caller never wrote.
    """

    def test_empty_clique(self) -> None:
        with pytest.raises(ValueError, match="empty clique"):
            Cliques(2, 1, ((0,), (), (0, 1)))

    def test_repeated_member(self) -> None:
        with pytest.raises(ValueError, match=r"clique \(0, 0\) repeats a node"):
            Cliques(2, 1, ((0, 0), (0, 1)))
