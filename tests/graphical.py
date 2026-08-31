"""Tests for ``LinearClique`` and the clique-indexed layouts in geometry/manifold/clique.py.

A clique manifold stores coordinates as the three partitions of one level ascent,
``[root | cross | deep]``. The tests pin that layout against what ``analytic_hmog`` and
``differentiable_hmog`` already produce, so any disagreement is a real difference and not
a change of convention. The decisive layout case is the embedding one: a hierarchical
model's posterior-to-prior embedding must transform the root partition and leave the other two
untouched, which is what lets a difference deep in the graph be expressed by nesting.

The last four classes test a clique's *form algebra* rather than its placement. The
decisive ones are at arity 2: they pin the contraction against the ``MatrixMap``
machinery every interaction in the library already runs on, so the arity-$n$
generalization is verified against working code rather than against a fresh derivation.
The arity-3 tests then check the two properties that make higher arity usable --- that
contraction order does not matter, and that partial contraction composes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import override

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueCut,
    Diagonal,
    ExponentialFamily,
    IdentityEmbedding,
    Interaction,
    InteractionEmbedding,
    LevelCliques,
    LinearClique,
    LinearCliques,
    Manifold,
    MatrixMap,
    ObservableEmbedding,
    PositiveDefinite,
    PosteriorEmbedding,
    Rectangular,
    RootEmbedding,
    Scale,
)
from goal.models import (
    CanonicalCorrelationAnalysis,
    Categorical,
    CompleteMixture,
    Euclidean,
    MixtureOfFactorAnalyzers,
    Normal,
    Poissons,
    analytic_hmog,
    differentiable_hmog,
    factor_analysis,
)


def _form(axes: tuple[int, ...]) -> LinearClique:
    """A form with the given axis sizes, built over ``Euclidean`` nodes.

    A clique is only its embeddings now, so every arity is one construction --- the layout
    that a coupling of three or more nodes has to reach through is a :class:`Interaction`'s
    business, not the clique's.
    """
    return LinearClique(
        Rectangular(), tuple(IdentityEmbedding(Euclidean(d)) for d in axes)
    )


def _place(
    members: tuple[int, ...], axes: tuple[int, ...]
) -> tuple[tuple[int, ...], LinearClique]:
    """A clique with the given axis sizes at the given nodes, for layouts built by hand."""
    return (members, _form(axes))


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class _Partitions(LevelCliques[ExponentialFamily, Manifold, ExponentialFamily]):
    """A clique manifold assembled from three explicit partition manifolds."""

    _root_man: ExponentialFamily
    _cross_man: Manifold
    _deep_man: ExponentialFamily
    _cross_placements: tuple[tuple[tuple[int, ...], LinearClique], ...]

    @property
    @override
    def cross_placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        return self._cross_placements

    @property
    @override
    def root_man(self) -> ExponentialFamily:
        return self._root_man

    @property
    @override
    def cross_man(self) -> Manifold:
        return self._cross_man

    @property
    @override
    def deep_man(self) -> ExponentialFamily:
        return self._deep_man


def _hmog_partitions():
    """The analytic HMoG alongside a bare clique manifold with the same three partitions."""
    model = analytic_hmog(obs_dim=8, obs_rep=Diagonal(), lat_dim=3, n_components=4)
    partitions = _Partitions(
        model.obs_man, model.int_man, model.pst_man, model.cross_placements
    )
    return model, partitions


class TestPartitionLayout:
    """The partition layout reproduces the harmonium's byte layout."""

    def test_span_dims_match_the_model(self) -> None:
        model, partitions = _hmog_partitions()
        assert partitions.root_man.dim == model.obs_man.dim
        assert partitions.cross_man.dim == model.int_man.dim
        assert partitions.deep_man.dim == model.pst_man.dim

    def test_total_dim_matches_the_model(self) -> None:
        model, partitions = _hmog_partitions()
        assert partitions.dim == model.dim

    def test_deep_span_is_laid_out_as_the_upper_harmonium(self) -> None:
        # The deep partition is byte-identical to what the mixture one level up produces,
        # which is what lets split_level be applied again to it.
        model, partitions = _hmog_partitions()
        coords = jnp.arange(float(partitions.dim))
        deep = partitions.split_level(coords)[2]
        assert deep.shape[0] == model.upr_hrm.dim
        y, yk, k = model.upr_hrm.split_level(deep)
        assert jnp.array_equal(jnp.concatenate([y, yk, k]), deep)
        assert y.shape[0] == model.upr_hrm.obs_man.dim

    def test_split_join_round_trip(self) -> None:
        _, partitions = _hmog_partitions()
        coords = jnp.arange(float(partitions.dim))
        assert jnp.array_equal(
            partitions.join_coords(*partitions.split_coords(coords)), coords
        )

    def test_split_level_is_split_coords(self) -> None:
        _, partitions = _hmog_partitions()
        coords = jnp.arange(float(partitions.dim))
        for a, b in zip(
            partitions.split_level(coords), partitions.split_coords(coords), strict=True
        ):
            assert jnp.array_equal(a, b)

    def test_join_rejects_wrong_arity(self) -> None:
        _, partitions = _hmog_partitions()
        parts = partitions.split_coords(jnp.arange(float(partitions.dim)))
        with pytest.raises(ValueError, match="expected 3 partitions, got 2"):
            partitions.join_coords(*parts[:2])


class TestHarmoniumSpans:
    """A harmonium's three partitions are its observable, interaction, and latent sides."""

    def test_hmog_declares_the_three_node_chain(self) -> None:
        model, _ = _hmog_partitions()
        assert model.root_nodes == frozenset({0})
        assert model.cliques == ((0,), (0, 1), (1,), (1, 2), (2,))
        assert model.level_sets == ((0,), (1,), (2,))

    def test_spans_are_obs_int_pst(self) -> None:
        """The three partitions are the observable, the interaction, and the posterior.

        The cross partition *is* the interaction --- no wrapper. Which nodes its pieces couple
        is reported separately, by ``cross_placements``, because that is the part only the model
        knows.
        """
        model, _ = _hmog_partitions()
        assert model.root_man == model.obs_man
        assert model.cross_man == model.int_man
        assert model.deep_man == model.pst_man
        ((members, form),) = model.cross_placements
        assert members == (0, 1)
        assert form.dim == model.int_man.dim

    @pytest.mark.parametrize(
        ("emb_cls", "idx"),
        [(ObservableEmbedding, 0), (InteractionEmbedding, 1), (PosteriorEmbedding, 2)],
    )
    def test_slot_embedding_isolates_its_span(self, emb_cls, idx: int) -> None:
        model, _ = _hmog_partitions()
        emb = emb_cls(model)
        v = jnp.arange(1.0, emb.sub_man.dim + 1)
        assert jnp.array_equal(emb.project(emb.embed(v)), v)
        partitions = model.split_level(emb.embed(v))
        for other in range(3):
            if other != idx:
                assert jnp.all(partitions[other] == 0.0)


class TestRootEmbedding:
    """Transforms the root partition; the cross and deep partitions pass through untouched."""

    @staticmethod
    def _asymmetric_pair():
        model = differentiable_hmog(
            obs_dim=6, obs_rep=Diagonal(), lat_dim=2, pst_rep=Scale(), n_components=3
        )
        return model, model.pst_prr_emb

    def test_dims_line_up(self) -> None:
        model, emb = self._asymmetric_pair()
        assert emb.sub_man.dim == model.pst_upr_hrm.dim
        assert emb.amb_man.dim == model.prr_upr_hrm.dim

    def test_root_span_is_the_lower_embedding(self) -> None:
        model, emb = self._asymmetric_pair()
        v = jax.random.normal(jax.random.PRNGKey(2), (model.pst_upr_hrm.dim,))
        pst_root, _, _ = model.pst_upr_hrm.split_level(v)
        prr_root, _, _ = model.prr_upr_hrm.split_level(emb.embed(v))
        assert jnp.array_equal(prr_root, model.lwr_hrm.pst_prr_emb.embed(pst_root))

    def test_cross_and_deep_spans_pass_through(self) -> None:
        model, emb = self._asymmetric_pair()
        v = jax.random.normal(jax.random.PRNGKey(5), (model.pst_upr_hrm.dim,))
        _, sub_cross, sub_deep = model.pst_upr_hrm.split_level(v)
        _, amb_cross, amb_deep = model.prr_upr_hrm.split_level(emb.embed(v))
        assert jnp.array_equal(amb_cross, sub_cross)
        assert jnp.array_equal(amb_deep, sub_deep)

        w = jax.random.normal(jax.random.PRNGKey(6), (model.prr_upr_hrm.dim,))
        _, amb_cross, amb_deep = model.prr_upr_hrm.split_level(w)
        _, sub_cross, sub_deep = model.pst_upr_hrm.split_level(emb.project(w))
        assert jnp.array_equal(sub_cross, amb_cross)
        assert jnp.array_equal(sub_deep, amb_deep)

    def test_translate_is_additive_on_the_root_span(self) -> None:
        model, emb = self._asymmetric_pair()
        key_p, key_q = jax.random.split(jax.random.PRNGKey(4))
        p = jax.random.normal(key_p, (model.prr_upr_hrm.dim,))
        q = jax.random.normal(key_q, (model.pst_upr_hrm.dim,))
        assert jnp.allclose(emb.translate(p, q), p + emb.embed(q))

    def test_rejects_mismatched_graphs(self) -> None:
        """The two sides must present the same cover, not merely the same dimensions.

        Since the merge, a layout's graph is read off its forms, so a mismatch cannot be
        faked by declaring one --- it has to come from partitions that genuinely cover
        differently. Here the ambient is the whole three-node model rather than its upper
        harmonium, so its cover has a node the sub's does not.
        """
        model, _ = self._asymmetric_pair()
        pst = model.pst_upr_hrm
        assert not pst.same_graph(model)
        with pytest.raises(ValueError, match="must share a clique set"):
            RootEmbedding(model.lwr_hrm.pst_prr_emb, pst, model)


def test_layout_is_jit_static() -> None:
    """Clique manifolds must hash, since models are passed as static jit arguments."""
    _, partitions = _hmog_partitions()

    @jax.jit
    def total(coords: Array, man: _Partitions = partitions) -> Array:
        return jnp.sum(man.split_level(coords)[0])

    coords = jnp.arange(float(partitions.dim))
    assert jnp.allclose(total(coords), jnp.sum(coords[: partitions.root_man.dim]))


class TestCliqueAddressing:
    """Per-clique coordinates, and re-viewing the layout across an arbitrary node cut."""

    def test_clique_dims_sum_to_dim(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        assert sum(model.clique_dims) == model.dim

    def test_one_form_per_clique(self) -> None:
        """The declared graph and the parameter layout must agree clique for clique."""
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        assert len(model.clique_dims) == len(model.canonical_cliques)

    def test_split_join_cliques_round_trip(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        params = jax.random.normal(jax.random.PRNGKey(0), (model.dim,))
        assert jnp.allclose(model.join_cliques(*model.split_cliques(params)), params)

    def test_split_cliques_matches_declared_dims(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        parts = model.split_cliques(jnp.zeros(model.dim))
        assert tuple(p.size for p in parts) == model.clique_dims

    def test_cut_isolating_the_deepest_node(self) -> None:
        """x-y-k re-viewed as (x, y) | k: the crossing clique gets one row band."""
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        cut = model.cut(2)
        assert cut.n_cols == model.clique_dims[-1]
        assert len(cut.cross_idx) == 1

    def test_cut_round_trips(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        params = jax.random.normal(jax.random.PRNGKey(2), (model.dim,))
        cut = model.cut(2)
        assert jnp.allclose(cut.join(*cut.project(params)), params)

    def test_cut_rejects_a_sub_statistic_coupling(self) -> None:
        """A Gaussian interaction couples part of the latent statistic, so has no view.

        This is why ``cut`` is not a generalization of ``split_level``: the level split's
        cross partition is free to couple a sub-statistic, a single matrix view is not.
        """
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        with pytest.raises(ValueError, match="does not couple the whole far side"):
            model.cut(1)

    def test_cut_rejects_a_far_node_with_no_clique(self) -> None:
        """Singleton cliques are optional, so the far node may carry no clique at all.

        Without this guard the cut has zero columns and every crossing clique is read as
        a row band of width nothing --- silently, since no arithmetic contradicts it.
        """
        forms = (_place((0,), (2,)), _place((0, 1), (2, 3)))
        layout = _Layout(frozenset({0}), forms)
        with pytest.raises(ValueError, match="has no clique of its own"):
            layout.cut(1)


class TestCliqueCutIndices:
    """The cut's index bookkeeping, over a cover and its clique sizes directly.

    Relocated from ``clique.py`` when ``CliqueCut`` moved to the manifold layer: the class
    reads dimensions and its operations are array work, so it cannot live in the JAX-free
    combinatorics module. The cover is given in *storage* order and built by hand, which is
    what these check the class accepts.
    """

    # x = 0 (dim 4), y = 1 (dim 2), k = 2 (dim 2): MFA's layout.
    COVER: tuple[tuple[int, ...], ...] = ((0,), (0, 1), (0, 1, 2), (1,), (1, 2), (2,))
    DIMS: tuple[int, ...] = (4, 8, 16, 2, 4, 2)

    def test_the_mixture_view_of_mfa(self) -> None:
        cut = CliqueCut(self.COVER, self.DIMS, 2)
        assert cut.near_idx == (0, 1, 3)
        assert cut.cross_idx == (2, 4)
        assert cut.far_idx == (5,)
        assert cut.cross_rows == (1, 2)
        assert cut.n_cols == 2

    def test_the_near_side_is_the_base_harmonium(self) -> None:
        """The near side sums to the factor analyzer's own parameter vector."""
        cut = CliqueCut(self.COVER, self.DIMS, 2)
        assert sum(self.DIMS[i] for i in cut.near_idx) == 4 + 8 + 2

    def test_every_crossing_clique_is_a_full_row_band(self) -> None:
        cut = CliqueCut(self.COVER, self.DIMS, 2)
        for pos, i in enumerate(cut.cross_idx):
            height = self.DIMS[cut.near_idx[cut.cross_rows[pos]]]
            assert self.DIMS[i] == height * cut.n_cols

    def test_rows_are_distinct_at_a_single_node(self) -> None:
        """What makes the row assignment total, and the matrix well formed."""
        cut = CliqueCut(self.COVER, self.DIMS, 2)
        assert len(set(cut.cross_rows)) == len(cut.cross_rows)

    def test_a_stray_far_node_is_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="far_node 7 is not one of the graph's nodes"
        ):
            CliqueCut(self.COVER, self.DIMS, 7)

    def test_cutting_the_only_node_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="leaves no near side"):
            CliqueCut(((0,),), (3,), 0)

    def test_a_far_node_without_a_clique_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no clique of its own"):
            CliqueCut(((0,), (0, 1)), (4, 8), 1)

    def test_a_crossing_clique_without_a_row_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no row band"):
            CliqueCut(((0, 1), (1,)), (8, 2), 1)

    def test_a_partial_coupling_is_rejected(self) -> None:
        """A linear Gaussian model has no cut at its latent node.

        Combinatorially admissible --- both ``(0,)`` and ``(1,)`` are in the cover --- but
        the interaction reaches only a subspace of node 1, so the crossing cliques do not
        share one column axis. This is the condition that keeps a cut from being a mere
        generalization of a level split.
        """
        with pytest.raises(ValueError, match="does not couple the whole far side"):
            CliqueCut(((0,), (0, 1), (1,)), (4, 4, 2), 1)


### Layout Invariants ###


def layout_problems(man: LinearCliques) -> list[str]:
    """Every way a clique manifold's graph and its parameter layout can disagree.

    Five invariants, in dependency order. **Storage order is the order the graph induces**
    comes first, and returns on its own: every later check pairs a form with the clique
    sitting at its position, so under a mismatch they would blame the wrong clique or pass
    by coincidence. Then the forms tile the coordinate vector, each form has one axis per
    node, its axes multiply out to its size, and finally every clique touching a node agrees
    about what occupies that node.

    That last one is what ``node_mans`` derives, and it is the invariant a clique's
    embeddings exist to make checkable: an axis embedding goes from a sub-space into a
    *node*, never into whatever larger manifold a caller happens to hold, so two cliques
    meeting at a node describe the same thing or one of them is coupling something that is
    not there.

    Nothing at runtime requires any of this --- :meth:`clique_offsets` and
    :meth:`clique_index` both read the layout's own cliques. They are properties of every
    model the library ships, enforced here rather than at construction.
    """
    canonical = man.canonical_cliques
    members = man.cliques
    if members != canonical:
        return [f"storage order {members} is not canonical order {canonical}"]
    dims = man.clique_dims
    axes = man.clique_axes
    out: list[str] = []
    if sum(dims) != man.dim:
        out.append(f"forms sum to {sum(dims)}, but dim is {man.dim}")
    for clique, size, form_axes in zip(members, dims, axes, strict=True):
        if len(form_axes) != len(clique):
            out.append(f"clique {clique} has {len(form_axes)} axes {form_axes}")
        if prod(form_axes) != size:
            out.append(f"clique {clique}: axes {form_axes} do not make {size}")
    try:
        node_mans = man.node_mans
    except ValueError as disagreement:
        out.append(str(disagreement))
    else:
        if len(node_mans) != len(man.nodes):
            out.append(f"{len(node_mans)} node manifolds for {len(man.nodes)} nodes")
    return out


def shipped_models() -> list[tuple[str, LinearCliques]]:
    """One instance of every model shape the library ships a graph for."""
    return [
        ("factor_analysis", factor_analysis(obs_dim=4, lat_dim=2)),
        (
            "hmog",
            analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4),
        ),
        ("cca", _cca()),
        ("mfa", _mfa()),
    ]


def _cca() -> CanonicalCorrelationAnalysis[
    PositiveDefinite, PositiveDefinite, PositiveDefinite
]:
    """Asymmetric branches, so a swapped layout is visible in the dimensions."""
    return CanonicalCorrelationAnalysis(
        fst_dim=3,
        fst_rep=PositiveDefinite(),
        snd_dim=2,
        snd_rep=PositiveDefinite(),
        lat_dim=2,
        pst_rep=PositiveDefinite(),
    )


def _mfa() -> MixtureOfFactorAnalyzers:
    return MixtureOfFactorAnalyzers(
        n_categories=3, bas_hrm=factor_analysis(obs_dim=4, lat_dim=2)
    )


class TestLayoutInvariants:
    """Every shipped model's graph agrees with its parameter layout."""

    @pytest.mark.parametrize("name", [n for n, _ in shipped_models()])
    def test_layout_agrees_with_graph(self, name: str) -> None:
        assert layout_problems(dict(shipped_models())[name]) == []

    @pytest.mark.parametrize("name", [n for n, _ in shipped_models()])
    def test_the_level_split_agrees_with_the_span_dimensions(self, name: str) -> None:
        """The two readings of where a level ends must coincide.

        ``split_coords`` slices at the summed sizes of the root and cross *forms*, while
        the partitions report their own dimensions. Both descriptions exist; the layout is only
        coherent if they agree, and only the form reading is what the split actually uses.
        """
        man = dict(shipped_models())[name]
        coords = jnp.arange(float(man.dim))
        root, cross, deep = man.split_level(coords)  # pyright: ignore[reportAttributeAccessIssue]
        partitions = (man.root_man.dim, man.cross_man.dim, man.deep_man.dim)  # pyright: ignore[reportAttributeAccessIssue]
        assert (root.size, cross.size, deep.size) == partitions
        assert jnp.array_equal(man.join_level(root, cross, deep), coords)  # pyright: ignore[reportAttributeAccessIssue]

    def test_the_arity_three_clique_has_three_axes(self) -> None:
        """MFA's $(x,y,k)$ clique is a three-way interaction, so it has three axes.

        The domain sub-statistic is the joint $(y,k)$ form, which contributes one axis per
        node rather than one for the pair --- which is what makes the axis count the arity.
        """
        mfa = _mfa()
        axes = mfa.clique_axes[mfa.clique_index((0, 1, 2))]
        assert len(axes) == 3
        assert axes == (4, 2, 2)

    def test_forms_tile_the_coordinate_vector(self) -> None:
        for name, man in shipped_models():
            coords = jnp.arange(float(man.dim))
            parts = man.split_cliques(coords)
            assert jnp.array_equal(man.join_cliques(*parts), coords), name


### Ordering Regressions ###


@dataclass(frozen=True)
class _ReversedCCA(
    CanonicalCorrelationAnalysis[PositiveDefinite, PositiveDefinite, PositiveDefinite]
):
    """CCA declaring its branches in non-canonical order.

    Nothing forbids this: the two branches are symmetric, and a model author has no reason
    to know that ``Cliques`` will sort them.
    """

    @property
    @override
    def cross_placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        return (
            ((1, 2), self._branch_clique(0)),
            ((0, 2), self._branch_clique(1)),
        )


@dataclass(frozen=True)
class _DerivedPartitions(LevelCliques[ExponentialFamily, Manifold, ExponentialFamily]):
    """Three explicit partitions whose graph is *derived* rather than declared."""

    _root_man: ExponentialFamily
    _cross_man: Manifold
    _deep_man: ExponentialFamily
    _cross_placements: tuple[tuple[tuple[int, ...], LinearClique], ...]

    @property
    @override
    def cross_placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        return self._cross_placements

    @property
    @override
    def root_man(self) -> ExponentialFamily:
        return self._root_man

    @property
    @override
    def cross_man(self) -> Manifold:
        return self._cross_man

    @property
    @override
    def deep_man(self) -> ExponentialFamily:
        return self._deep_man


def _misrooted() -> _DerivedPartitions:
    """A level whose cross clique reaches past the deep partition's own root.

    The deep partition is a mixture rooted at $y$, but the crossing clique couples the
    observable to $k$. The glued graph therefore reroots at $k$, while the deep partition still
    lays itself out $y$-first --- so $k$ lands at level 1 and $y$ at level 2, and levels
    descend with the node index. This is the shape the numbering invariant now rejects.
    """
    obs = Normal(3, Diagonal())
    mix = CompleteMixture(Normal(2, Diagonal()), 4)
    clique = LinearClique(
        Rectangular(), (IdentityEmbedding(obs), IdentityEmbedding(Categorical(4)))
    )
    cross = Interaction(obs, mix, (((0, 2), clique),), ((None, mix.clique_emb((1,))),))
    return _DerivedPartitions(obs, cross, mix, (((0, 2), clique),))


class TestDeclarationOrderRegressions:
    """A clique's layout slot must hold that clique's parameters."""

    def test_reversed_branches_address_their_own_clique(self) -> None:
        model = _ReversedCCA(
            fst_dim=3,
            fst_rep=PositiveDefinite(),
            snd_dim=2,
            snd_rep=PositiveDefinite(),
            lat_dim=2,
            pst_rep=PositiveDefinite(),
        )
        params = jnp.arange(float(model.dim))
        declared = model.int_man.coord_blocks(model.split_level(params)[1])
        # cross_placements[1] is on (0, 2), so the second piece is the (0, 2) clique.
        found = model.split_cliques(params)[model.clique_index((0, 2))]
        assert jnp.array_equal(found, declared[1])

    def test_misrooted_deep_span_is_still_a_graph(self) -> None:
        """Labels need not ascend with level, so this is a graph like any other.

        Node 2 is at level 1 and node 1 at level 2. Nothing renumbers between levels ---
        ``LevelCliques`` relabels its deep partition past the root nodes and the graph reads
        membership rather than comparing indices --- so the level structure comes out
        as the connectivity dictates rather than as the labels suggest.
        """
        man = _misrooted()
        assert man.node_levels == {0: 0, 2: 1, 1: 2}
        assert man.level_sets == ((0,), (2,), (1,))

    def test_misrooted_deep_span_labels_its_own_cliques(self) -> None:
        """Independent of the clique_set: ``clique_index`` reads the layout's own cliques."""
        man = _misrooted()
        params = jnp.arange(float(man.dim))
        deep = man.split_level(params)[2]
        y_bias, _, k_bias = man.deep_man.split_level(deep)  # pyright: ignore[reportAttributeAccessIssue]
        parts = man.split_cliques(params)
        assert jnp.array_equal(parts[man.clique_index((1,))], y_bias)
        assert jnp.array_equal(parts[man.clique_index((2,))], k_bias)


### Guards ###


@dataclass(frozen=True)
class _Layout(LinearCliques):
    """A clique manifold given directly as a root set plus one form per clique.

    The cover and the dimension are both *read off* the forms, so a test can no longer
    hand over a graph its forms do not cover --- there is only one description to
    disagree with.
    """

    _root_nodes: frozenset[int]
    _placements: tuple[tuple[tuple[int, ...], LinearClique], ...]

    @property
    @override
    def root_nodes(self) -> frozenset[int]:
        return self._root_nodes

    @property
    @override
    def placements(self) -> tuple[tuple[tuple[int, ...], LinearClique], ...]:
        return self._placements


def _forked() -> _Layout:
    """A fork: two crossing cliques, ``(0,1)`` and ``(0,2)``, sharing the near part ``(0,)``.

    The shape that collided when ``cut`` admitted an arbitrary set of far nodes --- cutting
    across ``{1, 2}`` gave both cliques the same row band. Cutting one node at a time, they
    land on different sides and the collision cannot arise.
    """
    forms = (
        _place((0,), (2,)),
        _place((0, 1), (2, 4)),
        _place((0, 2), (2, 4)),
        _place((1,), (4,)),
        _place((2,), (4,)),
    )
    return _Layout(frozenset({0}), forms)


class TestCutGuards:
    """``cut`` rejects the inputs for which no single matrix view exists."""

    def test_two_crossing_cliques_cannot_share_a_row(self) -> None:
        """Splitting off one node makes the row assignment total, not merely checked.

        Two crossing cliques sharing a near part $P$ would both be $P \\cup \\{far\\}$ and
        so be the same clique. The forked graph is the shape that used to collide when an
        arbitrary set of far nodes was allowed; cutting either branch node is now fine.
        """
        man = _forked()
        for far in (1, 2):
            cut = man.cut(far)
            assert len(set(cut.cross_rows)) == len(cut.cross_rows)
            coords = jnp.arange(float(man.dim))
            assert jnp.array_equal(cut.join(*cut.project(coords)), coords)

    def test_rejects_a_node_outside_the_graph(self) -> None:
        with pytest.raises(
            ValueError, match="far_node 7 is not one of the graph's nodes"
        ):
            _forked().cut(7)

    def test_rejects_cutting_the_only_node(self) -> None:
        forms = (_place((0,), (4,)),)
        man = _Layout(frozenset({0}), forms)
        with pytest.raises(ValueError, match="leaves no near side"):
            man.cut(0)


class TestPlacementRules:
    """The rules pairing a form with nodes has to satisfy, checked where the two meet.

    A form knows how many axes it has and a layout knows which nodes it couples;
    ``_validate_placement`` is the only place the two are put together, so it is the only
    place they can disagree. Node labels are otherwise free --- non-contiguous, not
    level-ordered, not zero-based --- so the only surviving rules are the ones that would
    make a clique mean two things at once: one axis per node, and one spelling per node
    set.
    """

    def test_a_placement_must_have_one_node_per_axis(self) -> None:
        man = _Layout(frozenset({0}), (_place((0, 1), (3,)),))
        with pytest.raises(ValueError, match=r"clique \(0, 1\) names 2 nodes"):
            _ = man.cliques

    @pytest.mark.parametrize("members", [(1, 1), (1, 0)])
    def test_members_must_be_distinct_and_ascending(
        self, members: tuple[int, ...]
    ) -> None:
        """Member order is axis order, so a node set has exactly one spelling."""
        man = _Layout(frozenset({members[0]}), (_place(members, (2, 3)),))
        with pytest.raises(ValueError, match="must name distinct nodes, ascending"):
            _ = man.cliques

    def test_clique_emb_refuses_nodes_no_form_holds_jointly(self) -> None:
        """The structural condition a coupling to a group of nodes needs."""
        man = CompleteMixture(Poissons(2), 3)
        with pytest.raises(ValueError, match=r"no clique on \(0, 5\)"):
            man.clique_emb((0, 5)).project(man.zeros())

    def test_labels_need_not_be_contiguous(self) -> None:
        """A layout over labels 10 and 40 is a layout like any other."""
        forms = (_place((10,), (2,)), _place((10, 40), (2, 3)), _place((40,), (3,)))
        man = _Layout(frozenset({10}), forms)
        assert man.nodes == (10, 40)
        assert man.node_levels == {10: 0, 40: 1}
        assert man.dim == 2 + 6 + 3
        assert man.clique_index((10, 40)) == 1
        assert man.canonical_cliques == ((10,), (10, 40), (40,))
        cut = man.cut(40)
        coords = jnp.arange(float(man.dim))
        assert jnp.array_equal(cut.join(*cut.project(coords)), coords)


class TestSplitJoinGuards:
    """Wrongly sized coordinates are an error, not a silent truncation."""

    def test_split_rejects_the_wrong_total(self) -> None:
        man = _forked()
        with pytest.raises(ValueError, match="expected a flat array of 26"):
            man.split_cliques(jnp.arange(float(man.dim + 3)))

    def test_join_rejects_a_wrongly_sized_part(self) -> None:
        man = _forked()
        parts = list(man.split_cliques(jnp.arange(float(man.dim))))
        parts[1] = parts[1][:-1]
        with pytest.raises(ValueError, match=r"clique 1 has shape"):
            man.join_cliques(*parts)

    def test_join_rejects_the_wrong_clique_count(self) -> None:
        man = _forked()
        parts = man.split_cliques(jnp.arange(float(man.dim)))
        with pytest.raises(ValueError, match="expected 5 cliques, got 4"):
            man.join_cliques(*parts[:-1])


### Form algebra ###


def _key(seed: int) -> Array:
    return jax.random.PRNGKey(seed)


def _axes(axes: tuple[int, ...]) -> LinearClique:
    """A form with the given axis sizes, for the form algebra."""
    return _form(axes)


class TestArityTwoMatchesMatrixMap:
    """At arity 2 a clique's form must reproduce the existing matrix machinery."""

    @staticmethod
    def _pair(cod_dim: int, dom_dim: int):
        emb_map = MatrixMap(Rectangular(), Euclidean(dom_dim), Euclidean(cod_dim))
        return emb_map, _axes((cod_dim, dom_dim))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6), (6, 1)])
    def test_dim_matches(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, form = self._pair(cod_dim, dom_dim)
        assert form.dim == emb_map.dim

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6)])
    def test_tensor_matches_outer_product(self, cod_dim: int, dom_dim: int) -> None:
        emb_map, form = self._pair(cod_dim, dom_dim)
        w = jax.random.normal(_key(0), (cod_dim,))
        v = jax.random.normal(_key(1), (dom_dim,))
        assert jnp.array_equal(form.tensor(w, v), emb_map.outer_product(w, v))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (6, 1)])
    def test_contract_domain_matches_application(
        self, cod_dim: int, dom_dim: int
    ) -> None:
        """``keep=0`` is matrix-vector multiplication."""
        emb_map, form = self._pair(cod_dim, dom_dim)
        params = jax.random.normal(_key(2), (form.dim,))
        v = jax.random.normal(_key(3), (dom_dim,))
        assert jnp.allclose(form.contract(params, 0, v), emb_map(params, v))

    @pytest.mark.parametrize(("cod_dim", "dom_dim"), [(3, 4), (5, 5), (1, 6)])
    def test_contract_codomain_matches_transpose(
        self, cod_dim: int, dom_dim: int
    ) -> None:
        """``keep=1`` is the transposed application."""
        emb_map, form = self._pair(cod_dim, dom_dim)
        params = jax.random.normal(_key(4), (form.dim,))
        w = jax.random.normal(_key(5), (cod_dim,))
        assert jnp.allclose(
            form.contract(params, 1, w), emb_map.transpose_apply(params, w)
        )

    def test_storage_order_is_row_major_over_codomain_then_domain(self) -> None:
        """The convention every ``int_man`` in the library already stores in."""
        emb_map, form = self._pair(2, 3)
        params = jnp.arange(6.0)
        assert jnp.array_equal(form.to_tensor(params), emb_map.to_matrix(params))


class TestHigherArity:
    """Properties that make arity 3 usable, which arity 2 cannot distinguish."""

    form: LinearClique = _axes((2, 3, 4))

    def test_dim_is_the_product(self) -> None:
        assert self.form.dim == 24
        assert self.form.arity == 3

    def test_tensor_round_trips_through_contraction(self) -> None:
        """Contracting a rank-one tensor against its own axes rescales the third."""
        u = jax.random.normal(_key(6), (2,))
        v = jax.random.normal(_key(7), (3,))
        w = jax.random.normal(_key(8), (4,))
        params = self.form.tensor(u, v, w)
        assert jnp.allclose(self.form.contract(params, 0, v, w), u * (v @ v) * (w @ w))
        assert jnp.allclose(self.form.contract(params, 1, u, w), v * (u @ u) * (w @ w))
        assert jnp.allclose(self.form.contract(params, 2, u, v), w * (u @ u) * (v @ v))

    def test_partial_contraction_composes(self) -> None:
        """Contracting axes one at a time equals contracting them together.

        This is what lets a level split contract the root axes and leave the deep ones:
        the result must not depend on the order the root axes are taken in.
        """
        params = jax.random.normal(_key(9), (self.form.dim,))
        u = jax.random.normal(_key(10), (2,))
        v = jax.random.normal(_key(11), (3,))
        both = self.form.contract(params, 2, u, v)

        # axis 0 first, leaving a (3, 4) form; then axis 0 of that (the old axis 1)
        step = _axes((3, 4))
        after_u = jnp.tensordot(self.form.to_tensor(params), u, axes=([0], [0]))
        stepwise = step.contract(step.from_tensor(after_u), 1, v)
        assert jnp.allclose(both, stepwise)

    def test_to_from_tensor_round_trip(self) -> None:
        params = jax.random.normal(_key(12), (self.form.dim,))
        assert jnp.array_equal(
            self.form.from_tensor(self.form.to_tensor(params)), params
        )

    def test_wrong_axis_count_is_rejected(self) -> None:
        u = jnp.ones(2)
        with pytest.raises(ValueError, match="expected 3 axes, got 2"):
            self.form.tensor(u, jnp.ones(3))
        params = jnp.zeros(self.form.dim)
        with pytest.raises(ValueError, match="expected 2 axes, got 1"):
            self.form.contract(params, 0, jnp.ones(3))


class TestArityOne:
    """A bias is a multilinear map of one axis; nothing should special-case it."""

    def test_tensor_and_contract_are_identity(self) -> None:
        form = _axes((5,))
        v = jax.random.normal(_key(13), (5,))
        assert form.dim == 5
        assert jnp.array_equal(form.tensor(v), v)
        assert jnp.array_equal(form.contract(v, 0), v)


class TestContractGuards:
    """``keep`` names an axis, so it must be one."""

    @pytest.mark.parametrize("keep", [-1, 3])
    def test_rejects_an_out_of_range_axis(self, keep: int) -> None:
        # Without the bounds check these contract every axis and return a scalar, and
        # keep=-1 additionally resolves to the last axis by Python indexing.
        man = _axes((2, 3, 4))
        params = jnp.arange(float(man.dim))
        vectors = [jnp.ones(2), jnp.ones(3), jnp.ones(4)]
        with pytest.raises(ValueError, match=r"keep must be in 0\.\.2"):
            man.contract(params, keep, *vectors)
