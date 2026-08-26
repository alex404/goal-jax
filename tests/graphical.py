"""Tests for the clique-indexed layouts in geometry/manifold/graphical.py.

A clique manifold stores coordinates as the three spans of one level ascent,
``[root | cross | deep]``. The tests pin that layout against what ``analytic_hmog`` and
``differentiable_hmog`` already produce, so any disagreement is a real difference and not
a change of convention. The decisive case is the last one: a hierarchical model's
posterior-to-prior embedding must transform the root span and leave the other two
untouched, which is what lets a difference deep in the graph be expressed by nesting.
"""

from dataclasses import dataclass
from math import prod
from typing import override

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueBlockEmbedding,
    CliqueSet,
    Diagonal,
    EmbeddedMap,
    IdentityEmbedding,
    InteractionEmbedding,
    LevelCliques,
    LinearClique,
    LinearCliques,
    Manifold,
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
    MixtureOfFactorAnalyzers,
    Normal,
    analytic_hmog,
    differentiable_hmog,
    factor_analysis,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

TWO_NODE = CliqueSet(n_nodes=2, n_roots=1, cliques=((0,), (1,), (0, 1)))
"""x --- z: the graph a plain harmonium lays out on."""

THREE_NODE = CliqueSet(n_nodes=3, n_roots=1, cliques=((0,), (1,), (2,), (0, 1), (1, 2)))
"""x --- y --- k: the graph a hierarchical mixture declares."""


@dataclass(frozen=True)
class _Spans(LevelCliques[Manifold, Manifold, Manifold]):
    """A clique manifold assembled from three explicit span manifolds."""

    _clq_set: CliqueSet
    _root_man: Manifold
    _cross_man: Manifold
    _deep_man: Manifold
    _cross_blocks: tuple[LinearClique, ...]

    @property
    @override
    def clq_set(self) -> CliqueSet:
        return self._clq_set

    @property
    @override
    def cross_blocks(self) -> tuple[LinearClique, ...]:
        return self._cross_blocks

    @property
    @override
    def root_man(self) -> Manifold:
        return self._root_man

    @property
    @override
    def cross_man(self) -> Manifold:
        return self._cross_man

    @property
    @override
    def deep_man(self) -> Manifold:
        return self._deep_man


def _hmog_spans():
    """The analytic HMoG alongside a bare clique manifold with the same three spans."""
    model = analytic_hmog(obs_dim=8, obs_rep=Diagonal(), lat_dim=3, n_components=4)
    spans = _Spans(
        THREE_NODE, model.obs_man, model.int_man, model.pst_man, model.cross_blocks
    )
    return model, spans


class TestSpanLayout:
    """The span layout reproduces the harmonium's byte layout."""

    def test_span_dims_match_the_model(self) -> None:
        model, spans = _hmog_spans()
        assert spans.root_man.dim == model.obs_man.dim
        assert spans.cross_man.dim == model.int_man.dim
        assert spans.deep_man.dim == model.pst_man.dim

    def test_total_dim_matches_the_model(self) -> None:
        model, spans = _hmog_spans()
        assert spans.dim == model.dim

    def test_deep_span_is_laid_out_as_the_upper_harmonium(self) -> None:
        # The deep span is byte-identical to what the mixture one level up produces,
        # which is what lets split_level be applied again to it.
        model, spans = _hmog_spans()
        coords = jnp.arange(float(spans.dim))
        deep = spans.split_level(coords)[2]
        assert deep.shape[0] == model.upr_hrm.dim
        y, yk, k = model.upr_hrm.split_level(deep)
        assert jnp.array_equal(jnp.concatenate([y, yk, k]), deep)
        assert y.shape[0] == model.upr_hrm.obs_man.dim

    def test_split_join_round_trip(self) -> None:
        _, spans = _hmog_spans()
        coords = jnp.arange(float(spans.dim))
        assert jnp.array_equal(spans.join_coords(*spans.split_coords(coords)), coords)

    def test_split_level_is_split_coords(self) -> None:
        _, spans = _hmog_spans()
        coords = jnp.arange(float(spans.dim))
        for a, b in zip(
            spans.split_level(coords), spans.split_coords(coords), strict=True
        ):
            assert jnp.array_equal(a, b)

    def test_join_rejects_wrong_arity(self) -> None:
        _, spans = _hmog_spans()
        parts = spans.split_coords(jnp.arange(float(spans.dim)))
        with pytest.raises(ValueError, match="expected 3 spans, got 2"):
            spans.join_coords(*parts[:2])


class TestHarmoniumSpans:
    """A harmonium's three spans are its observable, interaction, and latent sides."""

    def test_hmog_declares_the_three_node_chain(self) -> None:
        model, _ = _hmog_spans()
        assert model.clq_set == THREE_NODE
        assert model.clq_set.level_sets == ((0,), (1,), (2,))

    def test_spans_are_obs_int_pst(self) -> None:
        """The three spans are the observable, the interaction, and the posterior.

        The cross span *is* the interaction --- no wrapper. Which nodes its blocks couple
        is reported separately, by ``cross_blocks``, because that is the part only the model
        knows.
        """
        model, _ = _hmog_spans()
        assert model.root_man == model.obs_man
        assert model.cross_man == model.int_man
        assert model.deep_man == model.pst_man
        (clique,) = model.cross_blocks
        assert clique.members == (0, 1)
        assert clique.dim == model.int_man.dim

    @pytest.mark.parametrize(
        ("emb_cls", "idx"),
        [(ObservableEmbedding, 0), (InteractionEmbedding, 1), (PosteriorEmbedding, 2)],
    )
    def test_slot_embedding_isolates_its_span(self, emb_cls, idx: int) -> None:
        model, _ = _hmog_spans()
        emb = emb_cls(model)
        v = jnp.arange(1.0, emb.sub_man.dim + 1)
        assert jnp.array_equal(emb.project(emb.embed(v)), v)
        spans = model.split_level(emb.embed(v))
        for other in range(3):
            if other != idx:
                assert jnp.all(spans[other] == 0.0)


class TestRootEmbedding:
    """Transforms the root span; the cross and deep spans pass through untouched."""

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

    def test_rejects_mismatched_clique_sets(self) -> None:
        model, _ = self._asymmetric_pair()
        pst, prr = model.pst_upr_hrm, model.prr_upr_hrm
        odd = _Spans(
            CliqueSet(2, 2, ((0,), (1,))),
            prr.obs_man,
            prr.int_man,
            prr.pst_man,
            prr.cross_blocks,
        )
        with pytest.raises(ValueError, match="must share a clique set"):
            RootEmbedding(model.lwr_hrm.pst_prr_emb, pst, odd)


def test_layout_is_jit_static() -> None:
    """Clique manifolds must hash, since models are passed as static jit arguments."""
    _, spans = _hmog_spans()

    @jax.jit
    def total(coords: Array, man: _Spans = spans) -> Array:
        return jnp.sum(man.split_level(coords)[0])

    coords = jnp.arange(float(spans.dim))
    assert jnp.allclose(total(coords), jnp.sum(coords[: spans.root_man.dim]))


class TestCliqueAddressing:
    """Per-clique blocks, and re-viewing the layout across an arbitrary node cut."""

    def test_clique_dims_sum_to_dim(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        assert sum(model.clique_dims) == model.dim

    def test_one_block_per_clique(self) -> None:
        """The declared graph and the parameter layout must agree block for block."""
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        assert len(model.clique_dims) == len(model.clq_set.canonical_cliques)

    def test_split_join_cliques_round_trip(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        params = jax.random.normal(jax.random.PRNGKey(0), (model.dim,))
        assert jnp.allclose(model.join_cliques(*model.split_cliques(params)), params)

    def test_split_cliques_matches_declared_dims(self) -> None:
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        blocks = model.split_cliques(jnp.zeros(model.dim))
        assert tuple(b.size for b in blocks) == model.clique_dims

    def test_cut_isolating_the_deepest_node(self) -> None:
        """x-y-k re-viewed as (x, y) | k: the crossing clique gets one row block."""
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
        cross span is free to couple a sub-statistic, a single matrix view is not.
        """
        model = analytic_hmog(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, n_components=4)
        with pytest.raises(ValueError, match="does not couple the whole far side"):
            model.cut(1)

    def test_cut_rejects_a_far_node_with_no_block(self) -> None:
        """Singleton cliques are optional, so the far node may carry no block at all.

        Without this guard the cut has zero columns and every crossing block is read as
        a row band of width nothing --- silently, since no arithmetic contradicts it.
        """
        blocks = (LinearClique((0,), (2,)), LinearClique((0, 1), (2, 3)))
        layout = _Layout(CliqueSet(2, 1, ((0,), (0, 1))), blocks)
        with pytest.raises(ValueError, match="has no block of its own"):
            layout.cut(1)


### Layout Invariants ###


def layout_problems(man: LinearCliques) -> list[str]:
    """Every way a clique manifold's graph and its parameter layout can disagree.

    Four invariants, in dependency order. **Storage order is the order the graph induces**
    comes first, and returns on its own: every later check pairs a block with the clique
    sitting at its position, so under a mismatch they would blame the wrong clique or pass
    by coincidence. Then blocks tile the coordinate vector, each block has one axis per
    member, and its axes multiply out to its size.

    Nothing at runtime requires the first invariant --- :meth:`clique_offsets` and
    :meth:`clique_index` both read the blocks' own members. It is a property of every model
    the library ships, enforced here rather than at construction.
    """
    canonical = man.clq_set.canonical_cliques
    members = man.clique_members
    if members != canonical:
        return [f"storage order {members} is not canonical order {canonical}"]
    dims = man.clique_dims
    axes = man.clique_axes
    out: list[str] = []
    if sum(dims) != man.dim:
        out.append(f"blocks sum to {sum(dims)}, but dim is {man.dim}")
    for clique, block_dim, block_axes in zip(members, dims, axes, strict=True):
        if len(block_axes) != len(clique):
            out.append(f"clique {clique} has {len(block_axes)} axes {block_axes}")
        if prod(block_axes) != block_dim:
            out.append(f"clique {clique}: axes {block_axes} do not make {block_dim}")
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

    def test_the_arity_three_clique_has_three_axes(self) -> None:
        """MFA's $(x,y,k)$ block is a three-way interaction, so it has three factors.

        The domain sub-statistic is the joint $(y,k)$ block, which contributes one axis per
        node rather than one for the pair --- which is what makes the axis count the arity.
        """
        mfa = _mfa()
        axes = mfa.clique_axes[mfa.clique_index((0, 1, 2))]
        assert len(axes) == 3
        assert axes == (4, 2, 2)

    def test_blocks_tile_the_coordinate_vector(self) -> None:
        for name, man in shipped_models():
            coords = jnp.arange(float(man.dim))
            blocks = man.split_cliques(coords)
            assert jnp.array_equal(man.join_cliques(*blocks), coords), name


### Ordering Regressions ###


@dataclass(frozen=True)
class _ReversedCCA(
    CanonicalCorrelationAnalysis[PositiveDefinite, PositiveDefinite, PositiveDefinite]
):
    """CCA declaring its branches in non-canonical order.

    Nothing forbids this: the two branches are symmetric, and a model author has no reason
    to know that ``CliqueSet`` will sort them.
    """

    @property
    @override
    def int_members(self) -> tuple[tuple[int, ...], ...]:
        return ((1, 2), (0, 2))


@dataclass(frozen=True)
class _DerivedSpans(LevelCliques[Manifold, Manifold, Manifold]):
    """Three explicit spans whose graph is *derived* rather than declared."""

    _root_man: Manifold
    _cross_man: Manifold
    _deep_man: Manifold
    _cross_blocks: tuple[LinearClique, ...]

    @property
    @override
    def cross_blocks(self) -> tuple[LinearClique, ...]:
        return self._cross_blocks

    @property
    @override
    def root_man(self) -> Manifold:
        return self._root_man

    @property
    @override
    def cross_man(self) -> Manifold:
        return self._cross_man

    @property
    @override
    def deep_man(self) -> Manifold:
        return self._deep_man


def _misrooted() -> _DerivedSpans:
    """A level whose cross clique reaches past the deep span's own root.

    The deep span is a mixture rooted at $y$, but the crossing clique couples the
    observable to $k$. The glued graph therefore reroots at $k$, while the deep span still
    lays itself out $y$-first --- so $k$ lands at level 1 and $y$ at level 2, and levels
    descend with the node index. This is the shape the numbering invariant now rejects.
    """
    obs = Normal(3, Diagonal())
    mix = CompleteMixture(Normal(2, Diagonal()), 4)
    cross = EmbeddedMap(
        Rectangular(),
        CliqueBlockEmbedding(mix, (1,), (IdentityEmbedding(Categorical(4)),)),
        IdentityEmbedding(obs),
    )
    block = LinearClique((0, 2), (obs.dim, Categorical(4).dim))
    return _DerivedSpans(obs, cross, mix, (block,))


class TestDeclarationOrderRegressions:
    """A clique's layout slot must hold that clique's parameters."""

    def test_reversed_branches_address_their_own_block(self) -> None:
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
        # int_members[1] == (0, 2), so the second block is the (0, 2) clique.
        found = model.split_cliques(params)[model.clique_index((0, 2))]
        assert jnp.array_equal(found, declared[1])

    def test_misrooted_deep_span_is_rejected_by_the_graph(self) -> None:
        """The numbering invariant catches this at the graph rather than the layout.

        Node 2 is at level 1 and node 1 at level 2, so the cliques one level up would need
        a permutation back into these indices rather than a shift by ``n_roots`` --- which
        is exactly what ``LevelCliques`` cannot do, since it splices its deep span in by
        renumbering. The layout stays self-consistent either way (below), but the graph the
        spans imply can no longer be built at all.
        """
        with pytest.raises(ValueError, match="number the nodes by level"):
            _ = _misrooted().clq_set

    def test_misrooted_deep_span_labels_its_own_blocks(self) -> None:
        """Independent of the graph: ``clique_index`` reads the blocks' own members."""
        man = _misrooted()
        params = jnp.arange(float(man.dim))
        deep = man.split_level(params)[2]
        y_bias, _, k_bias = man.deep_man.split_level(deep)  # pyright: ignore[reportAttributeAccessIssue]
        blocks = man.split_cliques(params)
        assert jnp.array_equal(blocks[man.clique_index((1,))], y_bias)
        assert jnp.array_equal(blocks[man.clique_index((2,))], k_bias)


### Guards ###


@dataclass(frozen=True)
class _Layout(LinearCliques):
    """A clique manifold given directly as a graph plus one block record per clique.

    The graph is passed separately so a test can hand over a *lying* one --- a graph its
    blocks do not cover --- which is what the layout-disagreement guard exists to catch.
    """

    _clq_set: CliqueSet
    _clique_blocks: tuple[LinearClique, ...]

    @property
    @override
    def dim(self) -> int:
        return sum(block.dim for block in self._clique_blocks)

    @property
    @override
    def clq_set(self) -> CliqueSet:
        return self._clq_set

    @property
    @override
    def clique_blocks(self) -> tuple[LinearClique, ...]:
        return self._clique_blocks


def _forked() -> _Layout:
    """A fork: two crossing cliques, ``(0,1)`` and ``(0,2)``, sharing the near part ``(0,)``.

    The shape that collided when ``cut`` admitted an arbitrary set of far nodes --- cutting
    across ``{1, 2}`` gave both cliques the same row band. Cutting one node at a time, they
    land on different sides and the collision cannot arise.
    """
    blocks = (
        LinearClique((0,), (2,)),
        LinearClique((0, 1), (2, 4)),
        LinearClique((0, 2), (2, 4)),
        LinearClique((1,), (4,)),
        LinearClique((2,), (4,)),
    )
    return _Layout(CliqueSet(3, 1, tuple(b.members for b in blocks)), blocks)


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
        with pytest.raises(ValueError, match="far_node 7 is not one of the graph's 3"):
            _forked().cut(7)

    def test_rejects_cutting_the_only_node(self) -> None:
        blocks = (LinearClique((0,), (4,)),)
        man = _Layout(CliqueSet(1, 1, ((0,),)), blocks)
        with pytest.raises(ValueError, match="leaves no near side"):
            man.cut(0)


class TestSplitJoinGuards:
    """Wrongly sized coordinates are an error, not a silent truncation."""

    def test_split_rejects_the_wrong_total(self) -> None:
        man = _forked()
        with pytest.raises(ValueError, match="expected a flat array of 26"):
            man.split_cliques(jnp.arange(float(man.dim + 3)))

    def test_join_rejects_a_wrongly_sized_block(self) -> None:
        man = _forked()
        blocks = list(man.split_cliques(jnp.arange(float(man.dim))))
        blocks[1] = blocks[1][:-1]
        with pytest.raises(ValueError, match=r"clique block 1 has shape"):
            man.join_cliques(*blocks)

    def test_join_rejects_the_wrong_block_count(self) -> None:
        man = _forked()
        blocks = man.split_cliques(jnp.arange(float(man.dim)))
        with pytest.raises(ValueError, match="expected 5 clique blocks, got 4"):
            man.join_cliques(*blocks[:-1])
