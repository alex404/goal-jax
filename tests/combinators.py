"""Tests for ``CliqueManifold`` in geometry/manifold/combinators.py and ``RootEmbedding``
in geometry/manifold/embedding.py.

A clique manifold stores coordinates as the three spans of one level ascent,
``[root | cross | deep]``. The tests pin that layout against what ``analytic_hmog`` and
``differentiable_hmog`` already produce, so any disagreement is a real difference and not
a change of convention. The decisive case is the last one: a hierarchical model's
posterior-to-prior embedding must transform the root span and leave the other two
untouched, which is what lets a difference deep in the graph be expressed by nesting.
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueManifold,
    CliqueSet,
    Diagonal,
    InteractionEmbedding,
    Manifold,
    ObservableEmbedding,
    PosteriorEmbedding,
    RootEmbedding,
    Scale,
)
from goal.models import analytic_hmog, differentiable_hmog

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

TWO_NODE = CliqueSet(n_nodes=2, n_roots=1, cliques=((0,), (1,), (0, 1)))
"""x --- z: the graph a plain harmonium lays out on."""

THREE_NODE = CliqueSet(n_nodes=3, n_roots=1, cliques=((0,), (1,), (2,), (0, 1), (1, 2)))
"""x --- y --- k: the graph a hierarchical mixture declares."""


@dataclass(frozen=True)
class _Spans(CliqueManifold[Manifold, Manifold, Manifold]):
    """A clique manifold assembled from three explicit span manifolds."""

    _clq_set: CliqueSet
    _root_man: Manifold
    _cross_man: Manifold
    _deep_man: Manifold

    @property
    @override
    def clq_set(self) -> CliqueSet:
        return self._clq_set

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
    spans = _Spans(THREE_NODE, model.obs_man, model.int_man, model.pst_man)
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
        assert model.clq_set.levels == ((0,), (1,), (2,))

    def test_spans_are_obs_int_pst(self) -> None:
        model, _ = _hmog_spans()
        assert model.root_man == model.obs_man
        assert model.cross_man == model.int_man
        assert model.deep_man == model.pst_man

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
            CliqueSet(2, 2, ((0,), (1,))), prr.obs_man, prr.int_man, prr.pst_man
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
