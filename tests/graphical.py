"""Tests for geometry/exponential_family/graphical.py.

A graphical harmonium is a deep model with harmoniums attached to its cliques, and its
crossings and conjugation are read off the observable harmoniums. The tests pin both against the
hierarchical mixture of Gaussians, a linear Gaussian model attached to the observable node
of a mixture, whose conjugation parameters are the lower model's placed in the prior
mixture's observable block. They also test ``SubCliquesEmbedding`` and ``RootEmbedding`` on it.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from goal.geometry import (
    Diagonal,
    ObservableEmbedding,
    RootEmbedding,
    Scale,
    SubCliquesEmbedding,
)
from goal.models import (
    analytic_hmog,
    differentiable_hmog,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def _hmogs() -> list[Any]:
    return [
        analytic_hmog(obs_dim=5, obs_rep=Diagonal(), lat_dim=2, n_components=3),
        differentiable_hmog(
            obs_dim=5, obs_rep=Diagonal(), lat_dim=2, pst_rep=Diagonal(), n_components=3
        ),
    ]


class TestComposedLayout:
    """An observable harmonium on node $0$ of the deep model keeps its crossings unchanged."""

    @pytest.mark.parametrize("model", _hmogs())
    def test_crossings_are_the_root_harmoniums(self, model: Any) -> None:
        assert model.crs_clqs == model.lwr_hrm.crs_clqs
        assert model.crs_maps == model.lwr_hrm.crs_maps

    @pytest.mark.parametrize("model", _hmogs())
    def test_observable_is_the_root_harmoniums(self, model: Any) -> None:
        assert model.obs_man == model.lwr_hrm.obs_man

    @pytest.mark.parametrize("model", _hmogs())
    def test_likelihood_splits_into_the_root_harmoniums(self, model: Any) -> None:
        params = model.initialize(jax.random.PRNGKey(0), shape=0.5)
        lkl = model.likelihood_function(params)
        (att_lkl,) = model.obs_likelihoods(lkl)
        assert jnp.array_equal(att_lkl, lkl)

    @pytest.mark.parametrize("model", _hmogs())
    def test_root_posterior_is_the_observable_block(self, model: Any) -> None:
        lat = jax.random.normal(jax.random.PRNGKey(1), (model.pst_man.dim,))
        expected = ObservableEmbedding(model.pst_man).project(lat)
        assert jnp.array_equal(model.obs_pst_emb(0).project(lat), expected)


class TestConjugation:
    """The conjugation parameters are the observable harmonium's, placed on its clique of the prior."""

    @pytest.mark.parametrize("model", _hmogs())
    def test_matches_observable_placement(self, model: Any) -> None:
        params = model.initialize(jax.random.PRNGKey(2), shape=0.5)
        lkl = model.likelihood_function(params)
        expected = ObservableEmbedding(model.prr_upr_hrm).embed(
            model.lwr_hrm.conjugation_parameters(lkl)
        )
        assert jnp.allclose(model.conjugation_parameters(lkl), expected)

    @pytest.mark.parametrize("model", _hmogs())
    def test_offset_is_the_root_harmoniums(self, model: Any) -> None:
        params = model.initialize(jax.random.PRNGKey(3), shape=0.5)
        lkl = model.likelihood_function(params)
        assert jnp.allclose(
            model.conjugation_offset(lkl), model.lwr_hrm.conjugation_offset(lkl)
        )

    @pytest.mark.parametrize("model", _hmogs())
    def test_placement_is_linear(self, model: Any) -> None:
        key_a, key_b = jax.random.split(jax.random.PRNGKey(4))
        dim = model.lwr_hrm.prr_man.dim
        a = jax.random.normal(key_a, (dim,))
        b = jax.random.normal(key_b, (dim,))
        assert jnp.allclose(
            model.place_conjugation((a + b,)),
            model.place_conjugation((a,)) + model.place_conjugation((b,)),
        )


class TestSubCliquesEmbedding:
    """Places a smaller layout on some cliques of a larger one, block for block."""

    @staticmethod
    def _deep_in_composite() -> tuple[Any, SubCliquesEmbedding]:
        """The upper mixture on nodes $(y, k) = (1, 2)$ of the whole model: three cliques."""
        model = _hmogs()[1]
        return model, SubCliquesEmbedding((1, 2), model, model.pst_man)

    def test_embed_fills_the_deep_partition(self) -> None:
        model, emb = self._deep_in_composite()
        v = jax.random.normal(jax.random.PRNGKey(7), (model.pst_man.dim,))
        expected = model.join_coords(model.obs_man.zeros(), model.int_man.zeros(), v)
        assert jnp.array_equal(emb.embed(v), expected)

    def test_project_inverts_embed(self) -> None:
        model, emb = self._deep_in_composite()
        v = jax.random.normal(jax.random.PRNGKey(8), (model.pst_man.dim,))
        assert jnp.array_equal(emb.project(emb.embed(v)), v)

    def test_project_is_the_transpose(self) -> None:
        model, emb = self._deep_in_composite()
        key_a, key_b = jax.random.split(jax.random.PRNGKey(9))
        a = jax.random.normal(key_a, (model.pst_man.dim,))
        b = jax.random.normal(key_b, (model.dim,))
        assert jnp.allclose(jnp.dot(emb.embed(a), b), jnp.dot(a, emb.project(b)))


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
        pst_root, _, _ = model.pst_upr_hrm.split_coords(v)
        prr_root, _, _ = model.prr_upr_hrm.split_coords(emb.embed(v))
        assert jnp.array_equal(prr_root, model.lwr_hrm.pst_prr_emb.embed(pst_root))

    def test_cross_and_deep_spans_pass_through(self) -> None:
        model, emb = self._asymmetric_pair()
        v = jax.random.normal(jax.random.PRNGKey(5), (model.pst_upr_hrm.dim,))
        _, sub_cross, sub_deep = model.pst_upr_hrm.split_coords(v)
        _, amb_cross, amb_deep = model.prr_upr_hrm.split_coords(emb.embed(v))
        assert jnp.array_equal(amb_cross, sub_cross)
        assert jnp.array_equal(amb_deep, sub_deep)

        w = jax.random.normal(jax.random.PRNGKey(6), (model.prr_upr_hrm.dim,))
        _, amb_cross, amb_deep = model.prr_upr_hrm.split_coords(w)
        _, sub_cross, sub_deep = model.pst_upr_hrm.split_coords(emb.project(w))
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

        A layout's graph is composed from its parts, so a mismatch has to come from parts
        that genuinely differ. Here the ambient is the whole three-node model rather than
        its upper harmonium, so its graph has a node the sub's does not.
        """
        model, _ = self._asymmetric_pair()
        pst = model.pst_upr_hrm
        assert pst.cliques != model.cliques
        with pytest.raises(ValueError, match="differ only in the root partition"):
            RootEmbedding(model.lwr_hrm.pst_prr_emb, model, pst)
