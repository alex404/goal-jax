"""Tests for geometry/exponential_family/graphical.py.

A graphical harmonium is a deep model with harmoniums attached to its cliques, and its
crossings and conjugation are read off the attachments. The tests pin both against the
hierarchical mixture of Gaussians, a linear Gaussian model attached to the observable node
of a mixture, whose conjugation parameters are the lower model's placed in the prior
mixture's observable block. They also test ``LatentHarmoniumEmbedding``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from goal.geometry import (
    Diagonal,
    LatentHarmoniumEmbedding,
    ObservableEmbedding,
    Scale,
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
    """An attachment on node $0$ of the deep model keeps its crossings unchanged."""

    @pytest.mark.parametrize("model", _hmogs())
    def test_crossings_are_the_attachments(self, model: Any) -> None:
        assert model.crs_cliques == model.lwr_hrm.crs_cliques
        assert model.crs_maps == model.lwr_hrm.crs_maps

    @pytest.mark.parametrize("model", _hmogs())
    def test_observable_is_the_attachments(self, model: Any) -> None:
        assert model.obs_man == model.lwr_hrm.obs_man

    @pytest.mark.parametrize("model", _hmogs())
    def test_likelihood_splits_into_the_attachments(self, model: Any) -> None:
        params = model.initialize(jax.random.PRNGKey(0), shape=0.5)
        lkl = model.likelihood_function(params)
        (att_lkl,) = model.attachment_likelihoods(lkl)
        assert jnp.array_equal(att_lkl, lkl)

    @pytest.mark.parametrize("model", _hmogs())
    def test_attachment_posterior_is_the_observable_block(self, model: Any) -> None:
        lat = jax.random.normal(jax.random.PRNGKey(1), (model.pst_man.dim,))
        expected = ObservableEmbedding(model.pst_man).project(lat)
        assert jnp.array_equal(model.attachment_posterior(0, lat), expected)


class TestConjugation:
    """The conjugation parameters are the attachment's, placed on its clique of the prior."""

    @pytest.mark.parametrize("model", _hmogs())
    def test_matches_observable_placement(self, model: Any) -> None:
        params = model.initialize(jax.random.PRNGKey(2), shape=0.5)
        lkl = model.likelihood_function(params)
        expected = ObservableEmbedding(model.prr_upr_hrm).embed(
            model.lwr_hrm.conjugation_parameters(lkl)
        )
        assert jnp.allclose(model.conjugation_parameters(lkl), expected)

    @pytest.mark.parametrize("model", _hmogs())
    def test_offset_is_the_attachments(self, model: Any) -> None:
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


class TestLatentHarmoniumEmbedding:
    """Transforms the observable; the interaction and posterior pass through untouched."""

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
        with pytest.raises(ValueError, match="differ only in their observable"):
            LatentHarmoniumEmbedding(model.lwr_hrm.pst_prr_emb, model, pst)
