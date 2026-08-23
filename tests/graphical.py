"""Tests for the clique-indexed layout: ``CliqueManifold`` in
geometry/manifold/combinators.py, and ``CliqueEmbedding`` / ``CliqueSetEmbedding`` in
geometry/manifold/embedding.py.

The layout and the slot embeddings are compared to what ``analytic_hmog`` and
``differentiable_hmog`` already produce through the nested-harmonium scheme, so any
disagreement is a real difference and not a change of convention. The decisive case is
the last one: ``CliqueSetEmbedding`` must reproduce ``LatentHarmoniumEmbedding``
elementwise on the asymmetric posterior/prior pair.
"""

from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueEmbedding,
    CliqueManifold,
    CliqueSet,
    CliqueSetEmbedding,
    Diagonal,
    Manifold,
    Scale,
)
from goal.models import analytic_hmog, differentiable_hmog

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class _Blocks(CliqueManifold):
    """A clique manifold assembled from one explicit block per canonical clique."""

    _clq_set: CliqueSet
    mans: tuple[Manifold, ...]

    @property
    @override
    def clq_set(self) -> CliqueSet:
        return self._clq_set

    @override
    def clq_man(self, clique: tuple[int, ...]) -> Manifold:
        return self.mans[self.block_index(clique)]


def _hmog_layout():
    """The analytic HMoG as x --- y --- k, with its five blocks in canonical order."""
    model = analytic_hmog(obs_dim=8, obs_rep=Diagonal(), lat_dim=3, n_components=4)
    blocks = _Blocks(
        CliqueSet.chain(3),
        (
            model.obs_man,
            model.lwr_hrm.int_man,
            model.lwr_hrm.pst_man,
            model.upr_hrm.int_man,
            model.upr_hrm.pst_man,
        ),
    )
    return model, blocks


class TestLayout:
    """The clique layout reproduces the nested harmonium's byte layout."""

    def test_block_dims_match_the_model(self) -> None:
        model, blocks = _hmog_layout()
        x_dim, xy_dim, y_dim, yk_dim, k_dim = (m.dim for m in blocks.clq_mans)
        assert x_dim == model.obs_man.dim
        assert xy_dim == model.int_man.dim
        assert y_dim + yk_dim + k_dim == model.pst_man.dim

    def test_total_dim_matches_the_model(self) -> None:
        model, blocks = _hmog_layout()
        assert blocks.dim == model.dim

    def test_tail_span_is_the_upper_harmonium(self) -> None:
        # The last three blocks are exactly the mixture's own [obs | int | lat].
        model, blocks = _hmog_layout()
        coords = jnp.arange(float(blocks.dim))
        tail = jnp.concatenate(blocks.split_coords(coords)[2:])
        assert jnp.array_equal(tail, model.split_coords(coords)[2])

    def test_split_join_round_trip(self) -> None:
        _, blocks = _hmog_layout()
        coords = jnp.arange(float(blocks.dim))
        assert jnp.array_equal(blocks.join_coords(*blocks.split_coords(coords)), coords)

    def test_join_rejects_wrong_arity(self) -> None:
        _, blocks = _hmog_layout()
        parts = blocks.split_coords(jnp.arange(float(blocks.dim)))
        with pytest.raises(ValueError, match="expected 5 clique blocks, got 4"):
            blocks.join_coords(*parts[:4])

    def test_block_index_rejects_non_cliques(self) -> None:
        _, blocks = _hmog_layout()
        assert blocks.block_index((1, 0)) == 1  # member order does not matter
        with pytest.raises(ValueError, match=r"\(0, 2\) is not a clique"):
            blocks.block_index((0, 2))


class TestSlotEmbeddings:
    """Slot embeddings isolate one clique's block and zero the rest."""

    @pytest.mark.parametrize("node", [0, 1, 2])
    def test_singleton_clique_round_trips(self, node: int) -> None:
        _, blocks = _hmog_layout()
        emb = CliqueEmbedding(blocks, (node,))
        v = jnp.arange(1.0, emb.sub_man.dim + 1)
        assert jnp.array_equal(emb.project(emb.embed(v)), v)

    def test_embedding_zeros_other_blocks(self) -> None:
        _, blocks = _hmog_layout()
        emb = CliqueEmbedding(blocks, (1,))
        embedded = emb.embed(jnp.ones(emb.sub_man.dim))
        parts = blocks.split_coords(embedded)
        assert jnp.all(parts[2] == 1.0)
        for i in (0, 1, 3, 4):
            assert jnp.all(parts[i] == 0.0)

    def test_embedding_selects_the_interaction(self) -> None:
        model, blocks = _hmog_layout()
        emb = CliqueEmbedding(blocks, (0, 1))
        assert emb.sub_man.dim == model.lwr_hrm.int_man.dim
        v = jnp.arange(1.0, emb.sub_man.dim + 1)
        assert jnp.array_equal(emb.project(emb.embed(v)), v)

    def test_translate_is_additive(self) -> None:
        _, blocks = _hmog_layout()
        emb = CliqueEmbedding(blocks, (2,))
        base = jnp.arange(1.0, blocks.dim + 1)
        shifted = emb.translate(base, jnp.ones(emb.sub_man.dim))
        assert jnp.array_equal(emb.project(shifted), emb.project(base) + 1.0)


class TestCliqueSetEmbedding:
    """The generalization must agree with LatentHarmoniumEmbedding, elementwise."""

    @staticmethod
    def _asymmetric_pair():
        model = differentiable_hmog(
            obs_dim=6, obs_rep=Diagonal(), lat_dim=2, pst_rep=Scale(), n_components=3
        )
        pst, prr = model.pst_upr_hrm, model.prr_upr_hrm
        chain = CliqueSet.chain(2)
        pst_blocks = _Blocks(chain, (pst.obs_man, pst.int_man, pst.pst_man))
        prr_blocks = _Blocks(chain, (prr.obs_man, prr.int_man, prr.pst_man))
        generalized = CliqueSetEmbedding(
            pst_blocks, prr_blocks, (model.lwr_hrm.pst_prr_emb, None, None)
        )
        return model, generalized

    def test_dims_line_up(self) -> None:
        model, generalized = self._asymmetric_pair()
        assert generalized.sub_man.dim == model.pst_upr_hrm.dim
        assert generalized.amb_man.dim == model.prr_upr_hrm.dim

    def test_embed_matches_latent_harmonium_embedding(self) -> None:
        model, generalized = self._asymmetric_pair()
        v = jax.random.normal(jax.random.PRNGKey(2), (model.pst_upr_hrm.dim,))
        assert jnp.array_equal(generalized.embed(v), model.pst_prr_emb.embed(v))

    def test_project_matches_latent_harmonium_embedding(self) -> None:
        model, generalized = self._asymmetric_pair()
        w = jax.random.normal(jax.random.PRNGKey(3), (model.prr_upr_hrm.dim,))
        assert jnp.array_equal(generalized.project(w), model.pst_prr_emb.project(w))

    def test_translate_matches_latent_harmonium_embedding(self) -> None:
        model, generalized = self._asymmetric_pair()
        key_p, key_q = jax.random.split(jax.random.PRNGKey(4))
        p = jax.random.normal(key_p, (model.prr_upr_hrm.dim,))
        q = jax.random.normal(key_q, (model.pst_upr_hrm.dim,))
        assert jnp.array_equal(
            generalized.translate(p, q), model.pst_prr_emb.translate(p, q)
        )

    def test_blocks_without_an_embedding_pass_through(self) -> None:
        # embed takes natural parameters and project takes mean parameters, so
        # composing them is not a meaningful operation to assert on. What this
        # embedding does claim is that the blocks marked None are untouched in either
        # direction --- here the interaction and the categorical prior.
        model, generalized = self._asymmetric_pair()
        v = jax.random.normal(jax.random.PRNGKey(5), (model.pst_upr_hrm.dim,))
        sub_blocks = generalized.sub_man.split_coords(v)
        amb_blocks = generalized.amb_man.split_coords(generalized.embed(v))
        assert jnp.array_equal(amb_blocks[1], sub_blocks[1])
        assert jnp.array_equal(amb_blocks[2], sub_blocks[2])

        w = jax.random.normal(jax.random.PRNGKey(6), (model.prr_upr_hrm.dim,))
        amb_blocks = generalized.amb_man.split_coords(w)
        sub_blocks = generalized.sub_man.split_coords(generalized.project(w))
        assert jnp.array_equal(sub_blocks[1], amb_blocks[1])
        assert jnp.array_equal(sub_blocks[2], amb_blocks[2])

    def test_rejects_mismatched_clique_sets(self) -> None:
        model, _ = self._asymmetric_pair()
        pst, prr = model.pst_upr_hrm, model.prr_upr_hrm
        pst_blocks = _Blocks(
            CliqueSet.chain(2), (pst.obs_man, pst.int_man, pst.pst_man)
        )
        prr_blocks = _Blocks(CliqueSet(2, 2, ((0,), (1,))), (prr.obs_man, prr.pst_man))
        with pytest.raises(ValueError, match="must share a clique set"):
            CliqueSetEmbedding(pst_blocks, prr_blocks, (None, None, None))

    def test_wrong_number_of_block_embeddings_is_not_silent(self) -> None:
        model, _ = self._asymmetric_pair()
        pst, prr = model.pst_upr_hrm, model.prr_upr_hrm
        chain = CliqueSet.chain(2)
        pst_blocks = _Blocks(chain, (pst.obs_man, pst.int_man, pst.pst_man))
        prr_blocks = _Blocks(chain, (prr.obs_man, prr.int_man, prr.pst_man))
        emb = CliqueSetEmbedding(pst_blocks, prr_blocks, (None, None))
        with pytest.raises(ValueError, match="argument 2 is longer"):
            emb.embed(jnp.zeros(pst_blocks.dim))


def test_layout_is_jit_static() -> None:
    """Clique manifolds must hash, since models are passed as static jit arguments."""
    _, blocks = _hmog_layout()

    @jax.jit
    def total(coords: Array, man: _Blocks = blocks) -> Array:
        return jnp.sum(man.split_coords(coords)[0])

    coords = jnp.arange(float(blocks.dim))
    assert jnp.allclose(total(coords), jnp.sum(coords[: blocks.clq_mans[0].dim]))
