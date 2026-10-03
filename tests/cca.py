"""Tests for models/harmonium/cca.py.

``CanonicalCorrelationAnalysis`` is the first model in the library over a graph with more
than one root, so these tests double as a check on the multi-root machinery: a
pair observable declared as two root nodes, a ``CrossMap`` cross partition holding one
clique per branch, and a conjugation that is the sum of the branches'.

The decisive test is :meth:`TestConjugation.test_conjugation_equation_holds` --- the
conjugation equation is exact for a fork exactly when the observable's log-partition
factorizes across branches, which is what makes the sum valid.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import (
    CliqueEmbedding,
    Diagonal,
    PositiveDefinite,
    Scale,
    TensorProduct,
)
from goal.models import CanonicalCorrelationAnalysis

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def cca(
    fst_dim: int = 3,
    snd_dim: int = 2,
    lat_dim: int = 2,
) -> CanonicalCorrelationAnalysis[PositiveDefinite, Diagonal, PositiveDefinite]:
    """A fork with differently-shaped observables, and a full-covariance posterior."""
    return CanonicalCorrelationAnalysis(
        fst_dim=fst_dim,
        fst_rep=PositiveDefinite(),
        snd_dim=snd_dim,
        snd_rep=Diagonal(),
        lat_dim=lat_dim,
        pst_rep=PositiveDefinite(),
    )


class TestGraph:
    """The fork: a two-node observable and one latent, and the layout matches it."""

    def test_clique_set(self) -> None:
        model = cca()
        assert model.n_nodes == 3
        assert model.crs_cliques == (((0,), (0,)), ((1,), (0,)))
        assert model.cliques == ((0,), (1,), (0, 2), (1, 2), (2,))

    def test_the_latent_follows_the_observables(self) -> None:
        """The pair's two nodes come first; the shared latent is numbered after them."""
        model = cca()
        assert model.obs_man.n_nodes == 2
        assert CliqueEmbedding((2,), model).sub_man.dim == model.pst_man.dim

    def test_one_form_per_clique(self) -> None:
        model = cca()
        assert len(model.clq_dims) == len(model.cliques)
        assert sum(model.clq_dims) == model.dim

    def test_observable_spans_two_nodes(self) -> None:
        """The observable is a flat container over the two root nodes, one bias each."""
        model = cca()
        obs = model.obs_man
        assert obs.cliques == ((0,), (1,))
        assert model.cliques[:2] == obs.cliques
        assert obs.clq_dims == (obs.fst_man.dim, obs.snd_man.dim)
        assert model.clq_dims[:2] == obs.clq_dims

    def test_interaction_holds_one_clique_per_branch(self) -> None:
        model = cca(fst_dim=3, snd_dim=2, lat_dim=2)
        assert model.crs_man.clq_dims == (3 * 2, 2 * 2)

    def test_branch_forms_contract_the_shared_latent(self) -> None:
        """What lets a single ``CrossMap`` hold both branches.

        Each branch's clique map contracts the same latent node and outputs into its own
        observable node; the interaction's two sides are the whole pair and the latent.
        """
        model = cca()
        fst, snd = (clq_map for _, _, clq_map in model.crs_man.terms)
        assert fst.dom_man == snd.dom_man
        assert fst.cod_man != snd.cod_man
        assert fst.cod_man == TensorProduct((model.obs_man.fst_man,))
        assert snd.cod_man == TensorProduct((model.obs_man.snd_man,))


class TestDimensions:
    """Data and parameter dimensions come from the three nodes."""

    def test_data_dim_is_the_three_nodes(self) -> None:
        model = cca(fst_dim=3, snd_dim=2, lat_dim=2)
        assert model.data_dim == 3 + 2 + 2

    @pytest.mark.parametrize(("fst_dim", "snd_dim"), [(3, 2), (2, 5), (1, 1)])
    def test_observable_dim_sums_the_branches(self, fst_dim: int, snd_dim: int) -> None:
        model = cca(fst_dim=fst_dim, snd_dim=snd_dim)
        assert (
            model.obs_man.dim == model.obs_man.fst_man.dim + model.obs_man.snd_man.dim
        )


class TestConjugation:
    """The conjugation equation, and the sum that solves it."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_conjugation_equation_holds(self, seed: int) -> None:
        """$\\psi(\\theta + \\Theta s(z)) = \\rho \\cdot s(z) + \\chi$ for all $z$.

        Exact for a fork because the observable pair's log-partition function is already
        the sum of its components', so each branch conjugates on its own.
        """
        model = cca()
        params = model.initialize(jax.random.PRNGKey(seed), shape=0.4)
        lkl_params = model.likelihood_function(params)
        rho = model.conjugation_parameters(lkl_params)
        chi = model.conjugation_offset(lkl_params)

        def residual(z: Array) -> Array:
            s_z = model.pst_man.sufficient_statistic(z)
            lhs = model.obs_man.log_partition_function(
                model.lkl_fun_man(lkl_params, s_z)
            )
            return lhs - (jnp.dot(rho, model.pst_prr_emb.embed(s_z)) + chi)

        zs = jax.random.normal(jax.random.PRNGKey(seed + 100), (100, model.lat_dim))
        assert jnp.allclose(jax.vmap(residual)(zs), 0.0, atol=1e-8)

    def test_conjugation_is_the_sum_of_the_branches(self) -> None:
        """Not just numerically right --- right for the stated reason."""
        model = cca()
        params = model.initialize(jax.random.PRNGKey(3), shape=0.4)
        lkl_params = model.likelihood_function(params)

        obs_bias, int_params = model.lkl_fun_man.split_coords(lkl_params)
        fst_bias, snd_bias = model.obs_man.split_coords(obs_bias)
        fst_int, snd_int = model.crs_man.coord_blocks(int_params)
        fst_lgm, snd_lgm = model.fst_lgm, model.snd_lgm

        expected = fst_lgm.conjugation_parameters(
            fst_lgm.lkl_fun_man.join_coords(fst_bias, fst_int)
        ) + snd_lgm.conjugation_parameters(
            snd_lgm.lkl_fun_man.join_coords(snd_bias, snd_int)
        )
        assert jnp.allclose(model.conjugation_parameters(lkl_params), expected)

    def test_dropping_a_branch_recovers_a_plain_lgm(self) -> None:
        """With the second interaction zeroed, conjugation matches the first branch alone."""
        model = cca()
        params = model.initialize(jax.random.PRNGKey(4), shape=0.4)
        obs_bias, int_params = model.lkl_fun_man.split_coords(
            model.likelihood_function(params)
        )
        fst_int, snd_int = model.crs_man.coord_blocks(int_params)
        muted = model.lkl_fun_man.join_coords(
            obs_bias, jnp.concatenate([fst_int, jnp.zeros_like(snd_int)])
        )

        fst_bias, _ = model.obs_man.split_coords(obs_bias)
        fst_lgm = model.fst_lgm
        expected = fst_lgm.conjugation_parameters(
            fst_lgm.lkl_fun_man.join_coords(fst_bias, fst_int)
        )
        assert jnp.allclose(model.conjugation_parameters(muted), expected)


class TestDensity:
    """Sampling and density evaluation over the joint $[x, y, z]$."""

    def test_sample_shape(self) -> None:
        model = cca()
        params = model.initialize(jax.random.PRNGKey(5), shape=0.3)
        assert model.sample(jax.random.PRNGKey(6), params, 20).shape == (
            20,
            model.data_dim,
        )

    def test_sufficient_statistic_shape(self) -> None:
        model = cca()
        x = jnp.zeros(model.data_dim)
        assert model.sufficient_statistic(x).shape == (model.dim,)

    def test_log_density_is_finite(self) -> None:
        model = cca()
        params = model.initialize(jax.random.PRNGKey(7), shape=0.3)
        xs = model.sample(jax.random.PRNGKey(8), params, 10)
        assert jnp.all(jnp.isfinite(jax.vmap(model.log_density, (None, 0))(params, xs)))

    def test_posterior_shape(self) -> None:
        model = cca()
        params = model.initialize(jax.random.PRNGKey(9), shape=0.3)
        xs = model.sample(jax.random.PRNGKey(10), params, 4)
        obs = xs[0, : model.obs_man.data_dim]
        assert model.posterior_at(params, obs).shape == (model.pst_man.dim,)


class TestLearning:
    """Gradient ascent on the observable log-likelihood improves the fit."""

    def test_gradient_ascent_increases_log_likelihood(self) -> None:
        model = cca(fst_dim=3, snd_dim=2, lat_dim=2)
        true_params = model.initialize(jax.random.PRNGKey(11), shape=0.5)
        sample = model.sample(jax.random.PRNGKey(12), true_params, 500)
        obs = sample[:, : model.obs_man.data_dim]

        params = model.initialize(jax.random.PRNGKey(13), shape=0.1)

        def loss(p: Array) -> Array:
            return -jnp.mean(
                jax.vmap(model.average_log_observable_density, (None, 0))(
                    p, obs[:, None, :]
                )
            )

        start = loss(params)
        for _ in range(20):
            params = params - 0.01 * jax.grad(loss)(params)
        assert loss(params) < start


class TestBranchRepresentations:
    """The two observables may differ in shape as well as dimension."""

    @pytest.mark.parametrize("snd_rep", [PositiveDefinite(), Diagonal(), Scale()])
    def test_mixed_covariance_structures(self, snd_rep: PositiveDefinite) -> None:
        model = CanonicalCorrelationAnalysis(
            fst_dim=3,
            fst_rep=PositiveDefinite(),
            snd_dim=2,
            snd_rep=snd_rep,
            lat_dim=2,
            pst_rep=PositiveDefinite(),
        )
        params = model.initialize(jax.random.PRNGKey(14), shape=0.3)
        assert params.shape == (model.dim,)
        assert sum(model.clq_dims) == model.dim
