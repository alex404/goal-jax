"""Tests for models/graphical/mixture.py.

Covers CompleteMixtureOfSymmetric, CompleteMixtureOfConjugated, and
MixtureOfFactorAnalyzers. Verifies dimension consistency, conjugation parameters,
posterior computation, interaction blocks, mixture representation round-trips,
asymmetric pst/prr handling, and to_natural/to_mean inversion.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Diagonal
from goal.models import DiagonalNormal, FullNormal, MixtureOfFactorAnalyzers, factor_analysis
from goal.models.graphical.mixture import (
    CompleteMixtureOfConjugated,
    CompleteMixtureOfSymmetric,
)
from goal.models.harmonium.lgm import NormalLGM

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


# --- Symmetric mixture (FactorAnalysis base) ---


class TestCompleteMixtureOfSymmetric:
    """Test CompleteMixtureOfSymmetric with FactorAnalysis base."""

    @pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array]:
        obs_dim, lat_dim, n_cat = request.param
        base_fa = factor_analysis(obs_dim=obs_dim, lat_dim=lat_dim)
        model = CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_fa
        )
        params = model.initialize(jax.random.PRNGKey(42), location=0.0, shape=1.0)
        return model, params

    def test_dimension_consistency(
        self,
        model_and_params: tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, params = model_and_params
        assert params.shape[0] == model.dim
        obs, int_params, lat = model.split_coords(params)
        assert obs.shape[0] == model.obs_man.dim
        assert int_params.shape[0] == model.int_man.dim
        assert lat.shape[0] == model.pst_man.dim

    def test_domain_codomain_consistency(
        self,
        model_and_params: tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, _ = model_and_params
        assert model.int_man.dom_man.dim == model.pst_man.dim
        assert model.int_man.cod_man.dim == model.obs_man.dim
        assert model.lat_man.dim == model.pst_man.dim
        assert model.lat_man.dim == model.prr_man.dim

    def test_conjugation_parameters_shape(
        self,
        model_and_params: tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, params = model_and_params
        obs, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs, int_params)
        rho = model.conjugation_parameters(lkl_params)
        assert rho.shape[0] == model.prr_man.dim

    def test_posterior(
        self,
        model_and_params: tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, params = model_and_params
        obs = jnp.ones(model.obs_man.data_dim)
        posterior = model.posterior_at(params, obs)
        assert posterior.shape[0] == model.pst_man.dim

        soft = model.posterior_soft_assignments(params, obs)
        assert soft.shape[0] == model.n_categories
        assert jnp.allclose(jnp.sum(soft), 1.0)

        hard = model.posterior_hard_assignment(params, obs)
        assert 0 <= hard < model.n_categories

    def test_mixture_round_trip(
        self,
        model_and_params: tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array],
    ) -> None:
        """to_mixture_coords / from_mixture_coords is invertible in both natural and mean coords."""
        model, params = model_and_params
        assert jnp.allclose(model.from_mixture_coords(model.to_mixture_coords(params)), params, atol=1e-10)

        means = model.to_mean(params)
        assert jnp.allclose(model.from_mixture_coords(model.to_mixture_coords(means)), means, atol=1e-10)

    def test_mixture_means_match(
        self,
        model_and_params: tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array],
    ) -> None:
        """Direct mean conversion matches natural->mean on mix_man."""
        model, params = model_and_params
        means = model.to_mean(params)
        direct = model.to_mixture_coords(means)
        via_natural = model.mix_man.to_mean(model.to_mixture_coords(params))
        assert jnp.allclose(direct, via_natural, atol=1e-10)

    def test_interaction_blocks(
        self,
        model_and_params: tuple[CompleteMixtureOfSymmetric[DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, params = model_and_params
        _, int_params, _ = model.split_coords(params)
        xy, xyk, xk = model.int_man.coord_blocks(int_params)
        assert xy.shape[0] + xyk.shape[0] + xk.shape[0] == model.int_man.dim


# --- Asymmetric mixture (NormalLGM base, pst_man != prr_man) ---


class TestCompleteMixtureOfConjugated:
    """Test CompleteMixtureOfConjugated with asymmetric base."""

    @pytest.fixture(params=[(3, 2, 2), (4, 2, 3)])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal], Array]:
        obs_dim, lat_dim, n_cat = request.param
        base_lgm = NormalLGM(obs_dim=obs_dim, obs_rep=Diagonal(), lat_dim=lat_dim, pst_rep=Diagonal())
        model = CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal](
            n_categories=n_cat, bas_hrm=base_lgm
        )
        params = model.initialize(jax.random.PRNGKey(42), location=0.0, shape=1.0)
        return model, params

    def test_dimension_consistency(
        self,
        model_and_params: tuple[CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, params = model_and_params
        assert params.shape[0] == model.dim
        obs, int_params, lat = model.split_coords(params)
        assert obs.shape[0] == model.obs_man.dim
        assert int_params.shape[0] == model.int_man.dim
        assert lat.shape[0] == model.pst_man.dim

    def test_pst_prr_dimensions_differ(
        self,
        model_and_params: tuple[CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, _ = model_and_params
        assert model.pst_man.dim != model.prr_man.dim
        assert model.pst_man.dim < model.prr_man.dim

    def test_conjugation_parameters_in_prr_space(
        self,
        model_and_params: tuple[CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, params = model_and_params
        obs, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs, int_params)
        rho = model.conjugation_parameters(lkl_params)
        assert rho.shape[0] == model.prr_man.dim

    def test_posterior_and_assignments(
        self,
        model_and_params: tuple[CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, params = model_and_params
        obs = jnp.ones(model.obs_man.data_dim)
        posterior = model.posterior_at(params, obs)
        assert posterior.shape[0] == model.pst_man.dim

        soft = model.posterior_soft_assignments(params, obs)
        assert soft.shape[0] == model.n_categories
        assert jnp.allclose(jnp.sum(soft), 1.0)

    def test_embedding_roundtrip(
        self,
        model_and_params: tuple[CompleteMixtureOfConjugated[DiagonalNormal, DiagonalNormal, FullNormal], Array],
    ) -> None:
        model, _ = model_and_params
        emb = model.pst_prr_emb
        pst_params = model.pst_man.zeros()
        prr_params = emb.embed(pst_params)
        assert prr_params.shape[0] == model.prr_man.dim
        recovered = emb.project(prr_params)
        assert jnp.allclose(recovered, pst_params)


# --- MFA to_natural round-trip (moved from whitening) ---


def test_mfa_to_natural_round_trip() -> None:
    """MFA to_natural is a left-inverse of to_mean: to_natural(to_mean(params)) == params."""
    fa = factor_analysis(obs_dim=8, lat_dim=3)
    mfa = MixtureOfFactorAnalyzers(n_categories=4, bas_hrm=fa)

    params = mfa.initialize(jax.random.PRNGKey(99), location=0.0, shape=0.5)
    means = mfa.to_mean(params)
    recovered_params = mfa.to_natural(means)

    assert jnp.allclose(params, recovered_params, atol=1e-5), (
        "MFA to_natural(to_mean(params)) != params"
    )
