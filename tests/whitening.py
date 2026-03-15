"""Tests for whitening across LGM, MFA, and HMoG models.

Verifies that whitening preserves the observable marginal distribution p(x)
while transforming the latent prior to N(0, I), and that EM steps match
manual whiten_prior + to_natural.
"""

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Diagonal
from goal.models import MixtureOfFactorAnalyzers, analytic_hmog, factor_analysis

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

ATOL = 1e-5


def test_fa_whiten_preserves_observable_distribution() -> None:
    """FA whitening preserves p(x) while setting latent prior to N(0,I)."""
    fa = factor_analysis(obs_dim=8, lat_dim=3)
    key = jax.random.PRNGKey(0)
    params0 = fa.initialize(key, location=0.0, shape=0.5)

    # Perturb latent mean away from N(0,I)
    means = fa.to_mean(params0)
    obs_m, int_m, lat_m = fa.split_coords(means)
    lat_mean, lat_cov = fa.lat_man.split_mean_covariance(lat_m)
    perturbed_lat_m = fa.lat_man.join_mean_covariance(
        lat_mean + 0.3 * jnp.ones_like(lat_mean), lat_cov
    )
    perturbed_means = fa.join_coords(obs_m, int_m, perturbed_lat_m)
    perturbed_params = fa.to_natural(perturbed_means)

    # Whiten
    whitened_means = fa.whiten_prior(fa.to_mean(perturbed_params))
    whitened_params = fa.to_natural(whitened_means)

    # Observable marginal unchanged
    _, obs_nat_before = fa.observable_distribution(perturbed_params)
    _, obs_nat_after = fa.observable_distribution(whitened_params)
    assert jnp.allclose(obs_nat_before, obs_nat_after, atol=ATOL)

    # Latent prior is now standard normal
    _, prior_nat = fa.split_conjugated(whitened_params)
    prior_mean = fa.lat_man.to_mean(prior_nat)
    assert jnp.allclose(prior_mean, fa.lat_man.standard_normal(), atol=ATOL)


def test_mfa_whiten_preserves_observable_distribution() -> None:
    """MFA whitening preserves per-component p(x|k) and sets each latent prior to N(0,I)."""
    fa = factor_analysis(obs_dim=8, lat_dim=3)
    mfa = MixtureOfFactorAnalyzers(n_categories=4, bas_hrm=fa)

    params = mfa.initialize(jax.random.PRNGKey(42), location=0.0, shape=0.5)

    mix_params = mfa.to_mixture_coords(params)
    comp_nats, cat_nat = mfa.mix_man.split_natural_mixture(mix_params)

    def perturb(comp_nat: Array) -> Array:
        m = fa.to_mean(comp_nat)
        obs_m, int_m, lat_m = fa.split_coords(m)
        lat_mean, lat_cov = fa.lat_man.split_mean_covariance(lat_m)
        new_lat_m = fa.lat_man.join_mean_covariance(lat_mean + 0.5, lat_cov)
        return fa.to_natural(fa.join_coords(obs_m, int_m, new_lat_m))

    perturbed_comp_nats = mfa.mix_man.cmp_man.map(perturb, comp_nats, flatten=True)
    perturbed_mix_params = mfa.mix_man.join_natural_mixture(perturbed_comp_nats, cat_nat)

    perturbed_params = mfa.from_mixture_coords(perturbed_mix_params)
    whitened_means = mfa.whiten_prior(mfa.to_mean(perturbed_params))
    whitened_mix_means = mfa.to_mixture_coords(whitened_means)

    orig_comp_nats, _ = mfa.mix_man.split_natural_mixture(perturbed_mix_params)
    wht_comp_means, _ = mfa.mix_man.split_mean_mixture(whitened_mix_means)

    for k in range(4):
        orig_k = mfa.mix_man.cmp_man.get_replicate(orig_comp_nats, k)
        wht_k_mean = mfa.mix_man.cmp_man.get_replicate(wht_comp_means, k)
        wht_k_nat = fa.to_natural(wht_k_mean)

        _, obs_nat_orig = fa.observable_distribution(orig_k)
        _, obs_nat_wht = fa.observable_distribution(wht_k_nat)
        assert jnp.allclose(obs_nat_orig, obs_nat_wht, atol=ATOL)

        _, prior_nat = fa.split_conjugated(wht_k_nat)
        prior_mean = fa.lat_man.to_mean(prior_nat)
        assert jnp.allclose(prior_mean, fa.lat_man.standard_normal(), atol=ATOL)


def test_hmog_whiten_preserves_log_likelihood() -> None:
    """HMoG whitening preserves the observable log-likelihood."""
    model = analytic_hmog(obs_dim=8, obs_rep=Diagonal(), lat_dim=3, n_components=4)
    key = jax.random.PRNGKey(7)
    params = model.initialize(key, location=0.0, shape=0.5)
    xs = jax.random.normal(jax.random.split(key)[0], (200, 8))

    whitened_params = model.to_natural(model.whiten_prior(model.to_mean(params)))

    ll_before = model.average_log_observable_density(params, xs)
    ll_after = model.average_log_observable_density(whitened_params, xs)
    assert jnp.allclose(ll_before, ll_after, atol=1e-4)


def test_hmog_em_applies_whitening_round_trip() -> None:
    """AnalyticHMoG EM matches manual whiten_prior + to_natural."""
    model = analytic_hmog(obs_dim=8, obs_rep=Diagonal(), lat_dim=3, n_components=4)
    key = jax.random.PRNGKey(11)
    params = model.initialize(key, location=0.0, shape=0.5)
    xs = jax.random.normal(jax.random.PRNGKey(12), (128, 8))

    q = model.mean_posterior_statistics(params, xs)
    manual_em = model.to_natural(model.whiten_prior(q))
    em_params = model.expectation_maximization(params, xs)
    assert jnp.allclose(em_params, manual_em, atol=ATOL)


def test_mfa_em_matches_whiten_then_to_natural() -> None:
    """MFA expectation_maximization matches manual whiten_prior + to_natural."""
    fa = factor_analysis(obs_dim=6, lat_dim=2)
    mfa = MixtureOfFactorAnalyzers(n_categories=3, bas_hrm=fa)

    params = mfa.initialize(jax.random.PRNGKey(77), location=0.0, shape=0.5)
    xs = jax.random.normal(jax.random.PRNGKey(78), (128, 6))

    q = mfa.mean_posterior_statistics(params, xs)
    manual_em = mfa.to_natural(mfa.whiten_prior(q))
    em_params = mfa.expectation_maximization(params, xs)
    assert jnp.allclose(em_params, manual_em, atol=ATOL)
