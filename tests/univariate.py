"""Tests for univariate exponential family models.

Tests density integration, sufficient statistic convergence, and parameter recovery.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import PositiveDefinite
from goal.models import Categorical, CoMPoisson, Normal, Poisson, VonMises

jax.config.update("jax_platform_name", "cpu")

# Test parameters
n_trials = 5
sample_sizes = [100, 1000]

# Model groups: all differentiable models
all_models = [
    ("categorical", Categorical(n_categories=5)),
    ("poisson", Poisson()),
    ("normal", Normal(1, PositiveDefinite())),
    ("von_mises", VonMises()),
    ("com_poisson", CoMPoisson()),
]

# Analytic models only (support to_natural, relative_entropy)
analytic_models = [
    ("categorical", Categorical(n_categories=5)),
    ("poisson", Poisson()),
    ("normal", Normal(1, PositiveDefinite())),
]


def integrate_density(model, params: Array, xs: Array, dx: float = 1.0) -> Array:
    """Integrate density over given points."""
    densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
    return jnp.sum(densities) * dx


@pytest.fixture(params=all_models, ids=lambda x: x[0])
def model(request):
    """All differentiable models."""
    return request.param[1]


@pytest.fixture(params=analytic_models, ids=lambda x: x[0])
def analytic_model(request):
    """Analytic models only."""
    return request.param[1]


def test_density_integrates_to_one(model):
    """Test that density integrates to 1."""
    key = jax.random.PRNGKey(0)
    results = []

    for i in range(n_trials):
        params = model.initialize(jax.random.fold_in(key, i))

        if isinstance(model, Categorical):
            xs = jnp.arange(model.n_categories)
            total = integrate_density(model, params, xs)
        elif isinstance(model, (Poisson, CoMPoisson)):
            xs = jnp.arange(500)
            total = integrate_density(model, params, xs)
        elif isinstance(model, VonMises):
            xs = jnp.linspace(-jnp.pi, jnp.pi, 1000)
            total = integrate_density(model, params, xs, dx=2 * jnp.pi / 1000)
        elif isinstance(model, Normal):
            mean_params = model.to_mean(params)
            mean, var = model.split_mean_covariance(mean_params)
            std = jnp.sqrt(var)
            xs = jnp.linspace(mean - 6 * std, mean + 6 * std, 1000).ravel()
            total = integrate_density(model, params, xs, dx=12 * std.item() / 1000)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        results.append(float(total))

    mean_total = jnp.mean(jnp.array(results))
    assert jnp.allclose(mean_total, 1.0, rtol=1e-2)


def test_sufficient_statistic_convergence(model):
    """Test that sample sufficient statistics converge to expected values."""
    key = jax.random.PRNGKey(42)
    params = model.initialize(key)
    true_means = model.to_mean(params)

    samples = model.sample(jax.random.fold_in(key, 1), params, max(sample_sizes))

    mses = []
    for n in sample_sizes:
        est_means = model.average_sufficient_statistic(samples[:n])
        mse = float(jnp.mean((est_means - true_means) ** 2))
        mses.append(mse)

    assert mses[0] > mses[-1], f"MSE did not decrease: {mses}"


def test_natural_parameters_valid(model):
    """Test that initialized parameters pass validity check."""
    key = jax.random.PRNGKey(789)

    for i in range(n_trials):
        params = model.initialize(jax.random.fold_in(key, i))
        assert model.check_natural_parameters(params)


def test_parameter_recovery(analytic_model):
    """Test natural -> mean -> natural round trip."""
    key = jax.random.PRNGKey(123)

    for i in range(n_trials):
        params = analytic_model.initialize(jax.random.fold_in(key, i))
        mean_params = analytic_model.to_mean(params)
        recovered = analytic_model.to_natural(mean_params)
        error = float(jnp.mean((recovered - params) ** 2))
        assert error < 1e-5, f"Parameter recovery error: {error}"


def test_relative_entropy_with_self(analytic_model):
    """Test that relative entropy with itself is zero."""
    key = jax.random.PRNGKey(456)

    for i in range(n_trials):
        params = analytic_model.initialize(jax.random.fold_in(key, i))
        mean_params = analytic_model.to_mean(params)
        re = analytic_model.relative_entropy(mean_params, params)
        assert re >= -1e-6, f"Negative relative entropy: {re}"
        assert jnp.abs(re) < 1e-5, f"Self relative entropy not zero: {re}"
