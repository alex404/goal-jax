"""Tests for population codes."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import VonMisesPopulationCode, von_mises_population_code

# Configure JAX
jax.config.update("jax_platform_name", "cpu")

# Test Parameters
relative_tol = 1e-4
absolute_tol = 1e-6


@pytest.fixture(params=[4, 8, 16])
def n_neurons(request: pytest.FixtureRequest) -> int:
    """Parametrize tests over different numbers of neurons."""
    return request.param


@pytest.fixture
def pop_code(n_neurons: int) -> VonMisesPopulationCode:
    """Create population code model."""
    return VonMisesPopulationCode(n_neurons)


@pytest.fixture
def pop_code_params(n_neurons: int) -> tuple[VonMisesPopulationCode, Array]:
    """Create population code model with parameters."""
    # Evenly spaced preferred directions
    preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
    # Uniform gains and baselines
    gains = jnp.ones(n_neurons) * 2.0
    baselines = jnp.zeros(n_neurons)

    model, params = von_mises_population_code(
        n_neurons=n_neurons,
        gains=gains,
        preferred=preferred,
        baselines=baselines,
        prior_mean=0.0,
        prior_concentration=0.0,  # Uniform prior
    )
    return model, params


def test_model_dimensions(pop_code: VonMisesPopulationCode):
    """Test model dimensions are correct."""
    n = pop_code.n_neurons

    # Observable manifold
    assert pop_code.obs_man.dim == n
    assert pop_code.obs_man.data_dim == n

    # Latent manifold (VonMises)
    assert pop_code.lat_man.dim == 2
    assert pop_code.lat_man.data_dim == 1

    # Interaction manifold
    assert pop_code.int_man.dim == n * 2

    # Total dimension
    assert pop_code.dim == n + n * 2 + 2


def test_tuning_curve_shape(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test tuning curve has correct shape."""
    model, params = pop_code_params

    z = jnp.array(0.5)
    log_rates = model.tuning_curve(params, z)

    assert log_rates.shape == (model.n_neurons,)


def test_tuning_curve_peaks_at_preferred(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test tuning curve peaks at preferred direction."""
    model, params = pop_code_params
    _, preferred, _, _ = model.split_tuning_parameters(params)

    # For each neuron, the tuning curve should peak at its preferred direction
    for i in range(model.n_neurons):
        pref = preferred[i]

        # Evaluate at preferred and at preferred + pi (opposite direction)
        rate_at_pref = model.tuning_curve(params, pref)[i]
        rate_at_opposite = model.tuning_curve(params, pref + jnp.pi)[i]

        # Rate should be higher at preferred direction
        assert rate_at_pref > rate_at_opposite, f"Neuron {i} should peak at preferred"


def test_join_split_round_trip(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test that split_tuning_parameters inverts join_tuning_parameters."""
    model, params = pop_code_params

    # Split and rejoin
    gains, preferred, baselines, prior = model.split_tuning_parameters(params)
    params_reconstructed = model.join_tuning_parameters(
        gains, preferred, baselines, prior
    )

    # Use larger tolerance because conjugation_parameters involves regression
    # which introduces small numerical differences on each computation
    assert jnp.allclose(params, params_reconstructed, rtol=1e-3, atol=1e-5), (
        "Round-trip should recover original parameters"
    )


def test_conjugation_regression_quality(n_neurons: int):
    """Test regression approximation is accurate.

    Uses non-uniform gains to create variation in the log-partition.
    With uniform coverage (evenly spaced neurons + equal gains),
    the log-partition is nearly constant and there's no variation to capture.
    """
    model = VonMisesPopulationCode(n_neurons)

    # Use non-uniform gains to create actual variation
    # This simulates a population with some neurons more sensitive than others
    key = jax.random.PRNGKey(42)
    preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
    gains = 1.0 + jax.random.uniform(key, (n_neurons,)) * 3.0  # Gains 1-4
    baselines = jnp.zeros(n_neurons)
    prior_params = model.lat_man.join_mean_concentration(0.0, 0.0)

    params = model.join_tuning_parameters(gains, preferred, baselines, prior_params)

    obs_params, int_params, _ = model.split_coords(params)
    int_matrix = int_params.reshape(model.n_neurons, 2)

    # Compute conjugation parameters
    lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
    rho = model.conjugation_parameters(lkl_params)

    # Test at many grid points
    test_grid = jnp.linspace(0, 2 * jnp.pi, 50, endpoint=False)

    def true_log_partition(z: Array) -> Array:
        suff = model.lat_man.sufficient_statistic(z)
        log_rates = obs_params + int_matrix @ suff
        return jnp.sum(jnp.exp(log_rates))

    def approx_log_partition(z: Array) -> Array:
        suff = model.lat_man.sufficient_statistic(z)
        return jnp.dot(rho, suff)

    true_vals = jax.vmap(true_log_partition)(test_grid)
    approx_vals = jax.vmap(approx_log_partition)(test_grid)

    # The approximation should capture the variation (up to a constant)
    # Check that the correlation is high when there's variation
    true_centered = true_vals - jnp.mean(true_vals)
    approx_centered = approx_vals - jnp.mean(approx_vals)

    # Only check correlation if there's significant variation
    true_std = jnp.std(true_vals)
    if true_std > 0.1:
        correlation = jnp.sum(true_centered * approx_centered) / (
            jnp.sqrt(jnp.sum(true_centered**2)) * jnp.sqrt(jnp.sum(approx_centered**2))
        )
        # The linear approximation is not exact, especially for small populations
        # With more neurons, the approximation improves
        min_correlation = 0.3 if n_neurons < 8 else 0.5 if n_neurons < 16 else 0.8
        assert correlation > min_correlation, (
            f"Regression correlation {correlation:.3f} should be > {min_correlation}"
        )
    else:
        # With uniform coverage, log-partition is constant - regression should give small rho
        assert jnp.linalg.norm(rho) < 1.0, (
            "With constant log-partition, rho should be small"
        )


def test_posterior_is_valid_vonmises(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test posterior parameters are valid VonMises."""
    model, params = pop_code_params

    # Generate some observations
    key = jax.random.PRNGKey(42)
    x = jax.random.poisson(key, 5.0 * jnp.ones(model.n_neurons))

    post_params = model.posterior_at(params, x)

    # VonMises has 2 natural parameters
    assert post_params.shape == (2,)

    # The concentration (norm of params) should be positive
    _, kappa = model.lat_man.split_mean_concentration(post_params)
    assert kappa >= 0, "Concentration should be non-negative"


def test_posterior_mean_near_true_stimulus(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test posterior mean is close to true stimulus with high firing rates."""
    model, params = pop_code_params

    key = jax.random.PRNGKey(123)

    # Use fixed stimulus values instead of sampling (VonMises sampling is slow)
    zs = jnp.array([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])

    # Sample spike counts at each stimulus
    keys = jax.random.split(key, len(zs))
    xs = jax.vmap(lambda k, z: model.sample_spikes(k, params, z))(keys, zs)

    # Compute posterior means
    posterior_means = jax.vmap(lambda x: model.posterior_mean(params, x))(xs)

    # Compute circular errors (handling wraparound)
    errors = jnp.abs(jnp.angle(jnp.exp(1j * (posterior_means - zs))))

    # Mean error should be reasonably small
    mean_error = jnp.mean(errors)
    assert mean_error < jnp.pi / 2, f"Mean error {mean_error} should be < π/2"


def test_posterior_concentration_increases_with_spike_count(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test that posterior concentration increases with more spikes."""
    model, params = pop_code_params

    # Low spike count
    x_low = jnp.ones(model.n_neurons)

    # High spike count (scaled up)
    x_high = 10 * jnp.ones(model.n_neurons)

    kappa_low = model.posterior_concentration(params, x_low)
    kappa_high = model.posterior_concentration(params, x_high)

    assert kappa_high > kappa_low, (
        "Higher spike counts should give higher concentration"
    )


def test_sample_spikes_shape(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test spike sampling produces correct shape."""
    model, params = pop_code_params

    key = jax.random.PRNGKey(0)
    z = jnp.array(1.0)

    spikes = model.sample_spikes(key, params, z)

    assert spikes.shape == (model.n_neurons,)
    assert jnp.all(spikes >= 0), "Spike counts should be non-negative"


def test_sample_spikes_with_multiple_stimuli(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test spike sampling at multiple stimuli produces correct shapes."""
    model, params = pop_code_params

    key = jax.random.PRNGKey(0)
    n_samples = 4

    # Use fixed stimuli instead of sampling (VonMises sampling is slow)
    zs = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)

    keys = jax.random.split(key, n_samples)
    xs = jax.vmap(lambda k, z: model.sample_spikes(k, params, z))(keys, zs)

    assert xs.shape == (n_samples, model.n_neurons)
    assert jnp.all(xs >= 0), "Spike counts should be non-negative"


def test_firing_rates_positive(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test firing rates are always positive."""
    model, params = pop_code_params

    # Test at various stimuli
    test_stimuli = jnp.linspace(0, 2 * jnp.pi, 20)

    for z in test_stimuli:
        rates = model.firing_rates(params, z)
        assert jnp.all(rates > 0), f"Firing rates should be positive at z={z}"


def test_log_observable_density(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test log observable density is well-defined."""
    model, params = pop_code_params

    # Sample some observations at fixed stimuli (avoid slow VonMises sampling)
    key = jax.random.PRNGKey(42)
    zs = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    keys = jax.random.split(key, len(zs))
    xs = jax.vmap(lambda k, z: model.sample_spikes(k, params, z))(keys, zs)

    # Compute log densities
    log_densities = jax.vmap(lambda x: model.log_observable_density(params, x))(xs)

    # Should be finite
    assert jnp.all(jnp.isfinite(log_densities)), "Log densities should be finite"


def test_prior_with_uniform_prior(
    pop_code_params: tuple[VonMisesPopulationCode, Array],
):
    """Test prior is near-uniform when initialized with zero concentration."""
    model, params = pop_code_params

    prior_params = model.prior(params)
    _, kappa = model.lat_man.split_mean_concentration(prior_params)

    # With zero prior concentration and regression adjustment,
    # the effective prior concentration should be small
    # (the regression shifts it slightly)
    assert kappa < 5.0, f"Prior should be relatively flat, got kappa={kappa}"
