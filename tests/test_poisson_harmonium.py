"""Tests for Poisson-based Harmoniums.

Tests PoissonVonMisesHarmonium, VonMisesPopulationCode, and PoissonBernoulliHarmonium.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import (
    PoissonBernoulliHarmonium,  # pyright: ignore[reportAttributeAccessIssue]
    PoissonVonMisesHarmonium,
    VonMisesPopulationCode,
    poisson_bernoulli_harmonium,  # pyright: ignore[reportAttributeAccessIssue]
    poisson_vonmises_harmonium,
    von_mises_population_code,
)

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-4
ATOL = 1e-6


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestVonMisesPopulationCodeBasics:
    """Test basic VonMisesPopulationCode properties."""

    @pytest.fixture(params=[4, 8, 16])
    def model(self, request: pytest.FixtureRequest) -> VonMisesPopulationCode:
        """Create population code model."""
        return VonMisesPopulationCode(request.param)

    def test_dimensions(self, model: VonMisesPopulationCode) -> None:
        """Test model dimensions are correct."""
        n = model.n_neurons

        # Observable manifold (Poissons)
        assert model.obs_man.dim == n
        assert model.obs_man.data_dim == n

        # Latent manifold (VonMises)
        assert model.lat_man.dim == 2
        assert model.lat_man.data_dim == 1

        # Interaction manifold
        assert model.int_man.dim == n * 2

        # Total dimension
        assert model.dim == n + n * 2 + 2


class TestVonMisesPopulationCodeTuning:
    """Test population code tuning curves."""

    @pytest.fixture(params=[4, 8])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[VonMisesPopulationCode, Array]:
        """Create population code with parameters."""
        n_neurons = request.param
        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        gains = jnp.ones(n_neurons) * 2.0
        baselines = jnp.zeros(n_neurons)

        model, params = von_mises_population_code(
            n_neurons=n_neurons,
            gains=gains,
            preferred=preferred,
            baselines=baselines,
            prior_mean=0.0,
            prior_concentration=0.0,
        )
        return model, params

    def test_tuning_curve_shape(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test tuning curve has correct shape."""
        model, params = model_and_params
        z = jnp.array(0.5)
        log_rates = model.tuning_curve(params, z)
        assert log_rates.shape == (model.n_neurons,)

    def test_tuning_curve_peaks_at_preferred(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test tuning curve peaks at preferred direction."""
        model, params = model_and_params
        _, preferred, _, _ = model.split_tuning_parameters(params)

        for i in range(model.n_neurons):
            pref = preferred[i]
            rate_at_pref = model.tuning_curve(params, pref)[i]
            rate_at_opposite = model.tuning_curve(params, pref + jnp.pi)[i]
            assert rate_at_pref > rate_at_opposite

    def test_firing_rates_positive(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test firing rates are always positive."""
        model, params = model_and_params
        test_stimuli = jnp.linspace(0, 2 * jnp.pi, 10)

        for z in test_stimuli:
            rates = model.firing_rates(params, z)
            assert jnp.all(rates > 0)


class TestVonMisesPopulationCodeParameters:
    """Test population code parameter operations."""

    @pytest.fixture(params=[4, 8])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[VonMisesPopulationCode, Array]:
        """Create population code with parameters."""
        n_neurons = request.param
        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        gains = jnp.ones(n_neurons) * 2.0
        baselines = jnp.zeros(n_neurons)

        model, params = von_mises_population_code(
            n_neurons=n_neurons,
            gains=gains,
            preferred=preferred,
            baselines=baselines,
            prior_mean=0.0,
            prior_concentration=0.0,
        )
        return model, params

    def test_split_join_round_trip(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test split/join tuning parameters round-trip."""
        model, params = model_and_params
        gains, preferred, baselines, prior = model.split_tuning_parameters(params)
        recovered = model.join_tuning_parameters(gains, preferred, baselines, prior)

        assert jnp.allclose(params, recovered, rtol=1e-3, atol=1e-5)


class TestVonMisesPopulationCodePosterior:
    """Test population code posterior computation."""

    @pytest.fixture(params=[8, 16])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[VonMisesPopulationCode, Array]:
        """Create population code with parameters."""
        n_neurons = request.param
        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        gains = jnp.ones(n_neurons) * 2.0
        baselines = jnp.zeros(n_neurons)

        model, params = von_mises_population_code(
            n_neurons=n_neurons,
            gains=gains,
            preferred=preferred,
            baselines=baselines,
            prior_mean=0.0,
            prior_concentration=0.0,
        )
        return model, params

    def test_posterior_is_valid_vonmises(
        self, model_and_params: tuple[VonMisesPopulationCode, Array], key: Array
    ) -> None:
        """Test posterior parameters are valid VonMises."""
        model, params = model_and_params
        x = jax.random.poisson(key, 5.0 * jnp.ones(model.n_neurons))

        post_params = model.posterior_at(params, x)

        assert post_params.shape == (2,)
        _, kappa = model.lat_man.split_mean_concentration(post_params)
        assert kappa >= 0

    def test_posterior_concentration_increases_with_spikes(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test posterior concentration increases with spike count."""
        model, params = model_and_params

        x_low = jnp.ones(model.n_neurons)
        x_high = 10 * jnp.ones(model.n_neurons)

        kappa_low = model.posterior_concentration(params, x_low)
        kappa_high = model.posterior_concentration(params, x_high)

        assert kappa_high > kappa_low


class TestVonMisesPopulationCodeSampling:
    """Test population code sampling."""

    @pytest.fixture(params=[8])
    def model_and_params(
        self, request: pytest.FixtureRequest
    ) -> tuple[VonMisesPopulationCode, Array]:
        """Create population code with parameters."""
        n_neurons = request.param
        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        gains = jnp.ones(n_neurons) * 2.0
        baselines = jnp.zeros(n_neurons)

        model, params = von_mises_population_code(
            n_neurons=n_neurons,
            gains=gains,
            preferred=preferred,
            baselines=baselines,
            prior_mean=0.0,
            prior_concentration=0.0,
        )
        return model, params

    def test_sample_spikes_shape(
        self, model_and_params: tuple[VonMisesPopulationCode, Array], key: Array
    ) -> None:
        """Test spike sampling produces correct shape."""
        model, params = model_and_params
        z = jnp.array(1.0)

        spikes = model.sample_spikes(key, params, z)

        assert spikes.shape == (model.n_neurons,)
        assert jnp.all(spikes >= 0)

    def test_log_observable_density_finite(
        self, model_and_params: tuple[VonMisesPopulationCode, Array], key: Array
    ) -> None:
        """Test log observable density is finite."""
        model, params = model_and_params
        zs = jnp.array([0.0, jnp.pi / 2, jnp.pi])
        keys = jax.random.split(key, len(zs))
        xs = jax.vmap(lambda k, z: model.sample_spikes(k, params, z))(keys, zs)

        log_densities = jax.vmap(lambda x: model.log_observable_density(params, x))(xs)
        assert jnp.all(jnp.isfinite(log_densities))


class TestVonMisesPopulationCodeConjugation:
    """Test conjugation regression quality."""

    def test_regression_with_variation(self, key: Array) -> None:
        """Test regression captures variation with non-uniform gains."""
        n_neurons = 16
        model = VonMisesPopulationCode(n_neurons)

        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        gains = 1.0 + jax.random.uniform(key, (n_neurons,)) * 3.0
        baselines = jnp.zeros(n_neurons)
        prior_params = model.lat_man.join_mean_concentration(0.0, 0.0)

        params = model.join_tuning_parameters(gains, preferred, baselines, prior_params)

        obs_params, int_params, _ = model.split_coords(params)
        int_matrix = int_params.reshape(model.n_neurons, 2)

        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        test_grid = jnp.linspace(0, 2 * jnp.pi, 30, endpoint=False)

        def true_log_partition(z: Array) -> Array:
            suff = model.lat_man.sufficient_statistic(z)
            log_rates = obs_params + int_matrix @ suff
            return jnp.sum(jnp.exp(log_rates))

        def approx_log_partition(z: Array) -> Array:
            suff = model.lat_man.sufficient_statistic(z)
            return jnp.dot(rho, suff)

        true_vals = jax.vmap(true_log_partition)(test_grid)
        approx_vals = jax.vmap(approx_log_partition)(test_grid)

        true_centered = true_vals - jnp.mean(true_vals)
        approx_centered = approx_vals - jnp.mean(approx_vals)

        true_std = jnp.std(true_vals)
        if true_std > 0.1:
            correlation = jnp.sum(true_centered * approx_centered) / (
                jnp.sqrt(jnp.sum(true_centered**2) + 1e-8)
                * jnp.sqrt(jnp.sum(approx_centered**2) + 1e-8)
            )
            # Linear approximation quality depends on population size and gain variation
            assert correlation > 0.3


class TestPoissonVonMisesHarmoniumBasics:
    """Test PoissonVonMisesHarmonium basics."""

    @pytest.fixture(params=[(8, 2), (16, 4)])
    def model(self, request: pytest.FixtureRequest) -> PoissonVonMisesHarmonium:
        """Create PoissonVonMisesHarmonium."""
        n_neurons, n_latent = request.param
        return poisson_vonmises_harmonium(n_neurons, n_latent)

    def test_factory_function(self) -> None:
        """Test factory function creates correct model."""
        model = poisson_vonmises_harmonium(16, 2)
        assert model.n_neurons == 16
        assert model.n_latent == 2

    def test_dimensions(self, model: PoissonVonMisesHarmonium) -> None:
        """Test dimensions are correct."""
        assert model.obs_man.data_dim == model.n_neurons
        assert model.pst_man.data_dim == model.n_latent

    def test_initialize(
        self, model: PoissonVonMisesHarmonium, key: Array
    ) -> None:
        """Test initialization produces valid parameters."""
        params = model.initialize(key)
        assert jnp.all(jnp.isfinite(params))


class TestPoissonBernoulliHarmoniumBasics:
    """Test PoissonBernoulliHarmonium basics."""

    @pytest.fixture(params=[(8, 4), (16, 8)])
    def model(self, request: pytest.FixtureRequest) -> PoissonBernoulliHarmonium:
        """Create PoissonBernoulliHarmonium."""
        n_obs, n_lat = request.param
        return poisson_bernoulli_harmonium(n_obs, n_lat)

    def test_factory_function(self) -> None:
        """Test factory function creates correct model."""
        model = poisson_bernoulli_harmonium(16, 8)
        assert model.n_observable == 16
        assert model.n_latent == 8

    def test_dimensions(self, model: PoissonBernoulliHarmonium) -> None:
        """Test dimensions are correct."""
        assert model.obs_man.data_dim == model.n_observable
        assert model.pst_man.data_dim == model.n_latent

    def test_initialize(
        self, model: PoissonBernoulliHarmonium, key: Array
    ) -> None:
        """Test initialization produces valid parameters."""
        params = model.initialize(key)
        assert jnp.all(jnp.isfinite(params))

    def test_posterior_at(
        self, model: PoissonBernoulliHarmonium, key: Array
    ) -> None:
        """Test posterior computation produces valid parameters."""
        params = model.initialize(key)
        x = jax.random.poisson(key, 3.0 * jnp.ones(model.n_observable))

        posterior = model.posterior_at(params, x)
        assert posterior.shape == (model.pst_man.dim,)
        assert jnp.all(jnp.isfinite(posterior))
