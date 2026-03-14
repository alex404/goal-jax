"""Tests for Poisson-based Harmoniums.

Tests PoissonVonMisesHarmonium and VonMisesPopulationCode.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import (
    PoissonVonMisesHarmonium,
    VonMisesPopulationCode,
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
        hrm = PoissonVonMisesHarmonium(request.param, 1)
        return VonMisesPopulationCode(hrm=hrm)

    def test_dimensions(self, model: VonMisesPopulationCode) -> None:
        """Test model dimensions are correct."""
        n = model.n_neurons

        # Observable manifold (Poissons)
        assert model.obs_man.dim == n
        assert model.obs_man.data_dim == n

        # Posterior manifold (VonMisesProduct(1))
        assert model.pst_man.dim == 2
        assert model.pst_man.data_dim == 1

        # Interaction manifold
        assert model.hrm.int_man.dim == n * 2

        # Total = rho_dim + hrm_dim = 2 + (n + n*2 + 2)
        assert model.dim == 2 + n + n * 2 + 2


class TestVonMisesPopulationCodeTuning:
    """Test population code tuning curves."""

    @pytest.fixture(params=[4, 8])
    def model_and_params(
        self, request: pytest.FixtureRequest, key: Array
    ) -> tuple[VonMisesPopulationCode, Array]:
        """Create population code with parameters."""
        n_neurons = request.param
        model = VonMisesPopulationCode(hrm=PoissonVonMisesHarmonium(n_neurons, 1))
        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        params = model.initialize_from_tuning_curves(
            key=key,
            gains=jnp.ones(n_neurons) * 2.0,
            preferred=preferred,
            baselines=jnp.zeros(n_neurons),
        )
        return model, params

    def test_tuning_curve_shape(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test likelihood produces correct shape."""
        model, params = model_and_params
        _, hrm_params = model.split_coords(params)
        z = jnp.array([0.5])  # VonMisesProduct(1) data is 1D
        lkl_params = model.hrm.likelihood_at(hrm_params, z)
        assert lkl_params.shape == (model.n_neurons,)

    def test_tuning_curve_peaks_at_preferred(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test tuning curve peaks at preferred direction."""
        model, params = model_and_params
        _, hrm_params = model.split_coords(params)
        _, int_params, _ = model.hrm.split_coords(hrm_params)
        int_matrix = int_params.reshape(model.n_neurons, 2)
        preferred = jnp.arctan2(int_matrix[:, 1], int_matrix[:, 0])

        for i in range(model.n_neurons):
            pref = jnp.array([preferred[i]])
            opp = jnp.array([preferred[i] + jnp.pi])
            rate_at_pref = model.hrm.likelihood_at(hrm_params, pref)[i]
            rate_at_opposite = model.hrm.likelihood_at(hrm_params, opp)[i]
            assert rate_at_pref > rate_at_opposite

    def test_firing_rates_positive(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test firing rates are always positive."""
        model, params = model_and_params
        _, hrm_params = model.split_coords(params)
        test_stimuli = jnp.linspace(0, 2 * jnp.pi, 10)

        for z_scalar in test_stimuli:
            z = jnp.array([z_scalar])
            lkl_params = model.hrm.likelihood_at(hrm_params, z)
            rates = model.obs_man.to_mean(lkl_params)
            assert jnp.all(rates > 0)


class TestVonMisesPopulationCodePosterior:
    """Test population code posterior computation."""

    @pytest.fixture(params=[8, 16])
    def model_and_params(
        self, request: pytest.FixtureRequest, key: Array
    ) -> tuple[VonMisesPopulationCode, Array]:
        """Create population code with parameters."""
        n_neurons = request.param
        model = VonMisesPopulationCode(hrm=PoissonVonMisesHarmonium(n_neurons, 1))
        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        params = model.initialize_from_tuning_curves(
            key=key,
            gains=jnp.ones(n_neurons) * 2.0,
            preferred=preferred,
            baselines=jnp.zeros(n_neurons),
            n_regression_samples=10000,
        )
        return model, params

    def test_posterior_is_valid_vonmises(
        self, model_and_params: tuple[VonMisesPopulationCode, Array], key: Array
    ) -> None:
        """Test approximate posterior parameters are valid VonMises."""
        model, params = model_and_params
        x = jax.random.poisson(key, 5.0 * jnp.ones(model.n_neurons))

        q_params = model.approximate_posterior_at(params, x)
        assert q_params.shape == (2,)

        vm = model.pst_man.rep_man
        _, kappa = vm.split_mean_concentration(q_params)
        assert kappa >= 0

    def test_posterior_concentration_increases_with_spikes(
        self, model_and_params: tuple[VonMisesPopulationCode, Array]
    ) -> None:
        """Test posterior concentration increases with spike count."""
        model, params = model_and_params
        vm = model.pst_man.rep_man

        x_low = jnp.ones(model.n_neurons)
        x_high = 10 * jnp.ones(model.n_neurons)

        q_low = model.approximate_posterior_at(params, x_low)
        q_high = model.approximate_posterior_at(params, x_high)

        _, kappa_low = vm.split_mean_concentration(q_low)
        _, kappa_high = vm.split_mean_concentration(q_high)

        assert kappa_high > kappa_low


class TestVonMisesPopulationCodeSampling:
    """Test population code sampling."""

    @pytest.fixture(params=[8])
    def model_and_params(
        self, request: pytest.FixtureRequest, key: Array
    ) -> tuple[VonMisesPopulationCode, Array]:
        """Create population code with parameters."""
        n_neurons = request.param
        model = VonMisesPopulationCode(hrm=PoissonVonMisesHarmonium(n_neurons, 1))
        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        params = model.initialize_from_tuning_curves(
            key=key,
            gains=jnp.ones(n_neurons) * 2.0,
            preferred=preferred,
            baselines=jnp.zeros(n_neurons),
        )
        return model, params

    def test_sample_shape(
        self, model_and_params: tuple[VonMisesPopulationCode, Array], key: Array
    ) -> None:
        """Test sampling produces correct shape."""
        model, params = model_and_params
        n = 5
        samples = model.sample(key, params, n)
        # joint samples: (n, obs_data_dim + lat_data_dim)
        assert samples.shape == (n, model.obs_man.data_dim + model.pst_man.data_dim)

    def test_reconstruction(
        self, model_and_params: tuple[VonMisesPopulationCode, Array], key: Array
    ) -> None:
        """Test reconstruction produces finite outputs."""
        model, params = model_and_params
        x = jax.random.poisson(key, 5.0 * jnp.ones(model.n_neurons)).astype(
            jnp.float32
        )
        recon = model.reconstruct(params, x)
        assert recon.shape == (model.n_neurons,)
        assert jnp.all(jnp.isfinite(recon))
        assert jnp.all(recon > 0)  # Poisson rates are positive


class TestVonMisesPopulationCodeConjugation:
    """Test conjugation regression quality."""

    def test_regression_with_variation(self, key: Array) -> None:
        """Test regression captures variation with non-uniform gains."""
        n_neurons = 16
        model = VonMisesPopulationCode(hrm=PoissonVonMisesHarmonium(n_neurons, 1))

        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        gains = 1.0 + jax.random.uniform(key, (n_neurons,)) * 3.0
        baselines = jnp.zeros(n_neurons)

        # Build harmonium params manually
        int_col_1 = gains * jnp.cos(preferred)
        int_col_2 = gains * jnp.sin(preferred)
        int_params = jnp.stack([int_col_1, int_col_2], axis=1).ravel()
        prior_params = jnp.zeros(2)
        hrm_params = model.hrm.join_coords(baselines, int_params, prior_params)
        zero_rho = jnp.zeros(model.rho_man.dim)
        params = model.join_coords(zero_rho, hrm_params)

        # Regress rho
        key, reg_key = jax.random.split(key)
        rho, r_squared, _, _ = model.regress_conjugation_parameters(
            reg_key, params, 5000
        )

        # With non-uniform gains, rho should be non-trivial and R^2 reasonable
        assert jnp.linalg.norm(rho) > 0.01 or float(r_squared) > 0.3


class TestPoissonVonMisesHarmoniumBasics:
    """Test PoissonVonMisesHarmonium basics."""

    @pytest.fixture(params=[(8, 2), (16, 4)])
    def model(self, request: pytest.FixtureRequest) -> PoissonVonMisesHarmonium:
        """Create PoissonVonMisesHarmonium."""
        n_neurons, n_latent = request.param
        return PoissonVonMisesHarmonium(n_neurons, n_latent)

    def test_factory_function(self) -> None:
        """Test factory function creates correct model."""
        model = PoissonVonMisesHarmonium(16, 2)
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
