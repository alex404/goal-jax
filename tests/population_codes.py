"""Tests for models/harmonium/population_codes.py.

Covers PoissonVonMisesHarmonium and VonMisesPopulationCode. Verifies dimensions,
tuning curves, posterior computation, sampling, conjugation regression, and
basic harmonium initialization.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import PoissonVonMisesHarmonium, VonMisesPopulationCode

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


def _make_population_code(
    n_neurons: int, key: Array, *, n_regression_samples: int = 1000
) -> tuple[VonMisesPopulationCode, Array]:
    """Create a VonMisesPopulationCode with uniform preferred directions."""
    model = VonMisesPopulationCode(hrm=PoissonVonMisesHarmonium(n_neurons, 1))
    preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
    params = model.initialize_from_tuning_curves(
        key=key,
        gains=jnp.ones(n_neurons) * 2.0,
        preferred=preferred,
        baselines=jnp.zeros(n_neurons),
        n_regression_samples=n_regression_samples,
    )
    return model, params


class TestVonMisesPopulationCode:
    """Test VonMisesPopulationCode dimensions, tuning curves, and posterior."""

    @pytest.mark.parametrize("n", [4, 8, 16])
    def test_dimensions(self, n: int) -> None:
        hrm = PoissonVonMisesHarmonium(n, 1)
        model = VonMisesPopulationCode(hrm=hrm)
        assert model.obs_man.dim == n
        assert model.obs_man.data_dim == n
        assert model.pst_man.dim == 2
        assert model.pst_man.data_dim == 1
        assert model.hrm.int_man.dim == n * 2
        assert model.dim == 2 + n + n * 2 + 2

    @pytest.mark.parametrize("n", [4, 8])
    def test_tuning_curve_peaks_at_preferred(self, n: int) -> None:
        """Each neuron's firing rate is higher at its preferred direction."""
        model, params = _make_population_code(n, jax.random.PRNGKey(42))
        _, hrm_params = model.split_coords(params)
        _, int_params, _ = model.hrm.split_coords(hrm_params)
        int_matrix = int_params.reshape(n, 2)
        preferred = jnp.arctan2(int_matrix[:, 1], int_matrix[:, 0])

        for i in range(n):
            pref = jnp.array([preferred[i]])
            opp = jnp.array([preferred[i] + jnp.pi])
            assert model.hrm.likelihood_at(hrm_params, pref)[i] > model.hrm.likelihood_at(
                hrm_params, opp
            )[i]

    def test_firing_rates_positive(self) -> None:
        model, params = _make_population_code(8, jax.random.PRNGKey(42))
        _, hrm_params = model.split_coords(params)
        for z_scalar in jnp.linspace(0, 2 * jnp.pi, 10):
            rates = model.obs_man.to_mean(model.hrm.likelihood_at(hrm_params, jnp.array([z_scalar])))
            assert jnp.all(rates > 0)

    def test_posterior_valid_vonmises(self) -> None:
        model, params = _make_population_code(8, jax.random.PRNGKey(42), n_regression_samples=10000)
        x = jax.random.poisson(jax.random.PRNGKey(0), 5.0 * jnp.ones(8))
        q_params = model.approximate_posterior_at(params, x)
        assert q_params.shape == (2,)
        _, kappa = model.pst_man.rep_man.split_mean_concentration(q_params)
        assert kappa >= 0

    def test_posterior_concentration_increases_with_spikes(self) -> None:
        model, params = _make_population_code(8, jax.random.PRNGKey(42), n_regression_samples=10000)
        vm = model.pst_man.rep_man
        _, kappa_low = vm.split_mean_concentration(
            model.approximate_posterior_at(params, jnp.ones(8))
        )
        _, kappa_high = vm.split_mean_concentration(
            model.approximate_posterior_at(params, 10 * jnp.ones(8))
        )
        assert kappa_high > kappa_low

    def test_sample_shape(self) -> None:
        model, params = _make_population_code(8, jax.random.PRNGKey(42))
        samples = model.sample(jax.random.PRNGKey(0), params, 5)
        assert samples.shape == (5, model.obs_man.data_dim + model.pst_man.data_dim)

    def test_reconstruction(self) -> None:
        model, params = _make_population_code(8, jax.random.PRNGKey(42))
        x = jax.random.poisson(jax.random.PRNGKey(0), 5.0 * jnp.ones(8)).astype(jnp.float32)
        recon = model.reconstruct(params, x)
        assert recon.shape == (8,)
        assert jnp.all(jnp.isfinite(recon))
        assert jnp.all(recon > 0)


class TestVonMisesPopulationCodeConjugation:
    """Test conjugation regression quality."""

    def test_regression_with_variation(self) -> None:
        n_neurons = 16
        model = VonMisesPopulationCode(hrm=PoissonVonMisesHarmonium(n_neurons, 1))
        key = jax.random.PRNGKey(42)

        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        gains = 1.0 + jax.random.uniform(key, (n_neurons,)) * 3.0

        int_col_1 = gains * jnp.cos(preferred)
        int_col_2 = gains * jnp.sin(preferred)
        int_params = jnp.stack([int_col_1, int_col_2], axis=1).ravel()
        hrm_params = model.hrm.join_coords(jnp.zeros(n_neurons), int_params, jnp.zeros(2))
        params = model.join_coords(jnp.zeros(model.rho_man.dim), hrm_params)

        key, reg_key = jax.random.split(key)
        rho, r_squared, _, _ = model.regress_conjugation_parameters(reg_key, params, 5000)
        assert jnp.linalg.norm(rho) > 0.01 or float(r_squared) > 0.3


class TestPoissonVonMisesHarmonium:
    """Test PoissonVonMisesHarmonium basics."""

    @pytest.mark.parametrize("n_neurons,n_latent", [(8, 2), (16, 4)])
    def test_dimensions(self, n_neurons: int, n_latent: int) -> None:
        model = PoissonVonMisesHarmonium(n_neurons, n_latent)
        assert model.n_neurons == n_neurons
        assert model.n_latent == n_latent
        assert model.obs_man.data_dim == n_neurons
        assert model.pst_man.data_dim == n_latent

    def test_initialize(self) -> None:
        model = PoissonVonMisesHarmonium(8, 2)
        params = model.initialize(jax.random.PRNGKey(42))
        assert jnp.all(jnp.isfinite(params))
