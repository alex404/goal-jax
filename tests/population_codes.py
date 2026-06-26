"""Tests for models/harmonium/population_codes.py.

Covers PoissonVonMisesHarmonium / VonMisesPopulationCode and the Boltzmann-
observable / Gaussian-latent BoltzmannPopulationCode. Verifies dimensions,
tuning curves, posterior computation, sampling, conjugation regression, and
basic harmonium initialization.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry.exponential_family.variational import (
    conjugation_metrics,
    reconstruct,
    regress_conjugation_parameters,
)
from goal.models import (
    BoltzmannPopulationCode,
    PoissonVonMisesHarmonium,
    VonMisesPopulationCode,
    chordal_boltzmann_population_code,
    diagonal_boltzmann_population_code,
    full_normal,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-6


def _make_population_code(
    n_neurons: int, key: Array, *, n_regression_samples: int = 1000
) -> tuple[VonMisesPopulationCode, Array]:
    """Create a VonMisesPopulationCode with uniform preferred directions."""
    model = VonMisesPopulationCode(_gen_hrm=PoissonVonMisesHarmonium(n_neurons, 1))
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
        model = VonMisesPopulationCode(_gen_hrm=hrm)
        assert model.obs_man.dim == n
        assert model.obs_man.data_dim == n
        assert model.pst_man.dim == 2
        assert model.pst_man.data_dim == 1
        assert model.gen_hrm.int_man.dim == n * 2
        assert model.dim == 2 + n + n * 2 + 2

    @pytest.mark.parametrize("n", [4, 8])
    def test_tuning_curve_peaks_at_preferred(self, n: int) -> None:
        """Each neuron's firing rate is higher at its preferred direction."""
        model, params = _make_population_code(n, jax.random.PRNGKey(42))
        _, lkl_params, _ = model.split_coords(params)
        _, int_params = model.gen_hrm.lkl_fun_man.split_coords(lkl_params)
        int_matrix = int_params.reshape(n, 2)
        preferred = jnp.arctan2(int_matrix[:, 1], int_matrix[:, 0])

        for i in range(n):
            pref = jnp.array([preferred[i]])
            opp = jnp.array([preferred[i] + jnp.pi])
            assert model.likelihood_at(params, pref)[i] > model.likelihood_at(
                params, opp
            )[i]

    def test_firing_rates_positive(self) -> None:
        model, params = _make_population_code(8, jax.random.PRNGKey(42))
        for z_scalar in jnp.linspace(0, 2 * jnp.pi, 10):
            rates = model.obs_man.to_mean(model.likelihood_at(params, jnp.array([z_scalar])))
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
        recon = reconstruct(model, params, x)
        assert recon.shape == (8,)
        assert jnp.all(jnp.isfinite(recon))
        assert jnp.all(recon > 0)


class TestVonMisesPopulationCodeConjugation:
    """Test conjugation regression quality."""

    def test_regression_with_variation(self) -> None:
        n_neurons = 16
        model = VonMisesPopulationCode(_gen_hrm=PoissonVonMisesHarmonium(n_neurons, 1))
        key = jax.random.PRNGKey(42)

        preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
        gains = 1.0 + jax.random.uniform(key, (n_neurons,)) * 3.0

        int_col_1 = gains * jnp.cos(preferred)
        int_col_2 = gains * jnp.sin(preferred)
        int_params = jnp.stack([int_col_1, int_col_2], axis=1).ravel()
        lkl_params = model.gen_hrm.lkl_fun_man.join_coords(jnp.zeros(n_neurons), int_params)
        params = model.join_coords(jnp.zeros(2), lkl_params, jnp.zeros(model.cnj_man.dim))

        key, reg_key = jax.random.split(key)
        rho, r_squared, _, _ = regress_conjugation_parameters(
            model, reg_key, params, 5000
        )
        assert jnp.linalg.norm(rho) > 0.01 or float(r_squared) > 0.3


def _make_boltzmann_pc(
    kind: str, n: int, d: int, key: Array, *, n_reg: int = 1500
) -> tuple[BoltzmannPopulationCode[Any], Array]:
    """Build a chain-chordal or diagonal Boltzmann PPC with rho seeded by regression."""
    edges = [(i, i + 1) for i in range(n - 1)]
    model: BoltzmannPopulationCode[Any]
    if kind == "chordal":
        model = chordal_boltzmann_population_code(n, edges, d)
    else:
        model = diagonal_boltzmann_population_code(n, d)
    params = model.initialize(key, location=0.0, shape=0.3)
    rho, _, _, _ = regress_conjugation_parameters(
        model, jax.random.fold_in(key, 1), params, n_reg
    )
    prior_p, lkl_p, _ = model.split_coords(params)
    return model, model.join_coords(prior_p, lkl_p, rho)


class TestBoltzmannPopulationCode:
    """Test the Boltzmann-observable / Gaussian-latent population code."""

    @pytest.mark.parametrize("kind", ["chordal", "diagonal"])
    def test_dimensions(self, kind: str) -> None:
        model, _ = _make_boltzmann_pc(kind, 6, 2, jax.random.PRNGKey(0))
        # the interaction carries the FULL Gaussian sufficient statistic (z, zz^T)
        assert model.gen_hrm.int_man.matrix_shape == (model.obs_man.dim, full_normal(2).dim)
        assert model.cnj_man.dim == model.lat_man.dim
        assert model.n_neurons == 6
        assert model.n_latent == 2

    def test_conjugation_residual_matches_recompute(self) -> None:
        """conjugation_residual equals an independent recompute from the public API."""
        model, params = _make_boltzmann_pc("chordal", 6, 2, jax.random.PRNGKey(1))
        z = 0.5 * jax.random.normal(jax.random.PRNGKey(2), (2,))
        r_model = model.conjugation_residual(params, z)
        _, lkl, _ = model.split_coords(params)
        s_z = model.lat_man.sufficient_statistic(z)
        psi_z = model.obs_man.log_partition_function(model.gen_hrm.lkl_fun_man(lkl, s_z))
        obs_p, _ = model.gen_hrm.lkl_fun_man.split_coords(lkl)
        psi_b = model.obs_man.log_partition_function(obs_p)
        r_direct = jnp.dot(model.conjugation_parameters(params), s_z) - psi_z + psi_b
        assert jnp.allclose(r_model, r_direct, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("kind", ["chordal", "diagonal"])
    def test_conjugation_regression_high_r2(self, kind: str) -> None:
        """The quadratic-in-z tuning makes the PPC nearly conjugate after regression."""
        model, params = _make_boltzmann_pc(kind, 6, 2, jax.random.PRNGKey(3), n_reg=4000)
        _, _, r2 = conjugation_metrics(model, jax.random.PRNGKey(4), params, 1000)
        assert float(r2) > 0.85

    def test_posterior_precision_is_data_dependent(self) -> None:
        """The second-order interaction gives the posterior an x-dependent precision."""
        model, params = _make_boltzmann_pc("chordal", 6, 2, jax.random.PRNGKey(5))
        q0 = model.approximate_posterior_at(params, jnp.zeros(6))
        q1 = model.approximate_posterior_at(params, jnp.ones(6))
        _, prec0 = model.lat_man.split_location_precision(q0)
        _, prec1 = model.lat_man.split_location_precision(q1)
        assert jnp.all(jnp.isfinite(q0))
        assert not jnp.allclose(prec0, prec1)

    def test_sample_observable_binary(self) -> None:
        model, params = _make_boltzmann_pc("chordal", 6, 2, jax.random.PRNGKey(6))
        s = model.sample(jax.random.PRNGKey(7), params, 16)
        assert s.shape == (16, model.obs_man.data_dim + model.lat_man.data_dim)
        x = s[:, : model.obs_man.data_dim]
        assert jnp.all((x == 0) | (x == 1))

    def test_elbo_finite_and_reconstruct(self) -> None:
        model, params = _make_boltzmann_pc("chordal", 6, 2, jax.random.PRNGKey(8))
        data = model.sample(jax.random.PRNGKey(9), params, 32)[:, : model.obs_man.data_dim]
        elbo = model.mean_elbo(jax.random.PRNGKey(10), params, data, 8)
        assert jnp.isfinite(elbo)
        recon = reconstruct(model, params, data[0])
        assert recon.shape == (model.obs_man.dim,)
        assert jnp.all(jnp.isfinite(recon))


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
