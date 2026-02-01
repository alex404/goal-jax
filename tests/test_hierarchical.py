"""Tests for Hierarchical Mixture Models.

Tests BinomialBernoulliMixture and related hierarchical variational models.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import (
    BinomialBernoulliMixture,
    binomial_bernoulli_mixture,
)

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-4
ATOL = 1e-6

# Test configuration
N_OBSERVABLE = 16
N_LATENT = 4
N_CLUSTERS = 3
N_TRIALS = 8


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def model() -> BinomialBernoulliMixture:
    """Create a test model."""
    return binomial_bernoulli_mixture(
        n_observable=N_OBSERVABLE,
        n_latent=N_LATENT,
        n_clusters=N_CLUSTERS,
        n_trials=N_TRIALS,
    )


@pytest.fixture
def params(model: BinomialBernoulliMixture, key: Array) -> Array:
    """Create initialized model parameters."""
    return model.initialize(key, location=0.0, shape=0.1)


class TestModelConstruction:
    """Tests for model construction."""

    def test_factory_function(self) -> None:
        """Test factory function creates correct model."""
        model = binomial_bernoulli_mixture(
            n_observable=10,
            n_latent=3,
            n_clusters=2,
            n_trials=16,
        )
        assert model.n_observable == 10
        assert model.n_latent == 3
        assert model.n_clusters == 2
        assert model.n_trials == 16

    def test_manifold_dimensions(self, model: BinomialBernoulliMixture) -> None:
        """Test manifold dimensions are correct."""
        assert model.obs_man.dim == N_OBSERVABLE
        assert model.bas_lat_man.dim == N_LATENT
        assert model.n_categories == N_CLUSTERS


class TestInitialization:
    """Tests for parameter initialization."""

    def test_initialize_valid_params(
        self, model: BinomialBernoulliMixture, key: Array
    ) -> None:
        """Test initialization produces valid parameters."""
        params = model.initialize(key, location=0.0, shape=0.1)
        expected_dim = model.rho_man.dim + model.hrm.dim
        assert params.shape == (expected_dim,)
        assert jnp.all(jnp.isfinite(params))

    def test_conjugation_params_zero(
        self, model: BinomialBernoulliMixture, params: Array
    ) -> None:
        """Test rho parameters are initialized to zero."""
        rho = model.conjugation_params(params)
        assert jnp.allclose(rho, 0.0)


class TestPosteriorComputation:
    """Tests for posterior computation."""

    def test_approximate_posterior_finite(
        self, model: BinomialBernoulliMixture, params: Array
    ) -> None:
        """Test approximate posterior produces finite values."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        q_params = model.approximate_posterior_at(params, x)

        assert jnp.all(jnp.isfinite(q_params))
        assert q_params.shape == (model.mix_man.dim,)

    def test_cluster_probs_sum_to_one(
        self, model: BinomialBernoulliMixture, params: Array
    ) -> None:
        """Test cluster probabilities sum to 1."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        q_params = model.approximate_posterior_at(params, x)
        probs = model.get_cluster_probs(q_params)

        assert jnp.allclose(jnp.sum(probs), 1.0)
        assert jnp.all(probs >= 0)
        assert jnp.all(probs <= 1)


class TestSampling:
    """Tests for sampling from the model."""

    def test_sample_from_posterior(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test posterior sampling produces correct shapes."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        q_params = model.approximate_posterior_at(params, x)

        n_samples = 10
        y_samples, k_samples = model.sample_from_posterior(key, q_params, n_samples)

        assert y_samples.shape == (n_samples, N_LATENT)
        assert k_samples.shape == (n_samples,)

    def test_sample_valid_clusters(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test sampled cluster indices are valid."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        q_params = model.approximate_posterior_at(params, x)

        _, k_samples = model.sample_from_posterior(key, q_params, 50)

        assert jnp.all(k_samples >= 0)
        assert jnp.all(k_samples < N_CLUSTERS)

    def test_sample_from_model(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test generative sampling produces correct shapes."""
        n_samples = 10
        x_samples, y_samples, k_samples = model.sample_from_model(key, params, n_samples)

        assert x_samples.shape == (n_samples, N_OBSERVABLE)
        assert y_samples.shape == (n_samples, N_LATENT)
        assert k_samples.shape == (n_samples,)


class TestELBO:
    """Tests for ELBO computation."""

    def test_elbo_finite(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test ELBO is finite."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        elbo = model.elbo_at(key, params, x, n_samples=5)
        assert jnp.isfinite(elbo)

    def test_elbo_components_consistent(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test ELBO = reconstruction - KL."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        elbo, recon, kl = model.elbo_components(key, params, x, n_samples=10)

        assert jnp.allclose(elbo, recon - kl, rtol=RTOL)

    def test_mean_elbo_finite(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test mean ELBO over batch is finite."""
        batch = jnp.ones((5, N_OBSERVABLE)) * N_TRIALS / 2
        mean_elbo = model.mean_elbo(key, params, batch, n_samples=5)
        assert jnp.isfinite(mean_elbo)


class TestReconstruction:
    """Tests for reconstruction."""

    def test_reconstruct_shape(
        self, model: BinomialBernoulliMixture, params: Array
    ) -> None:
        """Test reconstruction has correct shape."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        recon = model.reconstruct(params, x)
        assert recon.shape == (N_OBSERVABLE,)

    def test_reconstruct_valid_range(
        self, model: BinomialBernoulliMixture, params: Array
    ) -> None:
        """Test reconstructed values are in valid range."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        recon = model.reconstruct(params, x)

        assert jnp.all(recon >= 0)
        assert jnp.all(recon <= N_TRIALS)


class TestClusterAssignment:
    """Tests for cluster assignment."""

    def test_cluster_assignment_valid(
        self, model: BinomialBernoulliMixture, params: Array
    ) -> None:
        """Test cluster assignment is valid index."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        k = model.get_cluster_assignments(params, x)

        assert k >= 0
        assert k < N_CLUSTERS


class TestConjugationError:
    """Tests for conjugation error computation."""

    def test_conjugation_error_finite(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test conjugation error is finite and non-negative."""
        conj_err = model.conjugation_error(key, params, n_samples=20)

        assert jnp.isfinite(conj_err)
        assert conj_err >= 0

    def test_reduced_learning_signal_scalar(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test reduced learning signal returns a scalar."""
        p_params = model.prior_params(params)
        z_sample = model.pst_man.sample(key, p_params, 1)[0]

        f_tilde = model.reduced_learning_signal(params, z_sample)

        assert f_tilde.shape == ()
        assert jnp.isfinite(f_tilde)

    def test_conjugation_metrics(
        self, model: BinomialBernoulliMixture, params: Array, key: Array
    ) -> None:
        """Test conjugation_metrics returns valid values."""
        var_f, std_f, r_squared = model.conjugation_metrics(key, params, n_samples=30)

        assert jnp.isfinite(var_f)
        assert var_f >= 0
        assert jnp.isfinite(std_f)
        assert jnp.allclose(std_f, jnp.sqrt(var_f))
        assert jnp.isfinite(r_squared)
        assert r_squared <= 1.0


class TestFilters:
    """Tests for filter visualization."""

    def test_get_filters_shape(
        self, model: BinomialBernoulliMixture, params: Array
    ) -> None:
        """Test filter extraction has correct shape."""
        img_shape = (4, 4)  # 4x4 = 16 = N_OBSERVABLE
        filters = model.get_filters(params, img_shape)
        assert filters.shape == (N_LATENT, 4, 4)
