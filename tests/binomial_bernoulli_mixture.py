"""Tests for BinomialBernoulliMixture model.

Tests the variational hierarchical mixture framework with Binomial observables
and Bernoulli latents.
"""

import jax
import jax.numpy as jnp
import pytest

from goal.models import (
    BinomialBernoulliMixture,
    binomial_bernoulli_mixture,
)


# Test configuration
N_OBSERVABLE = 16
N_LATENT = 4
N_CLUSTERS = 3
N_TRIALS = 8


@pytest.fixture
def model() -> BinomialBernoulliMixture:
    """Create a small test model."""
    return binomial_bernoulli_mixture(
        n_observable=N_OBSERVABLE,
        n_latent=N_LATENT,
        n_clusters=N_CLUSTERS,
        n_trials=N_TRIALS,
    )


@pytest.fixture
def initialized_params(model: BinomialBernoulliMixture):
    """Create initialized model parameters."""
    key = jax.random.PRNGKey(42)
    return model.initialize(key, location=0.0, shape=0.1)


class TestModelConstruction:
    """Tests for model construction and properties."""

    def test_factory_function(self):
        """Test that factory function creates correct model."""
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

    def test_manifold_dimensions(self, model: BinomialBernoulliMixture):
        """Test that manifold dimensions are correct."""
        # Observable: n_observable
        assert model.obs_man.dim == N_OBSERVABLE

        # Base latent: n_latent
        assert model.bas_lat_man.dim == N_LATENT

        # Mixture: obs_bias + int_mat + cat_params
        # obs_bias: n_latent
        # int_mat: n_latent * (n_clusters - 1)
        # cat_params: n_clusters - 1
        mix_dim = N_LATENT + N_LATENT * (N_CLUSTERS - 1) + (N_CLUSTERS - 1)
        assert model.mix_man.dim == mix_dim

    def test_n_categories(self, model: BinomialBernoulliMixture):
        """Test that n_categories property works."""
        assert model.n_categories == N_CLUSTERS


class TestInitialization:
    """Tests for parameter initialization."""

    def test_initialize_creates_valid_params(self, model: BinomialBernoulliMixture):
        """Test that initialization produces valid parameters."""
        key = jax.random.PRNGKey(0)
        params = model.initialize(key, location=0.0, shape=0.1)

        # Check that params have correct dimension
        expected_dim = model.rho_man.dim + model.hrm.dim
        assert params.shape == (expected_dim,)

        # Check that params are finite
        assert jnp.all(jnp.isfinite(params))

    def test_conjugation_params_initially_zero(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that rho parameters are initialized to zero."""
        rho = model.conjugation_params(initialized_params)
        assert jnp.allclose(rho, 0.0)


class TestPosteriorComputation:
    """Tests for posterior computation."""

    def test_approximate_posterior_finite(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that approximate posterior produces finite values."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2  # Mid-range counts
        q_params = model.approximate_posterior_at(initialized_params, x)

        assert jnp.all(jnp.isfinite(q_params))
        assert q_params.shape == (model.mix_man.dim,)

    def test_cluster_probs_sum_to_one(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that cluster probabilities sum to 1."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        q_params = model.approximate_posterior_at(initialized_params, x)
        probs = model.get_cluster_probs(q_params)

        assert jnp.allclose(jnp.sum(probs), 1.0)
        assert jnp.all(probs >= 0)
        assert jnp.all(probs <= 1)


class TestSampling:
    """Tests for sampling from the model."""

    def test_sample_from_posterior_shapes(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that posterior sampling produces correct shapes."""
        key = jax.random.PRNGKey(1)
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        q_params = model.approximate_posterior_at(initialized_params, x)

        n_samples = 10
        y_samples, k_samples = model.sample_from_posterior(key, q_params, n_samples)

        assert y_samples.shape == (n_samples, N_LATENT)
        assert k_samples.shape == (n_samples,)

    def test_sample_from_posterior_valid_clusters(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that sampled cluster indices are valid."""
        key = jax.random.PRNGKey(2)
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        q_params = model.approximate_posterior_at(initialized_params, x)

        _, k_samples = model.sample_from_posterior(key, q_params, 100)

        assert jnp.all(k_samples >= 0)
        assert jnp.all(k_samples < N_CLUSTERS)

    def test_sample_from_model_shapes(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that generative sampling produces correct shapes."""
        key = jax.random.PRNGKey(3)
        n_samples = 10

        x_samples, y_samples, k_samples = model.sample_from_model(
            key, initialized_params, n_samples
        )

        assert x_samples.shape == (n_samples, N_OBSERVABLE)
        assert y_samples.shape == (n_samples, N_LATENT)
        assert k_samples.shape == (n_samples,)


class TestELBO:
    """Tests for ELBO computation."""

    def test_elbo_finite(self, model: BinomialBernoulliMixture, initialized_params):
        """Test that ELBO computation produces finite values."""
        key = jax.random.PRNGKey(4)
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2

        elbo = model.elbo_at(key, initialized_params, x, n_samples=5)
        assert jnp.isfinite(elbo)

    def test_elbo_components_consistent(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that ELBO components are consistent."""
        key = jax.random.PRNGKey(5)
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2

        elbo, recon, kl = model.elbo_components(
            key, initialized_params, x, n_samples=10
        )

        # ELBO should equal recon - kl
        assert jnp.allclose(elbo, recon - kl, rtol=1e-4)

    def test_mean_elbo_finite(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that mean ELBO over a batch is finite."""
        key = jax.random.PRNGKey(6)
        batch = jnp.ones((5, N_OBSERVABLE)) * N_TRIALS / 2

        mean_elbo = model.mean_elbo(key, initialized_params, batch, n_samples=5)
        assert jnp.isfinite(mean_elbo)


class TestReconstruction:
    """Tests for reconstruction methods."""

    def test_reconstruct_shape(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that reconstruction has correct shape."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        recon = model.reconstruct(initialized_params, x)

        assert recon.shape == (N_OBSERVABLE,)

    def test_reconstruct_in_valid_range(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that reconstructed values are in valid range."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        recon = model.reconstruct(initialized_params, x)

        # Reconstructed counts should be in [0, n_trials]
        assert jnp.all(recon >= 0)
        assert jnp.all(recon <= N_TRIALS)


class TestClusterAssignment:
    """Tests for cluster assignment."""

    def test_get_cluster_assignments_valid(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that cluster assignment is valid."""
        x = jnp.ones(N_OBSERVABLE) * N_TRIALS / 2
        k = model.get_cluster_assignments(initialized_params, x)

        assert k >= 0
        assert k < N_CLUSTERS


class TestFilters:
    """Tests for filter visualization."""

    def test_get_filters_shape(
        self, model: BinomialBernoulliMixture, initialized_params
    ):
        """Test that filter extraction has correct shape."""
        img_shape = (4, 4)  # 4x4 = 16 = N_OBSERVABLE
        filters = model.get_filters(initialized_params, img_shape)

        assert filters.shape == (N_LATENT, 4, 4)
