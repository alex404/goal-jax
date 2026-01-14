"""Tests for Bernoulli, Bernoullis, and DiagonalBoltzmann distributions.

Tests the mathematical properties of:
- Bernoulli (single binary variable, equivalent to Categorical(2))
- Bernoullis (product of n independent Bernoullis, core exponential family)
- DiagonalBoltzmann (mean-field wrapper with GeneralizedGaussian interface)
- BoltzmannEmbedding (embedding from DiagonalBoltzmann to full Boltzmann)
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from pytest import FixtureRequest

from goal.models import (
    Bernoulli,
    Bernoullis,
    Boltzmann,
    BoltzmannEmbedding,
    DiagonalBoltzmann,
)

jax.config.update("jax_platform_name", "cpu")

rtol, atol = 1e-4, 1e-6


# Bernoulli Tests


@pytest.fixture
def bernoulli() -> Bernoulli:
    """Create a Bernoulli distribution."""
    return Bernoulli()


def test_bernoulli_dim(bernoulli: Bernoulli):
    """Test Bernoulli has correct dimensions."""
    assert bernoulli.dim == 1
    assert bernoulli.data_dim == 1


def test_bernoulli_sufficient_statistic(bernoulli: Bernoulli):
    """Test sufficient statistic is identity."""
    assert jnp.allclose(bernoulli.sufficient_statistic(jnp.array(0.0)), jnp.array([0.0]))
    assert jnp.allclose(bernoulli.sufficient_statistic(jnp.array(1.0)), jnp.array([1.0]))


def test_bernoulli_log_partition_function(bernoulli: Bernoulli):
    """Test log partition function matches softplus."""
    theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    for theta in theta_values:
        params = jnp.array([theta])
        log_z = bernoulli.log_partition_function(params)
        expected = jax.nn.softplus(theta)
        assert jnp.allclose(log_z, expected, rtol=rtol, atol=atol)


def test_bernoulli_density_normalization(bernoulli: Bernoulli):
    """Test p(0) + p(1) = 1."""
    theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    for theta in theta_values:
        params = jnp.array([theta])
        log_z = bernoulli.log_partition_function(params)

        # p(x) = exp(θ*x - log Z)
        log_p0 = 0.0 * theta - log_z  # x=0
        log_p1 = 1.0 * theta - log_z  # x=1

        total = jnp.exp(log_p0) + jnp.exp(log_p1)
        assert jnp.allclose(total, 1.0, rtol=rtol, atol=atol)


def test_bernoulli_mean_natural_conversion(bernoulli: Bernoulli):
    """Test round-trip conversion between natural and mean parameters."""
    theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    for theta in theta_values:
        params = jnp.array([theta])
        means = bernoulli.to_mean(params)
        recovered = bernoulli.to_natural(means)
        assert jnp.allclose(params, recovered, rtol=rtol, atol=atol)


def test_bernoulli_mean_is_probability(bernoulli: Bernoulli):
    """Test that mean parameter equals P(x=1)."""
    theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    for theta in theta_values:
        params = jnp.array([theta])
        means = bernoulli.to_mean(params)
        expected_prob = jax.nn.sigmoid(theta)
        assert jnp.allclose(means[0], expected_prob, rtol=rtol, atol=atol)


def test_bernoulli_sampling(bernoulli: Bernoulli):
    """Test sampling produces binary values with correct distribution."""
    key = jax.random.PRNGKey(42)
    theta = 1.0  # P(x=1) ≈ 0.73
    params = jnp.array([theta])

    n_samples = 10000
    samples = bernoulli.sample(key, params, n_samples)

    assert samples.shape == (n_samples, 1)
    assert jnp.all((samples == 0) | (samples == 1))

    empirical_mean = jnp.mean(samples)
    expected_prob = jax.nn.sigmoid(theta)
    assert jnp.allclose(empirical_mean, expected_prob, rtol=0.05, atol=0.02)


def test_bernoulli_negative_entropy(bernoulli: Bernoulli):
    """Test negative entropy computation."""
    # At θ=0, p=0.5, entropy is maximal: -0.5*log(0.5) - 0.5*log(0.5) = log(2)
    params = jnp.array([0.0])
    means = bernoulli.to_mean(params)
    neg_ent = bernoulli.negative_entropy(means)
    expected = -jnp.log(2.0)  # H = log(2) at p=0.5, so -H = -log(2)
    assert jnp.allclose(neg_ent, expected, rtol=rtol, atol=atol)


# Bernoullis Tests


@pytest.fixture(params=[2, 3, 5])
def bernoullis(request: FixtureRequest) -> Bernoullis:
    """Create Bernoullis distributions of various sizes."""
    return Bernoullis(request.param)


def test_bernoullis_dim(bernoullis: Bernoullis):
    """Test Bernoullis has correct dimensions."""
    n = bernoullis.n_neurons
    assert bernoullis.dim == n
    assert bernoullis.data_dim == n


def test_bernoullis_log_partition_function(bernoullis: Bernoullis):
    """Test log partition function is sum of individual softplus values."""
    key = jax.random.PRNGKey(123)
    n = bernoullis.n_neurons
    params = jax.random.normal(key, (n,))

    log_z = bernoullis.log_partition_function(params)
    expected = jnp.sum(jax.nn.softplus(params))
    assert jnp.allclose(log_z, expected, rtol=rtol, atol=atol)


def test_bernoullis_sampling(bernoullis: Bernoullis):
    """Test sampling produces independent binary values."""
    key = jax.random.PRNGKey(456)
    n = bernoullis.n_neurons
    params = jnp.zeros(n)  # All p=0.5

    n_samples = 5000
    samples = bernoullis.sample(key, params, n_samples)

    assert samples.shape == (n_samples, n)
    assert jnp.all((samples == 0) | (samples == 1))

    # Each component should have empirical mean ≈ 0.5
    empirical_means = jnp.mean(samples, axis=0)
    assert jnp.allclose(empirical_means, 0.5 * jnp.ones(n), rtol=0.1, atol=0.05)


def test_bernoullis_mean_natural_conversion(bernoullis: Bernoullis):
    """Test round-trip conversion between natural and mean parameters."""
    key = jax.random.PRNGKey(789)
    n = bernoullis.n_neurons
    params = jax.random.normal(key, (n,))

    means = bernoullis.to_mean(params)
    recovered = bernoullis.to_natural(means)
    assert jnp.allclose(params, recovered, rtol=rtol, atol=atol)


# DiagonalBoltzmann Tests


@pytest.fixture(params=[2, 3, 5])
def diagonal_boltzmann(request: FixtureRequest) -> DiagonalBoltzmann:
    """Create DiagonalBoltzmann distributions of various sizes."""
    return DiagonalBoltzmann(request.param)


def test_diagonal_boltzmann_dim(diagonal_boltzmann: DiagonalBoltzmann):
    """Test DiagonalBoltzmann has correct dimensions."""
    n = diagonal_boltzmann.n_neurons
    assert diagonal_boltzmann.dim == n
    assert diagonal_boltzmann.data_dim == n


def test_diagonal_boltzmann_generalized_gaussian_interface(
    diagonal_boltzmann: DiagonalBoltzmann,
):
    """Test DiagonalBoltzmann provides GeneralizedGaussian interface."""
    n = diagonal_boltzmann.n_neurons

    # loc_man and shp_man should both be Bernoullis
    assert isinstance(diagonal_boltzmann.loc_man, Bernoullis)
    assert isinstance(diagonal_boltzmann.shp_man, Bernoullis)
    assert diagonal_boltzmann.loc_man.n_neurons == n
    assert diagonal_boltzmann.shp_man.n_neurons == n


def test_diagonal_boltzmann_split_join_round_trip(diagonal_boltzmann: DiagonalBoltzmann):
    """Test split/join location_precision round-trip."""
    key = jax.random.PRNGKey(999)
    n = diagonal_boltzmann.n_neurons
    params = jax.random.normal(key, (n,))

    loc, prec = diagonal_boltzmann.split_location_precision(params)
    recovered = diagonal_boltzmann.join_location_precision(loc, prec)

    assert jnp.allclose(params, recovered, rtol=rtol, atol=atol)


def test_diagonal_boltzmann_mean_split_join(diagonal_boltzmann: DiagonalBoltzmann):
    """Test split/join mean_second_moment (complete redundancy for binary)."""
    key = jax.random.PRNGKey(888)
    n = diagonal_boltzmann.n_neurons
    # Mean params are probabilities
    means = jax.random.uniform(key, (n,), minval=0.1, maxval=0.9)

    first, second = diagonal_boltzmann.split_mean_second_moment(means)

    # For independent binary, first and second moments are identical
    assert jnp.allclose(first, means, rtol=rtol, atol=atol)
    assert jnp.allclose(second, means, rtol=rtol, atol=atol)

    recovered = diagonal_boltzmann.join_mean_second_moment(first, second)
    assert jnp.allclose(recovered, means, rtol=rtol, atol=atol)


# BoltzmannEmbedding Tests


@pytest.fixture(params=[2, 3, 4])
def boltzmann_embedding(request: FixtureRequest) -> BoltzmannEmbedding:
    """Create BoltzmannEmbedding for various sizes."""
    n = request.param
    return BoltzmannEmbedding(DiagonalBoltzmann(n), Boltzmann(n))


def test_boltzmann_embedding_manifolds(boltzmann_embedding: BoltzmannEmbedding):
    """Test embedding has correct sub and ambient manifolds."""
    n = boltzmann_embedding.sub_man.n_neurons
    assert boltzmann_embedding.sub_man.dim == n
    assert boltzmann_embedding.amb_man.dim == n * (n + 1) // 2


def test_boltzmann_embedding_embed_zero_coupling(boltzmann_embedding: BoltzmannEmbedding):
    """Test that embedding creates zero off-diagonal coupling."""
    key = jax.random.PRNGKey(111)
    n = boltzmann_embedding.sub_man.n_neurons
    biases = jax.random.normal(key, (n,))

    # Embed biases into Boltzmann
    boltz_params = boltzmann_embedding.embed(biases)

    # Extract location and precision
    loc, prec = boltzmann_embedding.amb_man.split_location_precision(boltz_params)

    # Location should be absorbed into diagonal, but precision should show biases
    # For Boltzmann, the bias is absorbed into diagonal of coupling matrix
    # Check that we can recover approximately the original biases
    # by looking at the log partition derivative structure

    # The key check: off-diagonal elements should be zero
    full_matrix = boltzmann_embedding.amb_man.shp_man.to_matrix(boltz_params)
    n = full_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                assert jnp.allclose(full_matrix[i, j], 0.0, rtol=rtol, atol=atol)


def test_boltzmann_embedding_project_embed_round_trip(
    boltzmann_embedding: BoltzmannEmbedding,
):
    """Test project(embed(x)) recovers original (in mean coordinates)."""
    key = jax.random.PRNGKey(222)
    n = boltzmann_embedding.sub_man.n_neurons

    # Start with DiagonalBoltzmann mean parameters (probabilities)
    probs = jax.random.uniform(key, (n,), minval=0.1, maxval=0.9)

    # Convert to natural, embed, convert to mean, project
    diag_boltz = boltzmann_embedding.sub_man
    boltzmann = boltzmann_embedding.amb_man

    # Natural params for DiagonalBoltzmann
    natural_params = diag_boltz.to_natural(probs)

    # Embed into Boltzmann
    boltz_natural = boltzmann_embedding.embed(natural_params)

    # Convert to mean coordinates
    boltz_means = boltzmann.to_mean(boltz_natural)

    # Project back to DiagonalBoltzmann
    recovered_means = boltzmann_embedding.project(boltz_means)

    assert jnp.allclose(probs, recovered_means, rtol=rtol, atol=atol)


def test_boltzmann_embedding_translate(boltzmann_embedding: BoltzmannEmbedding):
    """Test translate correctly adds delta to location/bias."""
    key = jax.random.PRNGKey(333)
    n = boltzmann_embedding.sub_man.n_neurons

    key1, key2 = jax.random.split(key)
    biases1 = jax.random.normal(key1, (n,))
    delta = jax.random.normal(key2, (n,))

    # Embed first set of biases
    boltz_params = boltzmann_embedding.embed(biases1)

    # Translate by delta
    translated = boltzmann_embedding.translate(boltz_params, delta)

    # The result should be equivalent to embedding (biases1 + delta)
    expected = boltzmann_embedding.embed(biases1 + delta)

    assert jnp.allclose(translated, expected, rtol=rtol, atol=atol)


def test_boltzmann_loc_man_is_bernoullis():
    """Test that Boltzmann.loc_man returns Bernoullis."""
    n = 4
    boltz = Boltzmann(n)
    loc_man = boltz.loc_man

    assert isinstance(loc_man, Bernoullis)
    assert loc_man.n_neurons == n
