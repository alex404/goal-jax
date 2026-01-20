"""Tests for univariate exponential family models.

Tests density integration, sufficient statistic convergence, and parameter recovery.
Includes tests for Bernoulli, Binomial, and related Boltzmann structures.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from pytest import FixtureRequest

from goal.geometry import Differentiable, PositiveDefinite
from goal.geometry.exponential_family.base import Analytic
from goal.models import (
    Bernoulli,
    Bernoullis,
    Binomial,
    Binomials,
    Boltzmann,
    BoltzmannEmbedding,
    Categorical,
    CoMPoisson,
    DiagonalBoltzmann,
    Normal,
    Poisson,
    VonMises,
)

jax.config.update("jax_platform_name", "cpu")

# Test parameters
n_trials = 5
sample_sizes = [100, 1000]

rtol, atol = 1e-4, 1e-6

# Model groups: all differentiable models
all_models = [
    ("categorical", Categorical(n_categories=5)),
    ("poisson", Poisson()),
    ("normal", Normal(1, PositiveDefinite())),
    ("von_mises", VonMises()),
    ("com_poisson", CoMPoisson()),
    ("bernoulli", Bernoulli()),
    ("binomial", Binomial(n_trials=10)),
]

# Analytic models only (support to_natural, relative_entropy)
analytic_models = [
    ("categorical", Categorical(n_categories=5)),
    ("poisson", Poisson()),
    ("normal", Normal(1, PositiveDefinite())),
    ("bernoulli", Bernoulli()),
    ("binomial", Binomial(n_trials=10)),
]


def integrate_density(
    model: Differentiable, params: Array, xs: Array, dx: float = 1.0
) -> Array:
    """Integrate density over given points."""
    densities = jax.vmap(model.density, in_axes=(None, 0))(params, xs)
    return jnp.sum(densities) * dx


@pytest.fixture(params=all_models, ids=lambda x: x[0])
def model(request: pytest.FixtureRequest) -> Differentiable:
    """All differentiable models."""
    return request.param[1]


@pytest.fixture(params=analytic_models, ids=lambda x: x[0])
def analytic_model(request: pytest.FixtureRequest) -> Analytic:
    """Analytic models only."""
    return request.param[1]


def test_density_integrates_to_one(model: Differentiable):
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
            means = model.to_mean(params)
            mean, var = model.split_mean_covariance(means)
            std = jnp.sqrt(var)
            xs = jnp.linspace(mean - 6 * std, mean + 6 * std, 1000).ravel()
            total = integrate_density(model, params, xs, dx=12 * std.item() / 1000)
        elif isinstance(model, Bernoulli):
            xs = jnp.array([0.0, 1.0])
            total = integrate_density(model, params, xs)
        elif isinstance(model, Binomial):
            xs = jnp.arange(model.n_trials + 1, dtype=jnp.float32)
            total = integrate_density(model, params, xs)
        else:
            raise TypeError(f"Unknown model type: {type(model)}")

        results.append(float(total))

    mean_total = jnp.mean(jnp.array(results))
    assert jnp.allclose(mean_total, 1.0, rtol=1e-2)


def test_sufficient_statistic_convergence(model: Differentiable):
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


def test_natural_parameters_valid(model: Differentiable):
    """Test that initialized parameters pass validity check."""
    key = jax.random.PRNGKey(789)

    for i in range(n_trials):
        params = model.initialize(jax.random.fold_in(key, i))
        assert model.check_natural_parameters(params)


def test_parameter_recovery(analytic_model: Analytic):
    """Test natural -> mean -> natural round trip."""
    key = jax.random.PRNGKey(123)

    for i in range(n_trials):
        params = analytic_model.initialize(jax.random.fold_in(key, i))
        means = analytic_model.to_mean(params)
        recovered = analytic_model.to_natural(means)
        error = float(jnp.mean((recovered - params) ** 2))
        assert error < 1e-5, f"Parameter recovery error: {error}"


def test_relative_entropy_with_self(analytic_model: Analytic):
    """Test that relative entropy with itself is zero."""
    key = jax.random.PRNGKey(456)

    for i in range(n_trials):
        params = analytic_model.initialize(jax.random.fold_in(key, i))
        means = analytic_model.to_mean(params)
        re = analytic_model.relative_entropy(means, params)
        assert re >= -1e-6, f"Negative relative entropy: {re}"
        assert jnp.abs(re) < 1e-5, f"Self relative entropy not zero: {re}"


# =============================================================================
# Bernoulli-specific Tests
# =============================================================================


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
    assert jnp.allclose(
        bernoulli.sufficient_statistic(jnp.array(0.0)), jnp.array([0.0])
    )
    assert jnp.allclose(
        bernoulli.sufficient_statistic(jnp.array(1.0)), jnp.array([1.0])
    )


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

        # p(x) = exp(theta*x - log Z)
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
    theta = 1.0  # P(x=1) ~ 0.73
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
    # At theta=0, p=0.5, entropy is maximal: -0.5*log(0.5) - 0.5*log(0.5) = log(2)
    params = jnp.array([0.0])
    means = bernoulli.to_mean(params)
    neg_ent = bernoulli.negative_entropy(means)
    expected = -jnp.log(2.0)  # H = log(2) at p=0.5, so -H = -log(2)
    assert jnp.allclose(neg_ent, expected, rtol=rtol, atol=atol)


# =============================================================================
# Binomial-specific Tests
# =============================================================================


@pytest.fixture(params=[1, 5, 10, 255])
def binomial(request: FixtureRequest) -> Binomial:
    """Create Binomial distributions with various n_trials."""
    return Binomial(n_trials=request.param)


def test_binomial_dim(binomial: Binomial):
    """Test Binomial has correct dimensions."""
    assert binomial.dim == 1
    assert binomial.data_dim == 1


def test_binomial_sufficient_statistic(binomial: Binomial):
    """Test sufficient statistic is identity (count)."""
    for x in [0, 1, binomial.n_trials // 2, binomial.n_trials]:
        assert jnp.allclose(
            binomial.sufficient_statistic(jnp.array(float(x))), jnp.array([float(x)])
        )


def test_binomial_log_partition_function(binomial: Binomial):
    """Test log partition function is n * softplus."""
    theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    for theta in theta_values:
        params = jnp.array([theta])
        log_z = binomial.log_partition_function(params)
        expected = binomial.n_trials * jax.nn.softplus(theta)
        assert jnp.allclose(log_z, expected, rtol=rtol, atol=atol)


def test_binomial_density_normalization(binomial: Binomial):
    """Test that pmf sums to 1."""
    theta_values = jnp.array([-1.0, 0.0, 1.0])
    for theta in theta_values:
        params = jnp.array([theta])
        xs = jnp.arange(binomial.n_trials + 1, dtype=jnp.float32)
        densities = jax.vmap(binomial.density, in_axes=(None, 0))(params, xs)
        total = jnp.sum(densities)
        assert jnp.allclose(total, 1.0, rtol=1e-3, atol=1e-6)


def test_binomial_mean_is_np(binomial: Binomial):
    """Test that mean parameter equals n*p."""
    theta_values = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    for theta in theta_values:
        params = jnp.array([theta])
        means = binomial.to_mean(params)
        p = jax.nn.sigmoid(theta)
        expected_mean = binomial.n_trials * p
        assert jnp.allclose(means[0], expected_mean, rtol=rtol, atol=atol)


def test_binomial_sampling(binomial: Binomial):
    """Test sampling produces counts with correct distribution."""
    key = jax.random.PRNGKey(42)
    theta = 0.0  # p = 0.5
    params = jnp.array([theta])

    n_samples = 5000
    samples = binomial.sample(key, params, n_samples)

    assert samples.shape == (n_samples, 1)
    assert jnp.all(samples >= 0)
    assert jnp.all(samples <= binomial.n_trials)

    empirical_mean = jnp.mean(samples)
    expected_mean = binomial.n_trials * 0.5
    assert jnp.allclose(empirical_mean, expected_mean, rtol=0.1, atol=1.0)


def test_binomial_reduces_to_bernoulli():
    """Test that Binomial(1) is equivalent to Bernoulli."""
    binomial = Binomial(n_trials=1)
    bernoulli = Bernoulli()

    theta_values = jnp.array([-2.0, 0.0, 2.0])
    for theta in theta_values:
        params = jnp.array([theta])

        # Log partition should match
        assert jnp.allclose(
            binomial.log_partition_function(params),
            bernoulli.log_partition_function(params),
            rtol=rtol,
            atol=atol,
        )

        # Means should match
        assert jnp.allclose(
            binomial.to_mean(params),
            bernoulli.to_mean(params),
            rtol=rtol,
            atol=atol,
        )


# =============================================================================
# Bernoullis Tests
# =============================================================================


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

    # Each component should have empirical mean ~ 0.5
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


# =============================================================================
# Binomials Tests
# =============================================================================


@pytest.fixture(params=[2, 3, 5])
def binomials(request: FixtureRequest) -> Binomials:
    """Create Binomials distributions of various sizes."""
    return Binomials(request.param, n_trials=10)


def test_binomials_dim(binomials: Binomials):
    """Test Binomials has correct dimensions."""
    n = binomials.n_neurons
    assert binomials.dim == n
    assert binomials.data_dim == n


def test_binomials_log_partition_function(binomials: Binomials):
    """Test log partition function is sum of individual n*softplus values."""
    key = jax.random.PRNGKey(123)
    n = binomials.n_neurons
    n_trials = binomials.n_trials
    params = jax.random.normal(key, (n,))

    log_z = binomials.log_partition_function(params)
    expected = jnp.sum(n_trials * jax.nn.softplus(params))
    assert jnp.allclose(log_z, expected, rtol=rtol, atol=atol)


def test_binomials_sampling(binomials: Binomials):
    """Test sampling produces valid counts."""
    key = jax.random.PRNGKey(456)
    n = binomials.n_neurons
    params = jnp.zeros(n)  # p=0.5 for all

    n_samples = 2000
    samples = binomials.sample(key, params, n_samples)

    assert samples.shape == (n_samples, n)
    assert jnp.all(samples >= 0)
    assert jnp.all(samples <= binomials.n_trials)


# =============================================================================
# DiagonalBoltzmann Tests
# =============================================================================


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


def test_diagonal_boltzmann_split_join_round_trip(
    diagonal_boltzmann: DiagonalBoltzmann,
):
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


# =============================================================================
# BoltzmannEmbedding Tests
# =============================================================================


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


def test_boltzmann_embedding_embed_zero_coupling(
    boltzmann_embedding: BoltzmannEmbedding,
):
    """Test that embedding creates zero off-diagonal coupling."""
    key = jax.random.PRNGKey(111)
    n = boltzmann_embedding.sub_man.n_neurons
    biases = jax.random.normal(key, (n,))

    # Embed biases into Boltzmann
    boltz_params = boltzmann_embedding.embed(biases)

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
