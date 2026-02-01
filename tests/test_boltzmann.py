"""Tests for Boltzmann distributions.

Tests Boltzmann, DiagonalBoltzmann, and BoltzmannEmbedding.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Symmetric
from goal.models import Bernoullis, Boltzmann, BoltzmannEmbedding, DiagonalBoltzmann

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-5
ATOL = 1e-7


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestBoltzmannBasics:
    """Test basic Boltzmann properties."""

    @pytest.fixture(params=[3, 4, 5])
    def model(self, request: pytest.FixtureRequest) -> Boltzmann:
        """Create Boltzmann model."""
        return Boltzmann(request.param)

    def test_dimensions(self, model: Boltzmann) -> None:
        """Test dimensions are correct."""
        n = model.n_neurons
        # Parameters are upper triangular: n(n+1)/2
        assert model.dim == n * (n + 1) // 2
        assert model.data_dim == n

    def test_sufficient_statistic(self, model: Boltzmann, key: Array) -> None:
        """Test sufficient statistic is upper triangular outer product."""
        state = jax.random.bernoulli(key, 0.5, (model.n_neurons,))
        suff_stat = model.sufficient_statistic(state)
        expected = Symmetric().from_matrix(jnp.outer(state, state))

        assert jnp.allclose(suff_stat, expected, rtol=RTOL, atol=ATOL)


class TestBoltzmannDensity:
    """Test Boltzmann density properties."""

    @pytest.fixture(params=[3, 4])
    def model(self, request: pytest.FixtureRequest) -> Boltzmann:
        """Create Boltzmann model."""
        return Boltzmann(request.param)

    @pytest.fixture
    def params(self, model: Boltzmann, key: Array) -> Array:
        """Generate random parameters."""
        return jax.random.uniform(key, (model.dim,), minval=-2.0, maxval=2.0)

    def test_log_partition_matches_sum(self, model: Boltzmann, params: Array) -> None:
        """Test log partition equals log sum over all states."""
        log_z = model.log_partition_function(params)

        def energy(state: Array) -> Array:
            return jnp.dot(params, model.sufficient_statistic(state))

        energies = jax.vmap(energy)(model.states)
        log_z_direct = jax.scipy.special.logsumexp(energies)

        assert jnp.allclose(log_z, log_z_direct, rtol=RTOL, atol=ATOL)

    def test_density_normalizes(self, model: Boltzmann, params: Array) -> None:
        """Test densities sum to 1 over all binary states."""
        densities = jax.vmap(lambda s: model.density(params, s))(model.states)
        total_prob = jnp.sum(densities)

        assert jnp.allclose(total_prob, 1.0, rtol=RTOL, atol=ATOL)


class TestBoltzmannConversions:
    """Test Boltzmann parameter conversions."""

    @pytest.fixture(params=[3, 4])
    def model(self, request: pytest.FixtureRequest) -> Boltzmann:
        """Create Boltzmann model."""
        return Boltzmann(request.param)

    @pytest.fixture
    def params(self, model: Boltzmann, key: Array) -> Array:
        """Generate random parameters."""
        return jax.random.uniform(key, (model.dim,), minval=-2.0, maxval=2.0)

    def test_split_join_round_trip(self, model: Boltzmann, params: Array) -> None:
        """Test split/join location_precision is invertible."""
        loc, prec = model.split_location_precision(params)
        recovered = model.join_location_precision(loc, prec)

        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_loc_man_is_bernoullis(self, model: Boltzmann) -> None:
        """Test that Boltzmann.loc_man returns Bernoullis."""
        loc_man = model.loc_man
        assert isinstance(loc_man, Bernoullis)
        assert loc_man.n_neurons == model.n_neurons


class TestBoltzmannConditional:
    """Test Boltzmann conditional probability computation."""

    @pytest.fixture
    def model(self) -> Boltzmann:
        """Create Boltzmann model."""
        return Boltzmann(4)

    @pytest.fixture
    def params(self, model: Boltzmann, key: Array) -> Array:
        """Generate random parameters."""
        return jax.random.uniform(key, (model.dim,), minval=-2.0, maxval=2.0)

    def test_unit_conditional_prob(self, model: Boltzmann, params: Array, key: Array) -> None:
        """Test unit_conditional_prob matches direct computation."""
        coupling = model.shp_man

        for _ in range(3):
            key, state_key, unit_key = jax.random.split(key, 3)
            state = jax.random.bernoulli(state_key, 0.5, (model.n_neurons,)).astype(
                jnp.float32
            )
            unit_idx = jax.random.randint(unit_key, (), 0, model.n_neurons)

            # Compute using unit_conditional_prob
            prob_fast = coupling.unit_conditional_prob(state, unit_idx, params)

            # Compute directly using energy difference
            state_0 = state.at[unit_idx].set(0.0)
            state_1 = state.at[unit_idx].set(1.0)

            energy_0 = jnp.dot(params, coupling.sufficient_statistic(state_0))
            energy_1 = jnp.dot(params, coupling.sufficient_statistic(state_1))

            prob_direct = jax.nn.sigmoid(energy_1 - energy_0)

            assert jnp.allclose(prob_fast, prob_direct, rtol=RTOL, atol=ATOL)


class TestDiagonalBoltzmannBasics:
    """Test DiagonalBoltzmann (independent binary units)."""

    @pytest.fixture(params=[2, 3, 5])
    def model(self, request: pytest.FixtureRequest) -> DiagonalBoltzmann:
        """Create DiagonalBoltzmann model."""
        return DiagonalBoltzmann(request.param)

    def test_dimensions(self, model: DiagonalBoltzmann) -> None:
        """Test dimensions are correct."""
        n = model.n_neurons
        # Only n parameters (no couplings)
        assert model.dim == n
        assert model.data_dim == n

    def test_generalized_gaussian_interface(self, model: DiagonalBoltzmann) -> None:
        """Test DiagonalBoltzmann provides GeneralizedGaussian interface."""
        n = model.n_neurons

        assert isinstance(model.loc_man, Bernoullis)
        assert isinstance(model.shp_man, Bernoullis)
        assert model.loc_man.n_neurons == n
        assert model.shp_man.n_neurons == n


class TestDiagonalBoltzmannConversions:
    """Test DiagonalBoltzmann parameter conversions."""

    @pytest.fixture(params=[2, 3, 5])
    def model(self, request: pytest.FixtureRequest) -> DiagonalBoltzmann:
        """Create DiagonalBoltzmann model."""
        return DiagonalBoltzmann(request.param)

    def test_split_join_location_precision(
        self, model: DiagonalBoltzmann, key: Array
    ) -> None:
        """Test split/join location_precision round-trip."""
        n = model.n_neurons
        params = jax.random.normal(key, (n,))

        loc, prec = model.split_location_precision(params)
        recovered = model.join_location_precision(loc, prec)

        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_split_join_mean_second_moment(
        self, model: DiagonalBoltzmann, key: Array
    ) -> None:
        """Test split/join mean_second_moment (for independent binary, identical)."""
        n = model.n_neurons
        # Mean params are probabilities
        means = jax.random.uniform(key, (n,), minval=0.1, maxval=0.9)

        first, second = model.split_mean_second_moment(means)

        # For independent binary, first and second moments are identical
        assert jnp.allclose(first, means, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(second, means, rtol=RTOL, atol=ATOL)

        recovered = model.join_mean_second_moment(first, second)
        assert jnp.allclose(recovered, means, rtol=RTOL, atol=ATOL)


class TestBoltzmannEmbeddingBasics:
    """Test BoltzmannEmbedding (DiagonalBoltzmann -> Boltzmann)."""

    @pytest.fixture(params=[2, 3, 4])
    def embedding(self, request: pytest.FixtureRequest) -> BoltzmannEmbedding:
        """Create BoltzmannEmbedding."""
        n = request.param
        return BoltzmannEmbedding(DiagonalBoltzmann(n), Boltzmann(n))

    def test_manifolds(self, embedding: BoltzmannEmbedding) -> None:
        """Test embedding has correct sub and ambient manifolds."""
        n = embedding.sub_man.n_neurons
        assert embedding.sub_man.dim == n
        assert embedding.amb_man.dim == n * (n + 1) // 2


class TestBoltzmannEmbeddingOperations:
    """Test BoltzmannEmbedding operations."""

    @pytest.fixture(params=[2, 3, 4])
    def embedding(self, request: pytest.FixtureRequest) -> BoltzmannEmbedding:
        """Create BoltzmannEmbedding."""
        n = request.param
        return BoltzmannEmbedding(DiagonalBoltzmann(n), Boltzmann(n))

    def test_embed_zero_coupling(self, embedding: BoltzmannEmbedding, key: Array) -> None:
        """Test embedding creates zero off-diagonal coupling."""
        n = embedding.sub_man.n_neurons
        biases = jax.random.normal(key, (n,))

        boltz_params = embedding.embed(biases)

        # Off-diagonal elements should be zero
        full_matrix = embedding.amb_man.shp_man.to_matrix(boltz_params)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert jnp.allclose(full_matrix[i, j], 0.0, rtol=RTOL, atol=ATOL)

    def test_project_embed_round_trip(
        self, embedding: BoltzmannEmbedding, key: Array
    ) -> None:
        """Test project(embed(x)) recovers original in mean coordinates."""
        n = embedding.sub_man.n_neurons

        # Start with DiagonalBoltzmann mean parameters (probabilities)
        probs = jax.random.uniform(key, (n,), minval=0.1, maxval=0.9)

        diag_boltz = embedding.sub_man
        boltzmann = embedding.amb_man

        # Natural params for DiagonalBoltzmann
        natural_params = diag_boltz.to_natural(probs)

        # Embed into Boltzmann
        boltz_natural = embedding.embed(natural_params)

        # Convert to mean coordinates
        boltz_means = boltzmann.to_mean(boltz_natural)

        # Project back to DiagonalBoltzmann
        recovered_means = embedding.project(boltz_means)

        assert jnp.allclose(probs, recovered_means, rtol=RTOL, atol=ATOL)

    def test_translate(self, embedding: BoltzmannEmbedding, key: Array) -> None:
        """Test translate correctly adds delta to location/bias."""
        n = embedding.sub_man.n_neurons

        key1, key2 = jax.random.split(key)
        biases1 = jax.random.normal(key1, (n,))
        delta = jax.random.normal(key2, (n,))

        boltz_params = embedding.embed(biases1)
        translated = embedding.translate(boltz_params, delta)

        expected = embedding.embed(biases1 + delta)

        assert jnp.allclose(translated, expected, rtol=RTOL, atol=ATOL)
