"""Tests for models/base/gaussian/boltzmann.py.

Covers Boltzmann, DiagonalBoltzmann, and BoltzmannEmbedding. Verifies sufficient
statistics, density normalization, split/join round-trips, unit conditional
probabilities, and embedding operations.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import Symmetric
from goal.models import Bernoullis, Boltzmann, BoltzmannEmbedding, DiagonalBoltzmann

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-5
ATOL = 1e-7


class TestBoltzmann:
    """Test Boltzmann (full coupling) distribution."""

    @pytest.mark.parametrize("n", [3, 4])
    def test_dimensions(self, n: int) -> None:
        model = Boltzmann(n)
        assert model.dim == n * (n + 1) // 2
        assert model.data_dim == n

    def test_sufficient_statistic(self) -> None:
        model = Boltzmann(4)
        state = jax.random.bernoulli(jax.random.PRNGKey(42), 0.5, (4,))
        suff_stat = model.sufficient_statistic(state)
        expected = Symmetric().from_matrix(jnp.outer(state, state))
        assert jnp.allclose(suff_stat, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("n", [3, 4])
    def test_density_normalizes(self, n: int) -> None:
        """Densities sum to 1 over all binary states."""
        model = Boltzmann(n)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-2.0, maxval=2.0)
        densities = jax.vmap(lambda s: model.density(params, s))(model.states)
        assert jnp.allclose(jnp.sum(densities), 1.0, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("n", [3, 4])
    def test_log_partition_matches_sum(self, n: int) -> None:
        """Log partition equals log sum over all states."""
        model = Boltzmann(n)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-2.0, maxval=2.0)
        log_z = model.log_partition_function(params)
        energies = jax.vmap(lambda s: jnp.dot(params, model.sufficient_statistic(s)))(model.states)
        assert jnp.allclose(log_z, jax.scipy.special.logsumexp(energies), rtol=RTOL, atol=ATOL)

    def test_split_join_round_trip(self) -> None:
        model = Boltzmann(4)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-2.0, maxval=2.0)
        loc, prec = model.split_location_precision(params)
        recovered = model.join_location_precision(loc, prec)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_loc_man_is_bernoullis(self) -> None:
        model = Boltzmann(4)
        assert isinstance(model.loc_man, Bernoullis)
        assert model.loc_man.n_neurons == 4

    def test_unit_conditional_prob(self) -> None:
        """unit_conditional_prob matches direct energy-difference computation."""
        model = Boltzmann(4)
        coupling = model.shp_man
        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, (model.dim,), minval=-2.0, maxval=2.0)

        for i in range(3):
            key, state_key, unit_key = jax.random.split(jax.random.fold_in(key, i), 3)
            state = jax.random.bernoulli(state_key, 0.5, (4,)).astype(jnp.float32)
            unit_idx = jax.random.randint(unit_key, (), 0, 4)

            prob_fast = coupling.unit_conditional_prob(state, unit_idx, params)

            state_0 = state.at[unit_idx].set(0.0)
            state_1 = state.at[unit_idx].set(1.0)
            energy_0 = jnp.dot(params, coupling.sufficient_statistic(state_0))
            energy_1 = jnp.dot(params, coupling.sufficient_statistic(state_1))
            prob_direct = jax.nn.sigmoid(energy_1 - energy_0)

            assert jnp.allclose(prob_fast, prob_direct, rtol=RTOL, atol=ATOL)


    def test_to_mean_via_autodiff(self) -> None:
        """to_mean (autodiff of log_partition) matches stochastic_to_mean (Gibbs sampling)."""
        model = Boltzmann(3)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-1.0, maxval=1.0)
        autodiff_means = model.to_mean(params)
        stochastic_means = model.stochastic_to_mean(jax.random.PRNGKey(7), params, 50_000)
        assert jnp.allclose(autodiff_means, stochastic_means, atol=0.05)

    def test_gibbs_sampling(self) -> None:
        """Gibbs samples produce correct equilibrium distribution."""
        model = Boltzmann(3)
        params = jax.random.uniform(jax.random.PRNGKey(42), (model.dim,), minval=-1.0, maxval=1.0)
        samples = model.sample(jax.random.PRNGKey(0), params, 50_000)
        assert samples.shape == (50_000, model.data_dim)
        empirical_means = model.average_sufficient_statistic(samples)
        theoretical_means = model.to_mean(params)
        assert jnp.allclose(empirical_means, theoretical_means, atol=0.05)


class TestDiagonalBoltzmann:
    """Test DiagonalBoltzmann (independent binary units)."""

    @pytest.mark.parametrize("n", [2, 5])
    def test_dimensions(self, n: int) -> None:
        model = DiagonalBoltzmann(n)
        assert model.dim == n
        assert model.data_dim == n

    def test_generalized_gaussian_interface(self) -> None:
        model = DiagonalBoltzmann(3)
        assert isinstance(model.loc_man, Bernoullis)
        assert isinstance(model.shp_man, Bernoullis)
        assert model.loc_man.n_neurons == 3

    @pytest.mark.parametrize("n", [2, 5])
    def test_split_join_location_precision(self, n: int) -> None:
        model = DiagonalBoltzmann(n)
        params = jax.random.normal(jax.random.PRNGKey(42), (n,))
        loc, prec = model.split_location_precision(params)
        recovered = model.join_location_precision(loc, prec)
        assert jnp.allclose(params, recovered, rtol=RTOL, atol=ATOL)

    def test_split_join_mean_second_moment(self) -> None:
        """For independent binary, first and second moments are identical."""
        model = DiagonalBoltzmann(3)
        means = jax.random.uniform(jax.random.PRNGKey(42), (3,), minval=0.1, maxval=0.9)
        first, second = model.split_mean_second_moment(means)
        assert jnp.allclose(first, means, rtol=RTOL, atol=ATOL)
        assert jnp.allclose(second, means, rtol=RTOL, atol=ATOL)
        recovered = model.join_mean_second_moment(first, second)
        assert jnp.allclose(recovered, means, rtol=RTOL, atol=ATOL)


class TestBoltzmannEmbedding:
    """Test BoltzmannEmbedding (DiagonalBoltzmann -> Boltzmann)."""

    @pytest.mark.parametrize("n", [2, 4])
    def test_manifolds(self, n: int) -> None:
        emb = BoltzmannEmbedding(DiagonalBoltzmann(n), Boltzmann(n))
        assert emb.sub_man.dim == n
        assert emb.amb_man.dim == n * (n + 1) // 2

    def test_embed_zero_coupling(self) -> None:
        """Embedding creates zero off-diagonal coupling."""
        n = 3
        emb = BoltzmannEmbedding(DiagonalBoltzmann(n), Boltzmann(n))
        biases = jax.random.normal(jax.random.PRNGKey(42), (n,))
        boltz_params = emb.embed(biases)
        full_matrix = emb.amb_man.shp_man.to_matrix(boltz_params)
        mask = ~jnp.eye(n, dtype=bool)
        assert jnp.allclose(full_matrix[mask], 0.0, rtol=RTOL, atol=ATOL)

    def test_project_embed_round_trip(self) -> None:
        """project(embed(natural)) recovers original in mean coordinates."""
        n = 3
        emb = BoltzmannEmbedding(DiagonalBoltzmann(n), Boltzmann(n))
        probs = jax.random.uniform(jax.random.PRNGKey(42), (n,), minval=0.1, maxval=0.9)
        natural_params = emb.sub_man.to_natural(probs)
        boltz_natural = emb.embed(natural_params)
        boltz_means = emb.amb_man.to_mean(boltz_natural)
        recovered_means = emb.project(boltz_means)
        assert jnp.allclose(probs, recovered_means, rtol=RTOL, atol=ATOL)

    def test_translate(self) -> None:
        """Translate correctly adds delta to location/bias."""
        n = 3
        emb = BoltzmannEmbedding(DiagonalBoltzmann(n), Boltzmann(n))
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        biases = jax.random.normal(key1, (n,))
        delta = jax.random.normal(key2, (n,))
        boltz_params = emb.embed(biases)
        translated = emb.translate(boltz_params, delta)
        expected = emb.embed(biases + delta)
        assert jnp.allclose(translated, expected, rtol=RTOL, atol=ATOL)
