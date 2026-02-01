"""Tests for Binomial-based Harmoniums.

Tests RestrictedBoltzmannMachine, BinomialBernoulliHarmonium, and BinomialVonMisesHarmonium.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import (
    BinomialBernoulliHarmonium,
    BinomialVonMisesHarmonium,
    RestrictedBoltzmannMachine,
    binomial_bernoulli_harmonium,
    binomial_vonmises_harmonium,
    rbm,
)

jax.config.update("jax_platform_name", "cpu")

# Tolerances
RTOL = 1e-4
ATOL = 1e-6


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestRBMBasics:
    """Test basic RBM properties."""

    @pytest.fixture(params=[(10, 5), (16, 8)])
    def model(self, request: pytest.FixtureRequest) -> RestrictedBoltzmannMachine:
        """Create RBM with various sizes."""
        n_obs, n_lat = request.param
        return rbm(n_obs, n_lat)

    def test_factory_function(self) -> None:
        """Test rbm factory function creates correct model."""
        model = rbm(100, 50)
        assert model.n_observable == 100
        assert model.n_latent == 50

    def test_dimensions(self, model: RestrictedBoltzmannMachine) -> None:
        """Test dimension properties are correct."""
        assert model.obs_man.data_dim == model.n_observable
        assert model.pst_man.data_dim == model.n_latent
        assert model.data_dim == model.n_observable + model.n_latent

    def test_initialize(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test parameter initialization."""
        params = model.initialize(key)
        expected_dim = (
            model.n_observable + model.n_observable * model.n_latent + model.n_latent
        )
        assert params.shape == (expected_dim,)
        assert jnp.all(jnp.isfinite(params))

    def test_split_join_coords(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test splitting and rejoining coordinates is invertible."""
        params = model.initialize(key)
        obs_bias, int_params, lat_bias = model.split_coords(params)

        assert obs_bias.shape == (model.n_observable,)
        assert int_params.shape == (model.n_observable * model.n_latent,)
        assert lat_bias.shape == (model.n_latent,)

        params_rejoined = model.join_coords(obs_bias, int_params, lat_bias)
        assert jnp.allclose(params, params_rejoined)


class TestRBMConditionals:
    """Test RBM conditional distribution computations."""

    @pytest.fixture(params=[(10, 5), (16, 8)])
    def model(self, request: pytest.FixtureRequest) -> RestrictedBoltzmannMachine:
        """Create RBM with various sizes."""
        n_obs, n_lat = request.param
        return rbm(n_obs, n_lat)

    def test_probabilities_in_range(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test probabilities are in [0, 1]."""
        params = model.initialize(key)
        x = jax.random.bernoulli(key, 0.5, (model.n_observable,)).astype(float)
        z = jax.random.bernoulli(key, 0.5, (model.n_latent,)).astype(float)

        x_probs = model.observable_probabilities(params, z)
        z_probs = model.latent_probabilities(params, x)

        assert jnp.all((x_probs >= 0) & (x_probs <= 1))
        assert jnp.all((z_probs >= 0) & (z_probs <= 1))

    def test_reconstruct_probabilities(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test reconstruction returns probabilities."""
        params = model.initialize(key)
        x = jax.random.bernoulli(key, 0.5, (model.n_observable,)).astype(float)
        recon = model.reconstruct(params, x)
        assert recon.shape == x.shape
        assert jnp.all((recon >= 0) & (recon <= 1))


class TestRBMFreeEnergy:
    """Test RBM free energy computation."""

    @pytest.fixture(params=[(10, 5), (16, 8)])
    def model(self, request: pytest.FixtureRequest) -> RestrictedBoltzmannMachine:
        """Create RBM with various sizes."""
        n_obs, n_lat = request.param
        return rbm(n_obs, n_lat)

    def test_free_energy_finite(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test free energy is finite."""
        params = model.initialize(key)
        x = jax.random.bernoulli(key, 0.5, (model.n_observable,)).astype(float)
        fe = model.free_energy(params, x)
        assert fe.shape == ()
        assert jnp.isfinite(fe)

    def test_mean_free_energy(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test mean free energy over batch."""
        params = model.initialize(key)
        key, subkey = jax.random.split(key)
        xs = jax.random.bernoulli(subkey, 0.5, (20, model.n_observable)).astype(float)
        mfe = model.mean_free_energy(params, xs)
        assert mfe.shape == ()
        assert jnp.isfinite(mfe)


class TestRBMGibbsSampling:
    """Test RBM Gibbs sampling."""

    @pytest.fixture(params=[(10, 5), (16, 8)])
    def model(self, request: pytest.FixtureRequest) -> RestrictedBoltzmannMachine:
        """Create RBM with various sizes."""
        n_obs, n_lat = request.param
        return rbm(n_obs, n_lat)

    def test_gibbs_step_binary(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test Gibbs step produces binary values."""
        params = model.initialize(key)
        state = jnp.zeros(model.data_dim)
        key, subkey = jax.random.split(key)
        new_state = model.gibbs_step(subkey, params, state)
        assert new_state.shape == state.shape
        assert jnp.all((new_state == 0) | (new_state == 1))

    def test_gibbs_chain(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test Gibbs chain runs multiple steps."""
        params = model.initialize(key)
        state = jnp.zeros(model.data_dim)
        key, subkey = jax.random.split(key)
        final_state = model.gibbs_chain(subkey, params, state, 10)
        assert final_state.shape == state.shape

    def test_sample(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test sampling from joint distribution."""
        params = model.initialize(key)
        key, subkey = jax.random.split(key)
        samples = model.sample(subkey, params, n=5)
        assert samples.shape == (5, model.data_dim)


class TestRBMContrastiveDivergence:
    """Test RBM contrastive divergence gradient."""

    @pytest.fixture(params=[(10, 5)])
    def model(self, request: pytest.FixtureRequest) -> RestrictedBoltzmannMachine:
        """Create RBM with various sizes."""
        n_obs, n_lat = request.param
        return rbm(n_obs, n_lat)

    def test_cd_gradient_finite(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test CD gradient is finite."""
        params = model.initialize(key)
        x = jax.random.bernoulli(key, 0.5, (model.n_observable,)).astype(float)
        key, subkey = jax.random.split(key)
        cd_grad = model.contrastive_divergence_step(subkey, params, x, k=1)
        assert cd_grad.shape == params.shape
        assert jnp.all(jnp.isfinite(cd_grad))

    def test_mean_cd_gradient(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test mean CD gradient over batch."""
        params = model.initialize(key)
        key, subkey = jax.random.split(key)
        xs = jax.random.bernoulli(subkey, 0.5, (10, model.n_observable)).astype(float)
        key, subkey = jax.random.split(key)
        mean_grad = model.mean_contrastive_divergence_gradient(subkey, params, xs, k=1)
        assert mean_grad.shape == params.shape
        assert jnp.all(jnp.isfinite(mean_grad))

    def test_cd_different_k(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test CD with different numbers of Gibbs steps."""
        params = model.initialize(key)
        x = jax.random.bernoulli(key, 0.5, (model.n_observable,)).astype(float)

        for k in [1, 3]:
            key, subkey = jax.random.split(key)
            cd_grad = model.contrastive_divergence_step(subkey, params, x, k=k)
            assert cd_grad.shape == params.shape


class TestRBMTraining:
    """Test RBM training integration."""

    @pytest.fixture
    def model(self) -> RestrictedBoltzmannMachine:
        """Create RBM."""
        return rbm(10, 5)

    def test_training_step(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test a single training step with CD."""
        import optax

        params = model.initialize(key)
        optimizer = optax.adamw(learning_rate=0.01)
        opt_state = optimizer.init(params)

        key, subkey = jax.random.split(key)
        data = jax.random.bernoulli(subkey, 0.5, (32, model.n_observable)).astype(float)

        key, subkey = jax.random.split(key)
        cd_grad = model.mean_contrastive_divergence_gradient(subkey, params, data, k=1)

        updates, opt_state = optimizer.update(cd_grad, opt_state, params)
        new_params: Array = optax.apply_updates(params, updates)  # pyright: ignore[reportAssignmentType]

        assert not jnp.allclose(params, new_params)
        assert jnp.all(jnp.isfinite(new_params))


class TestRBMVisualization:
    """Test RBM visualization helpers."""

    def test_get_filters(self) -> None:
        """Test filter extraction for visualization."""
        model = rbm(28 * 28, 64)
        key = jax.random.PRNGKey(42)
        params = model.initialize(key)

        filters = model.get_filters(params, (28, 28))
        assert filters.shape == (64, 28, 28)

    def test_reconstruction_error(self, key: Array) -> None:
        """Test reconstruction error computation."""
        model = rbm(10, 5)
        params = model.initialize(key)
        key, subkey = jax.random.split(key)
        xs = jax.random.bernoulli(subkey, 0.5, (20, model.n_observable)).astype(float)
        error = model.reconstruction_error(params, xs)
        assert error.shape == ()
        assert jnp.isfinite(error)
        assert error >= 0


class TestBinomialBernoulliHarmoniumBasics:
    """Test BinomialBernoulliHarmonium properties."""

    @pytest.fixture(params=[(10, 5, 8), (16, 8, 12)])
    def model(self, request: pytest.FixtureRequest) -> BinomialBernoulliHarmonium:
        """Create BinomialBernoulliHarmonium."""
        n_obs, n_lat, n_trials = request.param
        return binomial_bernoulli_harmonium(n_obs, n_lat, n_trials)

    def test_factory_function(self) -> None:
        """Test factory function creates correct model."""
        model = binomial_bernoulli_harmonium(100, 50, 16)
        assert model.n_observable == 100
        assert model.n_latent == 50
        assert model.n_trials == 16

    def test_dimensions(self, model: BinomialBernoulliHarmonium) -> None:
        """Test dimensions are correct."""
        assert model.obs_man.data_dim == model.n_observable
        assert model.pst_man.data_dim == model.n_latent

    def test_initialize(
        self, model: BinomialBernoulliHarmonium, key: Array
    ) -> None:
        """Test initialization produces valid parameters."""
        params = model.initialize(key)
        assert jnp.all(jnp.isfinite(params))


class TestBinomialVonMisesHarmoniumBasics:
    """Test BinomialVonMisesHarmonium properties."""

    @pytest.fixture(params=[(8, 2, 10)])
    def model(self, request: pytest.FixtureRequest) -> BinomialVonMisesHarmonium:
        """Create BinomialVonMisesHarmonium."""
        n_obs, n_lat, n_trials = request.param
        return binomial_vonmises_harmonium(n_obs, n_lat, n_trials)

    def test_factory_function(self) -> None:
        """Test factory function creates correct model."""
        model = binomial_vonmises_harmonium(16, 2, 10)
        assert model.n_observable == 16
        assert model.n_latent == 2
        assert model.n_trials == 10

    def test_dimensions(self, model: BinomialVonMisesHarmonium) -> None:
        """Test dimensions are correct."""
        assert model.obs_man.data_dim == model.n_observable
        # Latent is VonMisesProduct with n_latent components
        assert model.pst_man.data_dim == model.n_latent

    def test_initialize(
        self, model: BinomialVonMisesHarmonium, key: Array
    ) -> None:
        """Test initialization produces valid parameters."""
        params = model.initialize(key)
        assert jnp.all(jnp.isfinite(params))
