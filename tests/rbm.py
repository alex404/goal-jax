"""Tests for Restricted Boltzmann Machines."""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import RestrictedBoltzmannMachine, rbm

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture(params=[(10, 5), (20, 10), (8, 4)])
def model(request: pytest.FixtureRequest) -> RestrictedBoltzmannMachine:
    """Create RBMs with various sizes."""
    n_vis, n_hid = request.param
    return rbm(n_vis, n_hid)


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestRBMBasics:
    """Test basic RBM properties and initialization."""

    def test_factory_function(self) -> None:
        """Test the rbm factory function creates correct model."""
        model = rbm(100, 50)
        assert model.n_visible == 100
        assert model.n_hidden == 50

    def test_dimensions(self, model: RestrictedBoltzmannMachine) -> None:
        """Test dimension properties are correct."""
        assert model.obs_man.data_dim == model.n_visible
        assert model.pst_man.data_dim == model.n_hidden
        assert model.data_dim == model.n_visible + model.n_hidden

    def test_manifold_aliases(self, model: RestrictedBoltzmannMachine) -> None:
        """Test vis_man and hid_man are correct aliases."""
        assert model.vis_man.data_dim == model.n_visible
        assert model.hid_man.data_dim == model.n_hidden

    def test_initialize(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test parameter initialization."""
        params = model.initialize(key)
        expected_dim = model.n_visible + model.n_visible * model.n_hidden + model.n_hidden
        assert params.shape == (expected_dim,)
        # Check parameters are finite
        assert jnp.all(jnp.isfinite(params))

    def test_split_join_params(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test splitting and rejoining parameters is invertible."""
        params = model.initialize(key)
        vis_bias, weights, hid_bias = model.split_params(params)

        assert vis_bias.shape == (model.n_visible,)
        assert weights.shape == (model.n_visible, model.n_hidden)
        assert hid_bias.shape == (model.n_hidden,)

        # Rejoin should give same params
        params_rejoined = model.join_params(vis_bias, weights, hid_bias)
        assert jnp.allclose(params, params_rejoined)


class TestConditionalDistributions:
    """Test conditional distribution computations."""

    def test_visible_probabilities_shape(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test visible probabilities have correct shape."""
        params = model.initialize(key)
        h = jnp.zeros(model.n_hidden)
        v_probs = model.visible_probabilities(params, h)
        assert v_probs.shape == (model.n_visible,)

    def test_hidden_probabilities_shape(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test hidden probabilities have correct shape."""
        params = model.initialize(key)
        v = jnp.zeros(model.n_visible)
        h_probs = model.hidden_probabilities(params, v)
        assert h_probs.shape == (model.n_hidden,)

    def test_probabilities_in_range(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test probabilities are in [0, 1]."""
        params = model.initialize(key)
        v = jax.random.bernoulli(key, 0.5, (model.n_visible,)).astype(float)
        h = jax.random.bernoulli(key, 0.5, (model.n_hidden,)).astype(float)

        v_probs = model.visible_probabilities(params, h)
        h_probs = model.hidden_probabilities(params, v)

        assert jnp.all((v_probs >= 0) & (v_probs <= 1))
        assert jnp.all((h_probs >= 0) & (h_probs <= 1))


class TestReconstruction:
    """Test reconstruction functionality."""

    def test_reconstruct_shape(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test reconstruction has correct shape."""
        params = model.initialize(key)
        v = jax.random.bernoulli(key, 0.5, (model.n_visible,)).astype(float)
        recon = model.reconstruct(params, v)
        assert recon.shape == v.shape

    def test_reconstruct_probabilities(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test reconstruction returns probabilities."""
        params = model.initialize(key)
        v = jax.random.bernoulli(key, 0.5, (model.n_visible,)).astype(float)
        recon = model.reconstruct(params, v)
        assert jnp.all((recon >= 0) & (recon <= 1))


class TestFreeEnergy:
    """Test free energy computation."""

    def test_free_energy_shape(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test free energy is a scalar."""
        params = model.initialize(key)
        v = jax.random.bernoulli(key, 0.5, (model.n_visible,)).astype(float)
        fe = model.free_energy(params, v)
        assert fe.shape == ()

    def test_free_energy_finite(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test free energy is finite."""
        params = model.initialize(key)
        v = jax.random.bernoulli(key, 0.5, (model.n_visible,)).astype(float)
        fe = model.free_energy(params, v)
        assert jnp.isfinite(fe)

    def test_mean_free_energy(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test mean free energy over batch."""
        params = model.initialize(key)
        key, subkey = jax.random.split(key)
        vs = jax.random.bernoulli(subkey, 0.5, (20, model.n_visible)).astype(float)
        mfe = model.mean_free_energy(params, vs)
        assert mfe.shape == ()
        assert jnp.isfinite(mfe)


class TestGibbsSampling:
    """Test Gibbs sampling functionality."""

    def test_gibbs_step_shape(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test Gibbs step preserves state shape."""
        params = model.initialize(key)
        state = jnp.zeros(model.data_dim)
        key, subkey = jax.random.split(key)
        new_state = model.gibbs_step(subkey, params, state)
        assert new_state.shape == state.shape

    def test_gibbs_step_binary(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test Gibbs step produces binary values."""
        params = model.initialize(key)
        state = jnp.zeros(model.data_dim)
        key, subkey = jax.random.split(key)
        new_state = model.gibbs_step(subkey, params, state)
        # Check that values are 0 or 1
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


class TestContrastiveDivergence:
    """Test contrastive divergence gradient computation."""

    def test_cd_step_shape(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test CD step returns gradient with parameter shape."""
        params = model.initialize(key)
        v = jax.random.bernoulli(key, 0.5, (model.n_visible,)).astype(float)
        key, subkey = jax.random.split(key)
        cd_grad = model.contrastive_divergence_step(subkey, params, v, k=1)
        assert cd_grad.shape == params.shape

    def test_cd_gradient_finite(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test CD gradient is finite."""
        params = model.initialize(key)
        v = jax.random.bernoulli(key, 0.5, (model.n_visible,)).astype(float)
        key, subkey = jax.random.split(key)
        cd_grad = model.contrastive_divergence_step(subkey, params, v, k=1)
        assert jnp.all(jnp.isfinite(cd_grad))

    def test_mean_cd_gradient(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test mean CD gradient over batch."""
        params = model.initialize(key)
        key, subkey = jax.random.split(key)
        vs = jax.random.bernoulli(subkey, 0.5, (10, model.n_visible)).astype(float)
        key, subkey = jax.random.split(key)
        mean_grad = model.mean_contrastive_divergence_gradient(subkey, params, vs, k=1)
        assert mean_grad.shape == params.shape
        assert jnp.all(jnp.isfinite(mean_grad))

    def test_cd_different_k(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test CD with different numbers of Gibbs steps."""
        params = model.initialize(key)
        v = jax.random.bernoulli(key, 0.5, (model.n_visible,)).astype(float)

        for k in [1, 3, 5]:
            key, subkey = jax.random.split(key)
            cd_grad = model.contrastive_divergence_step(subkey, params, v, k=k)
            assert cd_grad.shape == params.shape


class TestVisualization:
    """Test visualization helper methods."""

    def test_get_filters(self) -> None:
        """Test filter extraction for visualization."""
        # Use image-like dimensions
        model = rbm(28 * 28, 64)
        key = jax.random.PRNGKey(42)
        params = model.initialize(key)

        filters = model.get_filters(params, (28, 28))
        assert filters.shape == (64, 28, 28)

    def test_reconstruction_error(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test reconstruction error computation."""
        params = model.initialize(key)
        key, subkey = jax.random.split(key)
        vs = jax.random.bernoulli(subkey, 0.5, (20, model.n_visible)).astype(float)
        error = model.reconstruction_error(params, vs)
        assert error.shape == ()
        assert jnp.isfinite(error)
        assert error >= 0  # MSE is non-negative


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_training_step(
        self, model: RestrictedBoltzmannMachine, key: Array
    ) -> None:
        """Test a single training step with CD."""
        from goal.geometry import Optimizer

        params = model.initialize(key)
        optimizer = Optimizer.adamw(man=model, learning_rate=0.01)
        opt_state = optimizer.init(params)

        # Generate some random data
        key, subkey = jax.random.split(key)
        data = jax.random.bernoulli(subkey, 0.5, (32, model.n_visible)).astype(float)

        # Compute CD gradient
        key, subkey = jax.random.split(key)
        cd_grad = model.mean_contrastive_divergence_gradient(subkey, params, data, k=1)

        # Update parameters
        opt_state, new_params = optimizer.update(opt_state, cd_grad, params)

        # Check parameters changed
        assert not jnp.allclose(params, new_params)
        assert jnp.all(jnp.isfinite(new_params))
