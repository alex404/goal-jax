"""Tests for Dynamical Models (Kalman Filter, HMM, and state-space inference).

Tests the generic LatentProcess framework, KalmanFilter, and HiddenMarkovModel
implementations. Includes brute-force validation for HMM algorithms on small
state spaces.
"""

import itertools

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.models import (
    CategoricalEmission,
    CategoricalTransition,
    HiddenMarkovModel,
    KalmanFilter,
    NormalEmission,
    NormalTransition,
    conjugated_filtering,
    conjugated_smoothing,
    create_hmm,
    create_kalman_filter,
    latent_process_log_density,
    latent_process_log_observable_density,
    sample_latent_process,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Tolerances
RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def key() -> Array:
    """Random key for tests."""
    return jax.random.PRNGKey(42)


class TestKalmanFilterStructure:
    """Test KalmanFilter model structure."""

    @pytest.fixture
    def model(self) -> KalmanFilter:
        """Create a simple KalmanFilter."""
        return create_kalman_filter(obs_dim=2, lat_dim=3)

    def test_dimensions(self, model: KalmanFilter) -> None:
        """Test that dimensions are correct."""
        assert model.obs_man.data_dim == 2
        assert model.lat_man.data_dim == 3

    def test_emission_structure(self, model: KalmanFilter) -> None:
        """Test emission harmonium structure."""
        emsn = model.emsn_hrm
        assert isinstance(emsn, NormalEmission)
        assert emsn.obs_dim == 2
        assert emsn.lat_dim == 3

    def test_transition_structure(self, model: KalmanFilter) -> None:
        """Test transition harmonium structure."""
        trns = model.trns_hrm
        assert isinstance(trns, NormalTransition)
        assert trns.lat_dim == 3

    def test_split_join_roundtrip(self, model: KalmanFilter, key: Array) -> None:
        """Test that split_coords and join_coords are inverses."""
        params = model.initialize(key, location=0.0, shape=0.5)
        prior, emsn, trns = model.split_coords(params)
        recovered = model.join_coords(prior, emsn, trns)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)


class TestNormalTransition:
    """Test NormalTransition harmonium."""

    @pytest.fixture
    def model(self) -> NormalTransition:
        """Create a transition model."""
        return NormalTransition(lat_dim=3)

    def test_dimensions(self, model: NormalTransition) -> None:
        """Test that dimensions are correct."""
        assert model.obs_man.data_dim == 3
        assert model.lat_man.data_dim == 3

    def test_conjugation_equation(self, model: NormalTransition, key: Array) -> None:
        """Test: ψ(θ_X + θ_{XZ}·s_Z(z)) = ρ·s_Z(z) + ψ_X(θ_X)."""
        params = model.initialize(key, location=0.0, shape=0.5)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        for i in range(3):
            z = jax.random.normal(jax.random.fold_in(key, i), (model.lat_man.data_dim,))
            s_z = model.lat_man.sufficient_statistic(z)

            # LHS: ψ(θ_X + θ_{XZ}·s_Z(z))
            conditional_obs = model.lkl_fun_man(lkl_params, z)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            # RHS: ρ·s_Z(z) + ψ_X(θ_X)
            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

            assert jnp.abs(lhs - rhs) < ATOL


class TestNormalEmission:
    """Test NormalEmission harmonium."""

    @pytest.fixture
    def model(self) -> NormalEmission:
        """Create an emission model."""
        return NormalEmission(obs_dim=2, _lat_dim=3)

    def test_dimensions(self, model: NormalEmission) -> None:
        """Test that dimensions are correct."""
        assert model.obs_man.data_dim == 2
        assert model.lat_man.data_dim == 3

    def test_conjugation_equation(self, model: NormalEmission, key: Array) -> None:
        """Test: ψ(θ_X + θ_{XZ}·s_Z(z)) = ρ·s_Z(z) + ψ_X(θ_X)."""
        params = model.initialize(key, location=0.0, shape=0.5)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        for i in range(3):
            z = jax.random.normal(jax.random.fold_in(key, i), (model.lat_man.data_dim,))
            s_z = model.lat_man.sufficient_statistic(z)

            # LHS: ψ(θ_X + θ_{XZ}·s_Z(z))
            conditional_obs = model.lkl_fun_man(lkl_params, z)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            # RHS: ρ·s_Z(z) + ψ_X(θ_X)
            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

            assert jnp.abs(lhs - rhs) < ATOL


class TestKalmanFilterSampling:
    """Test sampling from Kalman Filter."""

    @pytest.fixture
    def model(self) -> KalmanFilter:
        """Create a simple KalmanFilter."""
        return create_kalman_filter(obs_dim=2, lat_dim=3)

    @pytest.fixture
    def params(self, model: KalmanFilter, key: Array) -> Array:
        """Initialize model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    def test_sample_shapes(
        self, model: KalmanFilter, params: Array, key: Array
    ) -> None:
        """Test that sampled trajectories have correct shapes."""
        n_steps = 10
        observations, latents = sample_latent_process(model, params, key, n_steps)

        assert observations.shape == (n_steps, model.obs_man.data_dim)
        assert latents.shape == (n_steps + 1, model.lat_man.data_dim)

    def test_sample_finite(
        self, model: KalmanFilter, params: Array, key: Array
    ) -> None:
        """Test that sampled values are finite."""
        observations, latents = sample_latent_process(model, params, key, n_steps=20)

        assert jnp.all(jnp.isfinite(observations))
        assert jnp.all(jnp.isfinite(latents))


class TestKalmanFilterFiltering:
    """Test forward filtering for Kalman Filter."""

    @pytest.fixture
    def model(self) -> KalmanFilter:
        """Create a simple KalmanFilter."""
        return create_kalman_filter(obs_dim=2, lat_dim=3)

    @pytest.fixture
    def params(self, model: KalmanFilter, key: Array) -> Array:
        """Initialize model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    @pytest.fixture
    def data(
        self, model: KalmanFilter, params: Array, key: Array
    ) -> tuple[Array, Array]:
        """Sample data from model."""
        key_sample = jax.random.fold_in(key, 100)
        return sample_latent_process(model, params, key_sample, n_steps=15)

    def test_filtering_shape(
        self, model: KalmanFilter, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that filtered distributions have correct shape."""
        observations, _ = data
        filtered, log_lik = conjugated_filtering(model, params, observations)

        assert filtered.shape == (observations.shape[0], model.lat_man.dim)
        assert log_lik.shape == ()

    def test_filtering_finite(
        self, model: KalmanFilter, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that filtered values are finite."""
        observations, _ = data
        filtered, log_lik = conjugated_filtering(model, params, observations)

        assert jnp.all(jnp.isfinite(filtered))
        assert jnp.isfinite(log_lik)


class TestKalmanFilterSmoothing:
    """Test forward-backward smoothing for Kalman Filter."""

    @pytest.fixture
    def model(self) -> KalmanFilter:
        """Create a simple KalmanFilter."""
        return create_kalman_filter(obs_dim=2, lat_dim=3)

    @pytest.fixture
    def params(self, model: KalmanFilter, key: Array) -> Array:
        """Initialize model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    @pytest.fixture
    def data(
        self, model: KalmanFilter, params: Array, key: Array
    ) -> tuple[Array, Array]:
        """Sample data from model."""
        key_sample = jax.random.fold_in(key, 100)
        return sample_latent_process(model, params, key_sample, n_steps=15)

    def test_smoothing_shape(
        self, model: KalmanFilter, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that smoothed distributions have correct shape."""
        observations, _ = data
        smoothed = conjugated_smoothing(model, params, observations)

        assert smoothed.shape == (observations.shape[0], model.lat_man.dim)

    def test_smoothing_finite(
        self, model: KalmanFilter, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that smoothed values are finite."""
        observations, _ = data
        smoothed = conjugated_smoothing(model, params, observations)

        assert jnp.all(jnp.isfinite(smoothed))

    def test_last_smoothed_equals_last_filtered(
        self, model: KalmanFilter, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that smoothed[T] == filtered[T] (boundary condition)."""
        observations, _ = data
        filtered, _ = conjugated_filtering(model, params, observations)
        smoothed = conjugated_smoothing(model, params, observations)

        assert jnp.allclose(smoothed[-1], filtered[-1], rtol=RTOL, atol=ATOL)


class TestKalmanFilterLogLikelihood:
    """Test log-likelihood computation for Kalman Filter."""

    @pytest.fixture
    def model(self) -> KalmanFilter:
        """Create a simple KalmanFilter."""
        return create_kalman_filter(obs_dim=2, lat_dim=3)

    @pytest.fixture
    def params(self, model: KalmanFilter, key: Array) -> Array:
        """Initialize model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    @pytest.fixture
    def data(
        self, model: KalmanFilter, params: Array, key: Array
    ) -> tuple[Array, Array]:
        """Sample data from model."""
        key_sample = jax.random.fold_in(key, 100)
        return sample_latent_process(model, params, key_sample, n_steps=15)

    def test_log_observable_density_finite(
        self, model: KalmanFilter, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that log observable density is finite."""
        observations, _ = data
        log_lik = latent_process_log_observable_density(model, params, observations)
        assert jnp.isfinite(log_lik)

    def test_log_density_finite(
        self, model: KalmanFilter, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that joint log density is finite."""
        observations, latents = data
        log_joint = latent_process_log_density(model, params, observations, latents)
        assert jnp.isfinite(log_joint)

    def test_log_observable_matches_filtering(
        self, model: KalmanFilter, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that log observable density matches filtering result."""
        observations, _ = data
        log_lik1 = latent_process_log_observable_density(model, params, observations)
        _, log_lik2 = conjugated_filtering(model, params, observations)

        assert jnp.allclose(log_lik1, log_lik2, rtol=RTOL, atol=ATOL)


# =============================================================================
# Hidden Markov Model Tests
# =============================================================================


class TestHMMStructure:
    """Test HiddenMarkovModel structure."""

    @pytest.fixture
    def model(self) -> HiddenMarkovModel:
        """Create a simple HMM."""
        return create_hmm(n_obs=2, n_states=3)

    def test_dimensions(self, model: HiddenMarkovModel) -> None:
        """Test that dimensions are correct."""
        assert model.obs_man.n_categories == 2
        assert model.lat_man.n_categories == 3
        assert model.obs_man.dim == 1  # n_categories - 1
        assert model.lat_man.dim == 2  # n_categories - 1

    def test_emission_structure(self, model: HiddenMarkovModel) -> None:
        """Test emission harmonium structure."""
        emsn = model.emsn_hrm
        assert isinstance(emsn, CategoricalEmission)
        assert emsn.n_obs == 2
        assert emsn.n_states == 3

    def test_transition_structure(self, model: HiddenMarkovModel) -> None:
        """Test transition harmonium structure."""
        trns = model.trns_hrm
        assert isinstance(trns, CategoricalTransition)
        assert trns.n_states == 3

    def test_split_join_roundtrip(self, model: HiddenMarkovModel, key: Array) -> None:
        """Test that split_coords and join_coords are inverses."""
        params = model.initialize(key, location=0.0, shape=0.5)
        prior, emsn, trns = model.split_coords(params)
        recovered = model.join_coords(prior, emsn, trns)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)


class TestCategoricalTransition:
    """Test CategoricalTransition harmonium."""

    @pytest.fixture
    def model(self) -> CategoricalTransition:
        """Create a transition model."""
        return CategoricalTransition(n_states=3)

    def test_dimensions(self, model: CategoricalTransition) -> None:
        """Test that dimensions are correct."""
        assert model.obs_man.n_categories == 3
        assert model.lat_man.n_categories == 3
        assert model.obs_man.dim == 2
        assert model.lat_man.dim == 2

    def test_conjugation_equation(self, model: CategoricalTransition, key: Array) -> None:
        """Test: ψ(θ_X + θ_{XZ}·s_Z(z)) = ρ·s_Z(z) + ψ_X(θ_X).

        This verifies the conjugation parameters satisfy the fundamental equation
        that makes conjugation work for filtering/smoothing.
        """
        params = model.initialize(key, location=0.0, shape=0.5)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        # Test for each possible latent state
        for z_val in range(model.n_states):
            z = jnp.array([z_val])
            s_z = model.lat_man.sufficient_statistic(z)

            # LHS: ψ(θ_X + θ_{XZ}·s_Z(z))
            conditional_obs = model.lkl_fun_man(lkl_params, s_z)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            # RHS: ρ·s_Z(z) + ψ_X(θ_X)
            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

            assert jnp.abs(lhs - rhs) < ATOL, f"Failed for z={z_val}: {lhs} vs {rhs}"


class TestCategoricalEmission:
    """Test CategoricalEmission harmonium."""

    @pytest.fixture
    def model(self) -> CategoricalEmission:
        """Create an emission model."""
        return CategoricalEmission(n_obs=2, _n_states=3)

    def test_dimensions(self, model: CategoricalEmission) -> None:
        """Test that dimensions are correct."""
        assert model.obs_man.n_categories == 2
        assert model.lat_man.n_categories == 3

    def test_conjugation_equation(self, model: CategoricalEmission, key: Array) -> None:
        """Test: ψ(θ_X + θ_{XZ}·s_Z(z)) = ρ·s_Z(z) + ψ_X(θ_X)."""
        params = model.initialize(key, location=0.0, shape=0.5)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        for z_val in range(model.n_states):
            z = jnp.array([z_val])
            s_z = model.lat_man.sufficient_statistic(z)

            conditional_obs = model.lkl_fun_man(lkl_params, s_z)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

            assert jnp.abs(lhs - rhs) < ATOL, f"Failed for z={z_val}: {lhs} vs {rhs}"


class TestHMMSampling:
    """Test sampling from HMM."""

    @pytest.fixture
    def model(self) -> HiddenMarkovModel:
        """Create a simple HMM."""
        return create_hmm(n_obs=2, n_states=3)

    @pytest.fixture
    def params(self, model: HiddenMarkovModel, key: Array) -> Array:
        """Initialize model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    def test_sample_shapes(
        self, model: HiddenMarkovModel, params: Array, key: Array
    ) -> None:
        """Test that sampled trajectories have correct shapes."""
        n_steps = 10
        observations, latents = sample_latent_process(model, params, key, n_steps)

        # Observations: (n_steps, data_dim) where data_dim=1 for Categorical
        assert observations.shape == (n_steps, model.obs_man.data_dim)
        # Latents: (n_steps+1, data_dim) including z_0
        assert latents.shape == (n_steps + 1, model.lat_man.data_dim)

    def test_sample_valid_values(
        self, model: HiddenMarkovModel, params: Array, key: Array
    ) -> None:
        """Test that sampled values are valid category indices."""
        observations, latents = sample_latent_process(model, params, key, n_steps=20)

        # All values should be valid category indices
        assert jnp.all(observations >= 0)
        assert jnp.all(observations < model.n_obs)
        assert jnp.all(latents >= 0)
        assert jnp.all(latents < model.n_states)


class TestHMMFiltering:
    """Test forward filtering for HMM."""

    @pytest.fixture
    def model(self) -> HiddenMarkovModel:
        """Create a simple HMM."""
        return create_hmm(n_obs=2, n_states=3)

    @pytest.fixture
    def params(self, model: HiddenMarkovModel, key: Array) -> Array:
        """Initialize model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    @pytest.fixture
    def data(
        self, model: HiddenMarkovModel, params: Array, key: Array
    ) -> tuple[Array, Array]:
        """Sample data from model."""
        key_sample = jax.random.fold_in(key, 100)
        return sample_latent_process(model, params, key_sample, n_steps=10)

    def test_filtering_shape(
        self, model: HiddenMarkovModel, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that filtered distributions have correct shape."""
        observations, _ = data
        filtered, log_lik = conjugated_filtering(model, params, observations)

        assert filtered.shape == (observations.shape[0], model.lat_man.dim)
        assert log_lik.shape == ()

    def test_filtering_finite(
        self, model: HiddenMarkovModel, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that filtered values are finite."""
        observations, _ = data
        filtered, log_lik = conjugated_filtering(model, params, observations)

        assert jnp.all(jnp.isfinite(filtered))
        assert jnp.isfinite(log_lik)


class TestHMMSmoothing:
    """Test forward-backward smoothing for HMM."""

    @pytest.fixture
    def model(self) -> HiddenMarkovModel:
        """Create a simple HMM."""
        return create_hmm(n_obs=2, n_states=3)

    @pytest.fixture
    def params(self, model: HiddenMarkovModel, key: Array) -> Array:
        """Initialize model parameters."""
        return model.initialize(key, location=0.0, shape=0.5)

    @pytest.fixture
    def data(
        self, model: HiddenMarkovModel, params: Array, key: Array
    ) -> tuple[Array, Array]:
        """Sample data from model."""
        key_sample = jax.random.fold_in(key, 100)
        return sample_latent_process(model, params, key_sample, n_steps=10)

    def test_smoothing_shape(
        self, model: HiddenMarkovModel, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that smoothed distributions have correct shape."""
        observations, _ = data
        smoothed = conjugated_smoothing(model, params, observations)

        assert smoothed.shape == (observations.shape[0], model.lat_man.dim)

    def test_smoothing_finite(
        self, model: HiddenMarkovModel, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that smoothed values are finite."""
        observations, _ = data
        smoothed = conjugated_smoothing(model, params, observations)

        assert jnp.all(jnp.isfinite(smoothed))

    def test_last_smoothed_equals_last_filtered(
        self, model: HiddenMarkovModel, params: Array, data: tuple[Array, Array]
    ) -> None:
        """Test that smoothed[T] == filtered[T] (boundary condition)."""
        observations, _ = data
        filtered, _ = conjugated_filtering(model, params, observations)
        smoothed = conjugated_smoothing(model, params, observations)

        assert jnp.allclose(smoothed[-1], filtered[-1], rtol=RTOL, atol=ATOL)


# =============================================================================
# Brute-Force Reference Implementation for HMM Validation
# =============================================================================


def brute_force_log_joint(
    model: HiddenMarkovModel,
    params: Array,
    observations: Array,
    latent_seq: tuple[int, ...],
) -> Array:
    """Compute log p(x_{1:T}, z_{0:T}) for a specific latent sequence.

    Args:
        model: HMM model
        params: Model parameters
        observations: Shape (T,) - observation indices
        latent_seq: Tuple of T+1 latent state indices (z_0, z_1, ..., z_T)

    Returns:
        Log joint probability (scalar)
    """
    prior_params, emsn_params, trns_params = model.split_coords(params)
    n_steps = len(observations)

    # Log prior: log p(z_0)
    z0 = jnp.array([latent_seq[0]])
    log_prob = model.lat_man.log_density(prior_params, z0)

    # Get likelihood params
    trns_lkl_params, _ = model.trns_hrm.split_conjugated(trns_params)
    emsn_lkl_params, _ = model.emsn_hrm.split_conjugated(emsn_params)

    for t in range(n_steps):
        z_prev = jnp.array([latent_seq[t]])
        z_curr = jnp.array([latent_seq[t + 1]])
        x_t = observations[t]

        # Log transition: log p(z_t | z_{t-1})
        s_prev = model.lat_man.sufficient_statistic(z_prev)
        z_params = model.trns_hrm.lkl_fun_man(trns_lkl_params, s_prev)
        log_prob += model.lat_man.log_density(z_params, z_curr)

        # Log emission: log p(x_t | z_t)
        s_curr = model.lat_man.sufficient_statistic(z_curr)
        x_params = model.emsn_hrm.lkl_fun_man(emsn_lkl_params, s_curr)
        log_prob += model.obs_man.log_density(x_params, x_t)

    return log_prob


def brute_force_filtering(
    model: HiddenMarkovModel,
    params: Array,
    observations: Array,
) -> tuple[Array, Array]:
    """Compute exact filtered marginals by enumerating all latent sequences.

    For small state spaces, we can compute exact marginals by:
    1. Enumerating all possible latent sequences
    2. Computing joint probability for each
    3. Marginalizing to get p(z_t | x_{1:t})

    Args:
        model: HMM model
        params: Model parameters
        observations: Shape (T, 1) - observation indices

    Returns:
        Tuple of (filtered_probs, log_likelihood):
        - filtered_probs: Shape (T, n_states) - p(z_t | x_{1:t})
        - log_likelihood: Scalar log p(x_{1:T})
    """
    n_steps = len(observations)
    n_states = model.n_states
    obs_flat = observations.flatten().astype(int)

    # Enumerate all possible latent sequences of length n_steps + 1
    all_seqs = list(itertools.product(range(n_states), repeat=n_steps + 1))

    # Compute log joint for each sequence
    log_joints = jnp.array([
        brute_force_log_joint(model, params, obs_flat, seq)
        for seq in all_seqs
    ])

    # Log marginal likelihood: log sum exp over all sequences
    log_likelihood = jax.scipy.special.logsumexp(log_joints)

    # For each time t, compute p(z_t | x_{1:t})
    # This requires computing p(z_t, x_{1:t}) and normalizing
    filtered_probs = []

    for t in range(n_steps):
        # For filtering at time t, we need p(z_t | x_{1:t})
        # This means summing over all sequences that agree on x_{1:t}
        # and marginalizing over z_{0:t-1} and z_{t+1:T}

        # Actually, for p(z_t | x_{1:t}), we sum over all z_{0:t-1}
        # but since we're computing from full sequences, we need
        # to be careful about what we're marginalizing

        # Let's compute p(z_t | x_{1:T}) first (smoothed), then handle filtering
        # Actually brute force filtering is trickier - let's compute the
        # unnormalized p(z_t, x_{1:t}) for each z_t

        probs_t = []
        for z_t in range(n_states):
            # Sum over all sequences where z[t+1] = z_t
            # (indexing: z_0 at index 0, z_1 at index 1, ..., z_t at index t, z_{t+1} at index t+1)
            # For filtering at step t (observation x_t), we want p(z_t | x_{1:t})
            # The latent sequence is (z_0, z_1, ..., z_T) where z_t corresponds to x_t
            log_sum = -jnp.inf
            for i, seq in enumerate(all_seqs):
                if seq[t + 1] == z_t:  # z_{t+1} in the sequence corresponds to z_t for obs x_t
                    log_sum = jnp.logaddexp(log_sum, log_joints[i])
            probs_t.append(log_sum)

        log_probs_t = jnp.array(probs_t)
        # Normalize to get p(z_t | x_{1:T})
        probs_t_norm = jnp.exp(log_probs_t - jax.scipy.special.logsumexp(log_probs_t))
        filtered_probs.append(probs_t_norm)

    return jnp.array(filtered_probs), log_likelihood


def brute_force_smoothing(
    model: HiddenMarkovModel,
    params: Array,
    observations: Array,
) -> Array:
    """Compute exact smoothed marginals by enumerating all latent sequences.

    Args:
        model: HMM model
        params: Model parameters
        observations: Shape (T, 1) - observation indices

    Returns:
        smoothed_probs: Shape (T, n_states) - p(z_t | x_{1:T})
    """
    n_steps = len(observations)
    n_states = model.n_states
    obs_flat = observations.flatten().astype(int)

    # Enumerate all possible latent sequences
    all_seqs = list(itertools.product(range(n_states), repeat=n_steps + 1))

    # Compute log joint for each sequence
    log_joints = jnp.array([
        brute_force_log_joint(model, params, obs_flat, seq)
        for seq in all_seqs
    ])

    # For each time t, compute p(z_t | x_{1:T})
    smoothed_probs = []

    for t in range(n_steps):
        probs_t = []
        for z_t in range(n_states):
            # Sum over all sequences where z[t+1] = z_t
            log_sum = -jnp.inf
            for i, seq in enumerate(all_seqs):
                if seq[t + 1] == z_t:
                    log_sum = jnp.logaddexp(log_sum, log_joints[i])
            probs_t.append(log_sum)

        log_probs_t = jnp.array(probs_t)
        probs_t_norm = jnp.exp(log_probs_t - jax.scipy.special.logsumexp(log_probs_t))
        smoothed_probs.append(probs_t_norm)

    return jnp.array(smoothed_probs)


class TestHMMBruteForceValidation:
    """Compare generic algorithms against brute-force on small HMMs.

    This is the key validation: we use small discrete state spaces where
    we can enumerate all possible latent sequences and compute exact
    marginals by brute force. If the generic algorithms match, they're correct.
    """

    @pytest.fixture
    def small_hmm(self) -> HiddenMarkovModel:
        """Create a small HMM for brute-force comparison."""
        return create_hmm(n_obs=2, n_states=3)

    @pytest.fixture
    def params(self, small_hmm: HiddenMarkovModel, key: Array) -> Array:
        """Initialize model parameters."""
        return small_hmm.initialize(key, location=0.0, shape=0.5)

    @pytest.fixture
    def short_data(
        self, small_hmm: HiddenMarkovModel, params: Array, key: Array
    ) -> tuple[Array, Array]:
        """Sample short trajectory for brute-force comparison."""
        key_sample = jax.random.fold_in(key, 200)
        # Use short sequence (4 steps) so brute force is tractable
        # 3^5 = 243 possible latent sequences
        return sample_latent_process(small_hmm, params, key_sample, n_steps=4)

    def test_smoothing_matches_brute_force(
        self,
        small_hmm: HiddenMarkovModel,
        params: Array,
        short_data: tuple[Array, Array],
    ) -> None:
        """Generic smoothing should match brute-force smoothing."""
        observations, _ = short_data

        # Get smoothed distributions from generic algorithm
        smoothed_nat = conjugated_smoothing(small_hmm, params, observations)

        # Convert to probabilities
        smoothed_probs_generic = jnp.array([
            small_hmm.lat_man.to_probs(small_hmm.lat_man.to_mean(nat))
            for nat in smoothed_nat
        ])

        # Get brute-force smoothed probabilities
        smoothed_probs_brute = brute_force_smoothing(small_hmm, params, observations)

        # Compare
        assert jnp.allclose(
            smoothed_probs_generic, smoothed_probs_brute, rtol=1e-3, atol=1e-3
        ), f"Mismatch:\nGeneric:\n{smoothed_probs_generic}\nBrute:\n{smoothed_probs_brute}"

    def test_log_likelihood_matches_brute_force(
        self,
        small_hmm: HiddenMarkovModel,
        params: Array,
        short_data: tuple[Array, Array],
    ) -> None:
        """Log likelihood from filtering should match brute-force."""
        observations, _ = short_data

        # Get log likelihood from filtering
        _, log_lik_generic = conjugated_filtering(small_hmm, params, observations)

        # Get brute-force log likelihood
        _, log_lik_brute = brute_force_filtering(small_hmm, params, observations)

        assert jnp.allclose(
            log_lik_generic, log_lik_brute, rtol=1e-3, atol=1e-3
        ), f"Mismatch: generic={log_lik_generic}, brute={log_lik_brute}"
