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
    CategoricalMarkovChain,
    CategoricalEmission,
    CategoricalTransition,
    GaussianMarkovChain,
    HiddenMarkovModel,
    KalmanFilter,
    NormalEmission,
    NormalTransition,
    conjugated_filtering,
    conjugated_smoothing,
    conjugated_smoothing0,
    create_categorical_markov_chain,
    create_gaussian_markov_chain,
    create_hmm,
    create_kalman_filter,
    latent_process_expectation_maximization,
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
        """Test: \\psi(\\theta_X + \\theta_{XZ} \\cdot s_Z(z)) = \\rho \\cdot s_Z(z) + \\psi_X(\\theta_X)."""
        params = model.initialize(key, location=0.0, shape=0.5)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        for i in range(3):
            z = jax.random.normal(jax.random.fold_in(key, i), (model.lat_man.data_dim,))
            s_z = model.lat_man.sufficient_statistic(z)

            # LHS: \psi(\theta_X + \theta_{XZ} \cdot s_Z(z))
            conditional_obs = model.lkl_fun_man(lkl_params, z)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            # RHS: \rho \cdot s_Z(z) + \psi_X(\theta_X)
            rhs = jnp.dot(rho, s_z) + model.obs_man.log_partition_function(obs_params)

            assert jnp.abs(lhs - rhs) < ATOL


class TestNormalEmission:
    """Test NormalEmission harmonium."""

    @pytest.fixture
    def model(self) -> NormalEmission:
        """Create an emission model."""
        return NormalEmission(obs_dim=2, lat_dim=3)

    def test_dimensions(self, model: NormalEmission) -> None:
        """Test that dimensions are correct."""
        assert model.obs_man.data_dim == 2
        assert model.lat_man.data_dim == 3

    def test_conjugation_equation(self, model: NormalEmission, key: Array) -> None:
        """Test: \\psi(\\theta_X + \\theta_{XZ} \\cdot s_Z(z)) = \\rho \\cdot s_Z(z) + \\psi_X(\\theta_X)."""
        params = model.initialize(key, location=0.0, shape=0.5)

        obs_params, int_params, _ = model.split_coords(params)
        lkl_params = model.lkl_fun_man.join_coords(obs_params, int_params)
        rho = model.conjugation_parameters(lkl_params)

        for i in range(3):
            z = jax.random.normal(jax.random.fold_in(key, i), (model.lat_man.data_dim,))
            s_z = model.lat_man.sufficient_statistic(z)

            # LHS: \psi(\theta_X + \theta_{XZ} \cdot s_Z(z))
            conditional_obs = model.lkl_fun_man(lkl_params, z)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            # RHS: \rho \cdot s_Z(z) + \psi_X(\theta_X)
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
# Fully Observed Markov Chain Tests
# =============================================================================


def sample_markov_trajectory(
    model: GaussianMarkovChain | CategoricalMarkovChain,
    params: Array,
    key: Array,
) -> Array:
    """Sample a trajectory in reverse chronological harmonium layout."""
    return model.sample(key, params, n=1)[0]


def split_markov_trajectory(
    model: GaussianMarkovChain | CategoricalMarkovChain,
    trajectory: Array,
) -> tuple[Array, Array]:
    """Split a trajectory into first-state and sub-chain blocks."""
    obs_dim = model.obs_man.data_dim
    return trajectory[:obs_dim], trajectory[obs_dim:]


def pack_markov_trajectory(states: Array) -> Array:
    """Pack chronological states ``[x_0, ..., x_T]`` into flat trajectory."""
    return states.reshape(-1)


def markov_process_recursive_log_density(
    model: GaussianMarkovChain | CategoricalMarkovChain,
    params: Array,
    trajectory: Array,
) -> Array:
    """Evaluate the recursive factorization ``p(x_0 | rest) p(rest)`` directly."""
    x0, rest = split_markov_trajectory(model, trajectory)
    log_prior = model.lat_man.log_density(model.prior(params), rest)
    log_obs = model.obs_man.log_density(model.likelihood_at(params, rest), x0)
    return log_prior + log_obs


class TestGaussianMarkovChain:
    """Test GaussianMarkovChain recursive harmonium structure."""

    @pytest.fixture
    def model(self) -> GaussianMarkovChain:
        return create_gaussian_markov_chain(lat_dim=2, n_steps=3)

    @pytest.fixture
    def base_model(self) -> NormalTransition:
        return NormalTransition(lat_dim=2)

    @pytest.fixture
    def one_step_model(self) -> GaussianMarkovChain:
        return create_gaussian_markov_chain(lat_dim=2, n_steps=1)

    def test_base_case_matches_transition(
        self,
        one_step_model: GaussianMarkovChain,
        base_model: NormalTransition,
        key: Array,
    ) -> None:
        params = one_step_model.initialize(key, location=0.0, shape=0.5)
        base_params = base_model.initialize(key, location=0.0, shape=0.5)
        trajectory = sample_markov_trajectory(
            one_step_model, params, jax.random.fold_in(key, 1)
        )
        x, z = split_markov_trajectory(one_step_model, trajectory)

        assert jnp.allclose(params, base_params, rtol=RTOL, atol=ATOL)
        assert one_step_model.dim == base_model.dim
        assert one_step_model.data_dim == base_model.data_dim
        assert jnp.allclose(
            one_step_model.sufficient_statistic(trajectory),
            base_model.sufficient_statistic(trajectory),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.log_base_measure(trajectory),
            base_model.log_base_measure(trajectory),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.prior(params),
            base_model.prior(base_params),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.likelihood_at(params, z),
            base_model.likelihood_at(base_params, z),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.posterior_at(params, x),
            base_model.posterior_at(base_params, x),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.log_partition_function(params),
            base_model.log_partition_function(base_params),
            rtol=RTOL,
            atol=ATOL,
        )
        means = one_step_model.to_mean(params)
        assert jnp.allclose(
            one_step_model.to_natural_likelihood(means),
            base_model.to_natural_likelihood(means),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.negative_entropy(means),
            base_model.negative_entropy(means),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_recursive_structure(self, model: GaussianMarkovChain) -> None:
        assert isinstance(model.lat_man, GaussianMarkovChain)
        assert model.lat_man.n_steps == model.n_steps - 1
        assert model.int_man.dim == model.trns_hrm.int_man.dim
        assert model.data_dim == (model.n_steps + 1) * model.obs_man.data_dim

    def test_split_join_roundtrip(self, model: GaussianMarkovChain, key: Array) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        obs, interaction, lat = model.split_coords(params)
        recovered = model.join_coords(obs, interaction, lat)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)

    def test_interaction_depends_only_on_sub_chain_observable(
        self, model: GaussianMarkovChain, key: Array
    ) -> None:
        """Interaction output should depend only on x_1 (observable of sub-chain)."""
        params = model.initialize(key, location=0.0, shape=0.5)
        _, int_params, _ = model.split_coords(params)

        # Random sub-chain sufficient statistics
        sub_stats = jax.random.normal(
            jax.random.fold_in(key, 1), (model.lat_man.dim,)
        )
        # x_1 stats are the observable portion of the sub-chain
        x1_stats = sub_stats[: model.obs_man.dim]

        # Full interaction and base interaction should match
        full_result = model.int_man(int_params, sub_stats)
        base_result = model.trns_hrm.int_man(int_params, x1_stats)
        assert jnp.allclose(full_result, base_result, rtol=RTOL, atol=ATOL)

    def test_sufficient_statistic_finite(
        self, model: GaussianMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        trajectory = sample_markov_trajectory(model, params, jax.random.fold_in(key, 1))
        suff_stats = model.sufficient_statistic(trajectory)

        assert suff_stats.shape == (model.dim,)
        assert jnp.all(jnp.isfinite(suff_stats))
        assert jnp.isfinite(model.log_base_measure(trajectory))

    def test_conjugation_equation(self, model: GaussianMarkovChain, key: Array) -> None:
        params = model.initialize(key, location=0.0, shape=0.3)
        lkl_params = model.likelihood_function(params)
        obs_params, _ = model.lkl_fun_man.split_coords(lkl_params)
        rho = model.conjugation_parameters(lkl_params)

        for i in range(3):
            # Sample a valid sub-chain trajectory as latent data
            z = jax.random.normal(
                jax.random.fold_in(key, i + 10), (model.lat_man.data_dim,)
            )
            z_stats = model.lat_man.sufficient_statistic(z)
            conditional_obs = model.lkl_fun_man(lkl_params, z_stats)
            lhs = model.obs_man.log_partition_function(conditional_obs)
            rhs = jnp.dot(rho, z_stats) + model.obs_man.log_partition_function(obs_params)
            assert jnp.abs(lhs - rhs) < ATOL

    def test_log_density_matches_recursive_factorization(
        self, model: GaussianMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.3)
        trajectory = sample_markov_trajectory(model, params, jax.random.fold_in(key, 2))

        log_density = model.log_density(params, trajectory)
        recursive_log_density = markov_process_recursive_log_density(
            model, params, trajectory
        )

        assert jnp.allclose(
            log_density, recursive_log_density, rtol=RTOL, atol=ATOL
        )

    def test_analytic_methods_finite(
        self, model: GaussianMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.2)
        means = model.to_mean(params)
        lkl_params = model.to_natural_likelihood(means)

        assert lkl_params.shape == (model.lkl_fun_man.dim,)
        assert jnp.all(jnp.isfinite(lkl_params))
        assert jnp.isfinite(model.negative_entropy(means))


class TestCategoricalMarkovChain:
    """Test CategoricalMarkovChain recursive harmonium structure."""

    @pytest.fixture
    def model(self) -> CategoricalMarkovChain:
        return create_categorical_markov_chain(n_states=3, n_steps=3)

    @pytest.fixture
    def base_model(self) -> CategoricalTransition:
        return CategoricalTransition(n_states=3)

    @pytest.fixture
    def one_step_model(self) -> CategoricalMarkovChain:
        return create_categorical_markov_chain(n_states=3, n_steps=1)

    def test_base_case_matches_transition(
        self,
        one_step_model: CategoricalMarkovChain,
        base_model: CategoricalTransition,
        key: Array,
    ) -> None:
        params = one_step_model.initialize(key, location=0.0, shape=0.5)
        base_params = base_model.initialize(key, location=0.0, shape=0.5)
        trajectory = sample_markov_trajectory(
            one_step_model, params, jax.random.fold_in(key, 3)
        )
        x, z = split_markov_trajectory(one_step_model, trajectory)

        assert jnp.allclose(params, base_params, rtol=RTOL, atol=ATOL)
        assert one_step_model.dim == base_model.dim
        assert one_step_model.data_dim == base_model.data_dim
        assert jnp.allclose(
            one_step_model.sufficient_statistic(trajectory),
            base_model.sufficient_statistic(trajectory),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.log_base_measure(trajectory),
            base_model.log_base_measure(trajectory),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.prior(params),
            base_model.prior(base_params),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.likelihood_at(params, z),
            base_model.likelihood_at(base_params, z),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.posterior_at(params, x),
            base_model.posterior_at(base_params, x),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.log_partition_function(params),
            base_model.log_partition_function(base_params),
            rtol=RTOL,
            atol=ATOL,
        )
        means = one_step_model.to_mean(params)
        assert jnp.allclose(
            one_step_model.to_natural_likelihood(means),
            base_model.to_natural_likelihood(means),
            rtol=RTOL,
            atol=ATOL,
        )
        assert jnp.allclose(
            one_step_model.negative_entropy(means),
            base_model.negative_entropy(means),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_recursive_structure(self, model: CategoricalMarkovChain) -> None:
        assert isinstance(model.lat_man, CategoricalMarkovChain)
        assert model.lat_man.n_steps == model.n_steps - 1
        assert model.int_man.dim == model.trns_hrm.int_man.dim
        assert model.data_dim == (model.n_steps + 1) * model.obs_man.data_dim

    def test_split_join_roundtrip(
        self, model: CategoricalMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        obs, interaction, lat = model.split_coords(params)
        recovered = model.join_coords(obs, interaction, lat)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)

    def test_interaction_depends_only_on_sub_chain_observable(
        self, model: CategoricalMarkovChain, key: Array
    ) -> None:
        """Interaction output should depend only on x_1 (observable of sub-chain)."""
        params = model.initialize(key, location=0.0, shape=0.5)
        _, int_params, _ = model.split_coords(params)

        # Random sub-chain sufficient statistics
        sub_stats = jax.random.normal(
            jax.random.fold_in(key, 1), (model.lat_man.dim,)
        )
        # x_1 stats are the observable portion of the sub-chain
        x1_stats = sub_stats[: model.obs_man.dim]

        # Full interaction and base interaction should match
        full_result = model.int_man(int_params, sub_stats)
        base_result = model.trns_hrm.int_man(int_params, x1_stats)
        assert jnp.allclose(full_result, base_result, rtol=RTOL, atol=ATOL)

    def test_sufficient_statistic_finite(
        self, model: CategoricalMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        trajectory = sample_markov_trajectory(model, params, jax.random.fold_in(key, 3))
        suff_stats = model.sufficient_statistic(trajectory)

        assert suff_stats.shape == (model.dim,)
        assert jnp.all(jnp.isfinite(suff_stats))
        assert jnp.isfinite(model.log_base_measure(trajectory))

    def test_conjugation_equation(
        self, model: CategoricalMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        lkl_params = model.likelihood_function(params)
        obs_params, _ = model.lkl_fun_man.split_coords(lkl_params)
        rho = model.conjugation_parameters(lkl_params)

        # Sample valid sub-chain trajectories as latent data
        sub_params = model.lat_man.initialize(
            jax.random.fold_in(key, 100), location=0.0, shape=0.5
        )
        for i in range(3):
            z = model.lat_man.sample(
                jax.random.fold_in(key, i + 10), sub_params, n=1
            )[0]
            z_stats = model.lat_man.sufficient_statistic(z)
            conditional_obs = model.lkl_fun_man(lkl_params, z_stats)
            lhs = model.obs_man.log_partition_function(conditional_obs)
            rhs = jnp.dot(rho, z_stats) + model.obs_man.log_partition_function(obs_params)
            assert jnp.abs(lhs - rhs) < ATOL

    def test_log_density_matches_recursive_factorization(
        self, model: CategoricalMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        trajectory = sample_markov_trajectory(model, params, jax.random.fold_in(key, 4))

        log_density = model.log_density(params, trajectory)
        recursive_log_density = markov_process_recursive_log_density(
            model, params, trajectory
        )

        assert jnp.allclose(
            log_density, recursive_log_density, rtol=RTOL, atol=ATOL
        )

    def test_analytic_methods_finite(
        self, model: CategoricalMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.3)
        means = model.to_mean(params)
        lkl_params = model.to_natural_likelihood(means)

        assert lkl_params.shape == (model.lkl_fun_man.dim,)
        assert jnp.all(jnp.isfinite(lkl_params))
        assert jnp.isfinite(model.negative_entropy(means))


class TestCategoricalMarkovChainBruteForce:
    """Brute-force validation of the recursive categorical chain density."""

    @pytest.fixture
    def model(self) -> CategoricalMarkovChain:
        return create_categorical_markov_chain(n_states=2, n_steps=3)

    @pytest.fixture
    def params(self, model: CategoricalMarkovChain, key: Array) -> Array:
        return model.initialize(key, location=0.0, shape=0.5)

    def test_log_density_normalizes(
        self, model: CategoricalMarkovChain, params: Array
    ) -> None:
        log_densities = []
        recursive_log_densities = []

        for seq in itertools.product(range(model.n_states), repeat=model.n_steps + 1):
            states = jnp.array(seq, dtype=jnp.int32)[:, None]
            trajectory = pack_markov_trajectory(states)
            log_densities.append(model.log_density(params, trajectory))
            recursive_log_densities.append(
                markov_process_recursive_log_density(model, params, trajectory)
            )

        log_densities_arr = jnp.array(log_densities)
        recursive_log_densities_arr = jnp.array(recursive_log_densities)
        total_probability = jnp.sum(jnp.exp(log_densities_arr))

        assert jnp.allclose(
            log_densities_arr,
            recursive_log_densities_arr,
            rtol=1e-10,
            atol=1e-10,
        )
        assert jnp.allclose(total_probability, 1.0, rtol=1e-10, atol=1e-10)


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
        """Test: \\psi(\\theta_X + \\theta_{XZ} \\cdot s_Z(z)) = \\rho \\cdot s_Z(z) + \\psi_X(\\theta_X).

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

            # LHS: \psi(\theta_X + \theta_{XZ} \cdot s_Z(z))
            conditional_obs = model.lkl_fun_man(lkl_params, s_z)
            lhs = model.obs_man.log_partition_function(conditional_obs)

            # RHS: \rho \cdot s_Z(z) + \psi_X(\theta_X)
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
        """Test: \\psi(\\theta_X + \\theta_{XZ} \\cdot s_Z(z)) = \\rho \\cdot s_Z(z) + \\psi_X(\\theta_X)."""
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
            smoothed_probs_generic, smoothed_probs_brute, rtol=1e-10, atol=1e-10
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
            log_lik_generic, log_lik_brute, rtol=1e-10, atol=1e-10
        ), f"Mismatch: generic={log_lik_generic}, brute={log_lik_brute}"


# =============================================================================
# Two-Slice Joint Validation
# =============================================================================


def brute_force_two_slice_joints(
    model: HiddenMarkovModel,
    params: Array,
    observations: Array,
) -> Array:
    """Compute exact two-slice joint marginals p(z_{t-1}, z_t | x_{1:T}).

    Returns mean parameters of the joint harmonium for each transition.

    Args:
        model: HMM model
        params: Model parameters
        observations: Shape (T, 1) - observation indices

    Returns:
        joints: Shape (T, joint_dim) - joint mean params for each (z_{t-1}, z_t)
    """
    n_steps = len(observations)
    n_states = model.n_states
    obs_flat = observations.flatten().astype(int)
    trns_hrm = model.trns_hrm

    # Enumerate all possible latent sequences
    all_seqs = list(itertools.product(range(n_states), repeat=n_steps + 1))
    log_joints = jnp.array([
        brute_force_log_joint(model, params, obs_flat, seq)
        for seq in all_seqs
    ])
    # Normalize
    log_Z = jax.scipy.special.logsumexp(log_joints)
    weights = jnp.exp(log_joints - log_Z)

    joints = []
    for t in range(n_steps):
        # Accumulate E[joint_stats] for the pair (z_t, z_{t+1}) in sequence indexing
        # z_t is at index t, z_{t+1} at index t+1
        joint_mean = jnp.zeros(trns_hrm.dim)
        for i, seq in enumerate(all_seqs):
            z_prev = jnp.array([seq[t]])
            z_curr = jnp.array([seq[t + 1]])
            # Compute joint sufficient statistics
            obs_stats = trns_hrm.obs_man.sufficient_statistic(z_curr)
            lat_stats = trns_hrm.lat_man.sufficient_statistic(z_prev)
            int_stats = trns_hrm.int_man.outer_product(obs_stats, lat_stats)
            joint_stats = trns_hrm.join_coords(obs_stats, int_stats, lat_stats)
            joint_mean = joint_mean + weights[i] * joint_stats
        joints.append(joint_mean)

    return jnp.array(joints)


class TestTwoSliceJointValidation:
    """Validate two-slice joint computation against brute-force for small HMMs."""

    @pytest.fixture
    def small_hmm(self) -> HiddenMarkovModel:
        return create_hmm(n_obs=2, n_states=3)

    @pytest.fixture
    def params(self, small_hmm: HiddenMarkovModel, key: Array) -> Array:
        return small_hmm.initialize(key, location=0.0, shape=0.5)

    @pytest.fixture
    def short_data(
        self, small_hmm: HiddenMarkovModel, params: Array, key: Array
    ) -> tuple[Array, Array]:
        key_sample = jax.random.fold_in(key, 200)
        return sample_latent_process(small_hmm, params, key_sample, n_steps=4)

    def test_joint_means_match_brute_force(
        self,
        small_hmm: HiddenMarkovModel,
        params: Array,
        short_data: tuple[Array, Array],
    ) -> None:
        """Two-slice joint mean params should match brute-force computation."""
        observations, _ = short_data

        _, _, joints = conjugated_smoothing0(small_hmm, params, observations)
        brute_joints = brute_force_two_slice_joints(small_hmm, params, observations)

        assert jnp.allclose(joints, brute_joints, rtol=1e-10, atol=1e-10), (
            f"Joint mismatch:\nGeneric:\n{joints}\nBrute:\n{brute_joints}"
        )

    def test_joint_marginals_match_smoothed(
        self,
        small_hmm: HiddenMarkovModel,
        params: Array,
        short_data: tuple[Array, Array],
    ) -> None:
        """Joint marginals should equal smoothed distributions."""
        observations, _ = short_data
        trns_hrm = small_hmm.trns_hrm
        lat_man = small_hmm.lat_man

        smoothed_z0, smoothed, joints = conjugated_smoothing0(
            small_hmm, params, observations
        )

        all_smoothed = jnp.concatenate([smoothed_z0[None, :], smoothed], axis=0)

        for t in range(len(observations)):
            _, _, lat_mean = trns_hrm.split_coords(joints[t])
            obs_mean = trns_hrm.split_coords(joints[t])[0]

            # lat marginal = E[s(z_{t-1})] should match smoothed z_{t-1}
            expected_lat = lat_man.to_mean(all_smoothed[t])
            assert jnp.allclose(lat_mean, expected_lat, rtol=1e-10, atol=1e-10), (
                f"Lat marginal mismatch at t={t}"
            )

            # obs marginal = E[s(z_t)] should match smoothed z_t
            expected_obs = lat_man.to_mean(all_smoothed[t + 1])
            assert jnp.allclose(obs_mean, expected_obs, rtol=1e-10, atol=1e-10), (
                f"Obs marginal mismatch at t={t}"
            )


# =============================================================================
# EM Learning Tests
# =============================================================================


class TestKalmanFilterEM:
    """Test EM learning for Kalman Filter."""

    @pytest.fixture
    def model(self) -> KalmanFilter:
        return create_kalman_filter(obs_dim=2, lat_dim=2)

    @pytest.fixture
    def true_params(self, model: KalmanFilter, key: Array) -> Array:
        return model.initialize(key, location=0.0, shape=0.3)

    @pytest.fixture
    def training_data(
        self, model: KalmanFilter, true_params: Array, key: Array
    ) -> Array:
        """Generate batch of training sequences."""
        key_sample = jax.random.fold_in(key, 300)
        keys = jax.random.split(key_sample, 5)

        def sample_one(k: Array) -> tuple[Array, Array]:
            return sample_latent_process(model, true_params, k, n_steps=20)

        observations_batch, _ = jax.vmap(sample_one)(keys)
        return observations_batch

    def test_em_monotonicity(
        self, model: KalmanFilter, true_params: Array, training_data: Array, key: Array  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        """Log-likelihood should be non-decreasing across EM steps."""
        init_params = model.initialize(
            jax.random.fold_in(key, 999), location=0.0, shape=0.5
        )

        params = init_params
        prev_ll = -jnp.inf

        for _ in range(8):
            params = latent_process_expectation_maximization(
                model, params, training_data
            )
            avg_ll = jnp.mean(
                jax.vmap(
                    lambda obs: latent_process_log_observable_density(
                        model, params, obs
                    )
                )(training_data)
            )
            assert jnp.isfinite(avg_ll), "Log-likelihood is NaN or Inf"
            assert avg_ll >= prev_ll - 0.1, (
                f"EM not monotone: {float(avg_ll):.4f} < {float(prev_ll):.4f}"
            )
            prev_ll = avg_ll

    def test_em_improves_likelihood(
        self, model: KalmanFilter, true_params: Array, training_data: Array, key: Array  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        """EM should improve log-likelihood over initial parameters."""
        init_params = model.initialize(
            jax.random.fold_in(key, 999), location=0.0, shape=0.5
        )

        initial_ll = float(
            jnp.mean(
                jax.vmap(
                    lambda obs: latent_process_log_observable_density(
                        model, init_params, obs
                    )
                )(training_data)
            )
        )

        params = init_params
        for _ in range(10):
            params = latent_process_expectation_maximization(
                model, params, training_data
            )

        final_ll = float(
            jnp.mean(
                jax.vmap(
                    lambda obs: latent_process_log_observable_density(
                        model, params, obs
                    )
                )(training_data)
            )
        )

        assert final_ll > initial_ll, (
            f"EM did not improve: {final_ll:.4f} <= {initial_ll:.4f}"
        )


class TestHMMEM:
    """Test EM learning for Hidden Markov Model."""

    @pytest.fixture
    def model(self) -> HiddenMarkovModel:
        return create_hmm(n_obs=2, n_states=3)

    @pytest.fixture
    def true_params(self, model: HiddenMarkovModel, key: Array) -> Array:
        return model.initialize(key, location=0.0, shape=0.5)

    @pytest.fixture
    def training_data(
        self, model: HiddenMarkovModel, true_params: Array, key: Array
    ) -> Array:
        """Generate batch of training sequences."""
        key_sample = jax.random.fold_in(key, 400)
        keys = jax.random.split(key_sample, 5)

        def sample_one(k: Array) -> tuple[Array, Array]:
            return sample_latent_process(model, true_params, k, n_steps=15)

        observations_batch, _ = jax.vmap(sample_one)(keys)
        return observations_batch

    def test_em_monotonicity(
        self,
        model: HiddenMarkovModel,
        true_params: Array,  # pyright: ignore[reportUnusedParameter]
        training_data: Array,
        key: Array,
    ) -> None:
        """Log-likelihood should be non-decreasing across EM steps."""
        init_params = model.initialize(
            jax.random.fold_in(key, 999), location=0.0, shape=0.5
        )

        params = init_params
        prev_ll = -jnp.inf

        for _ in range(8):
            params = latent_process_expectation_maximization(
                model, params, training_data
            )
            avg_ll = jnp.mean(
                jax.vmap(
                    lambda obs: latent_process_log_observable_density(
                        model, params, obs
                    )
                )(training_data)
            )
            assert jnp.isfinite(avg_ll), "Log-likelihood is NaN or Inf"
            assert avg_ll >= prev_ll - 0.1, (
                f"EM not monotone: {float(avg_ll):.4f} < {float(prev_ll):.4f}"
            )
            prev_ll = avg_ll

    def test_em_improves_likelihood(
        self,
        model: HiddenMarkovModel,
        true_params: Array,  # pyright: ignore[reportUnusedParameter]
        training_data: Array,
        key: Array,
    ) -> None:
        """EM should improve log-likelihood over initial parameters."""
        # Use a distant initialization (shape=2.0) so EM has room to improve
        init_params = model.initialize(
            jax.random.fold_in(key, 999), location=0.0, shape=2.0
        )

        initial_ll = float(
            jnp.mean(
                jax.vmap(
                    lambda obs: latent_process_log_observable_density(
                        model, init_params, obs
                    )
                )(training_data)
            )
        )

        params = init_params
        for _ in range(10):
            params = latent_process_expectation_maximization(
                model, params, training_data
            )

        final_ll = float(
            jnp.mean(
                jax.vmap(
                    lambda obs: latent_process_log_observable_density(
                        model, params, obs
                    )
                )(training_data)
            )
        )

        assert final_ll > initial_ll, (
            f"EM did not improve: {final_ll:.4f} <= {initial_ll:.4f}"
        )


# =============================================================================
# Roundtrip Tests: to_mean → to_natural
# =============================================================================


class TestNormalTransitionRoundtrip:
    """Test NormalTransition to_mean/to_natural roundtrip."""

    @pytest.fixture
    def model(self) -> NormalTransition:
        return NormalTransition(lat_dim=3)

    def test_to_mean_to_natural_roundtrip(self, model: NormalTransition, key: Array) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)


class TestNormalEmissionRoundtrip:
    """Test NormalEmission to_mean/to_natural roundtrip."""

    @pytest.fixture
    def model(self) -> NormalEmission:
        return NormalEmission(obs_dim=2, lat_dim=3)

    def test_to_mean_to_natural_roundtrip(self, model: NormalEmission, key: Array) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)


class TestGaussianMarkovChainRoundtrip:
    """Test GaussianMarkovChain to_mean/to_natural roundtrip."""

    @pytest.fixture
    def model(self) -> GaussianMarkovChain:
        return create_gaussian_markov_chain(lat_dim=2, n_steps=1)

    def test_to_mean_to_natural_roundtrip(self, model: GaussianMarkovChain, key: Array) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)


class TestCategoricalTransitionRoundtrip:
    """Test CategoricalTransition to_mean/to_natural roundtrip."""

    @pytest.fixture
    def model(self) -> CategoricalTransition:
        return CategoricalTransition(n_states=3)

    def test_to_mean_to_natural_roundtrip(
        self, model: CategoricalTransition, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)


class TestCategoricalEmissionRoundtrip:
    """Test CategoricalEmission to_mean/to_natural roundtrip."""

    @pytest.fixture
    def model(self) -> CategoricalEmission:
        return CategoricalEmission(n_obs=4, _n_states=3)

    def test_to_mean_to_natural_roundtrip(
        self, model: CategoricalEmission, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)


class TestCategoricalMarkovChainRoundtrip:
    """Test CategoricalMarkovChain to_mean/to_natural roundtrip."""

    @pytest.fixture
    def model(self) -> CategoricalMarkovChain:
        return create_categorical_markov_chain(n_states=3, n_steps=1)

    def test_to_mean_to_natural_roundtrip(
        self, model: CategoricalMarkovChain, key: Array
    ) -> None:
        params = model.initialize(key, location=0.0, shape=0.5)
        means = model.to_mean(params)
        recovered = model.to_natural(means)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)
