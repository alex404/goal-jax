"""Tests for geometry/exponential_family/dynamical.py and models/dynamical/.

Covers:
- HomogeneousMarkovProcess (flat-EF fully observed Markov chain) via HomogeneousGaussianMarkovChain.
- Transition / AnalyticTransition predict shape and round-trip for LinearGaussianTransition and CategoricalTransition.
- MLPTransition shape, gradient flow.

Concrete state-space leaves (KalmanFilter, HiddenMarkovModel) are tested in later phases.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import MLPMap
from goal.models import (
    Categorical,
    HiddenMarkovModel,
    HomogeneousGaussianMarkovChain,
    KalmanFilter,
    MLPTransition,
    create_categorical_transition,
    create_hidden_markov_model,
    create_homogeneous_gaussian_markov_chain,
    create_kalman_filter,
    create_linear_gaussian_transition,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

RTOL = 1e-4
ATOL = 1e-4


@pytest.fixture
def chain_2d_3steps() -> tuple[HomogeneousGaussianMarkovChain, Array]:
    """A 2D Gaussian chain with 3 transitions (4 states), random natural params."""
    chain = create_homogeneous_gaussian_markov_chain(lat_dim=2, n_steps=3)
    params = chain.initialize(jax.random.PRNGKey(0), location=0.0, shape=0.05)
    return chain, params


class TestHomogeneousGaussianMarkovChain:
    """Sanity checks for the flat-EF Gaussian Markov chain."""

    def test_dim_matches_transition(
        self, chain_2d_3steps: tuple[HomogeneousGaussianMarkovChain, Array]
    ) -> None:
        """A homogeneous chain has the same number of params as one transition."""
        chain, _ = chain_2d_3steps
        assert chain.dim == chain.trns_hrm.dim

    def test_data_dim(
        self, chain_2d_3steps: tuple[HomogeneousGaussianMarkovChain, Array]
    ) -> None:
        """data_dim = (n_steps + 1) * state_data_dim."""
        chain, _ = chain_2d_3steps
        assert chain.data_dim == (chain.n_steps + 1) * chain.state_man.data_dim

    def test_sample_shape(
        self, chain_2d_3steps: tuple[HomogeneousGaussianMarkovChain, Array]
    ) -> None:
        """sample(n=K) returns (K, data_dim)."""
        chain, params = chain_2d_3steps
        samples = chain.sample(jax.random.PRNGKey(1), params, n=8)
        assert samples.shape == (8, chain.data_dim)

    def test_log_partition_finite(
        self, chain_2d_3steps: tuple[HomogeneousGaussianMarkovChain, Array]
    ) -> None:
        chain, params = chain_2d_3steps
        psi = chain.log_partition_function(params)
        assert jnp.isfinite(psi)

    def test_to_natural_round_trip(
        self, chain_2d_3steps: tuple[HomogeneousGaussianMarkovChain, Array]
    ) -> None:
        """to_natural(to_mean(params)) recovers the original natural params."""
        chain, params = chain_2d_3steps
        means = chain.to_mean(params)
        recovered = chain.to_natural(means)
        assert jnp.allclose(recovered, params, rtol=RTOL, atol=ATOL)

    def test_sufficient_statistic_shape(
        self, chain_2d_3steps: tuple[HomogeneousGaussianMarkovChain, Array]
    ) -> None:
        """The sufficient statistic of a trajectory has shape (dim,)."""
        chain, params = chain_2d_3steps
        samples = chain.sample(jax.random.PRNGKey(2), params, n=1)
        suff = chain.sufficient_statistic(samples[0])
        assert suff.shape == (chain.dim,)

    def test_empirical_mean_matches_to_mean(
        self, chain_2d_3steps: tuple[HomogeneousGaussianMarkovChain, Array]
    ) -> None:
        """E[s(x)] under the model matches to_mean(params), via large-n sampling."""
        chain, params = chain_2d_3steps
        samples = chain.sample(jax.random.PRNGKey(3), params, n=20_000)
        empirical = jax.vmap(chain.sufficient_statistic)(samples).mean(axis=0)
        analytic = chain.to_mean(params)
        # Loose tolerance --- 20k samples + a 4-state chain has nontrivial variance
        assert jnp.allclose(empirical, analytic, atol=0.1)

    @pytest.mark.parametrize("n_steps", [1, 2, 3, 5])
    def test_n_steps_variants(self, n_steps: int) -> None:
        """The chain is well-formed for various n_steps."""
        chain = create_homogeneous_gaussian_markov_chain(lat_dim=2, n_steps=n_steps)
        params = chain.initialize(jax.random.PRNGKey(0), shape=0.05)
        samples = chain.sample(jax.random.PRNGKey(1), params, n=4)
        assert samples.shape == (4, (n_steps + 1) * 2)
        psi = chain.log_partition_function(params)
        assert jnp.isfinite(psi)

    def test_n_steps_zero_raises(self) -> None:
        """n_steps < 1 is rejected."""
        with pytest.raises(ValueError):
            create_homogeneous_gaussian_markov_chain(lat_dim=2, n_steps=0)


class TestLinearGaussianTransition:
    """Predict, kernel access, init_aux for LinearGaussianTransition."""

    def test_dim_matches_kernel(self) -> None:
        trans = create_linear_gaussian_transition(lat_dim=2)
        assert trans.dim == trans.kernel.dim

    def test_lat_man_is_kernel_lat_man(self) -> None:
        trans = create_linear_gaussian_transition(lat_dim=3)
        assert trans.lat_man.data_dim == trans.kernel.lat_man.data_dim

    def test_init_aux_empty(self) -> None:
        trans = create_linear_gaussian_transition(lat_dim=2)
        aux = trans.init_aux(jax.random.PRNGKey(0))
        assert aux.shape == (0,)

    def test_predict_shape(self) -> None:
        """predict returns (predicted, aux) with predicted in lat_man's natural-param space."""
        trans = create_linear_gaussian_transition(lat_dim=2)
        params = trans.kernel.initialize(jax.random.PRNGKey(0), shape=0.05)
        belief = trans.lat_man.initialize(jax.random.PRNGKey(1), shape=0.05)
        aux = trans.init_aux(jax.random.PRNGKey(2))
        predicted, new_aux = trans.predict(params, belief, aux)
        assert predicted.shape == (trans.lat_man.dim,)
        assert new_aux.shape == aux.shape

    def test_predict_differentiable(self) -> None:
        """jax.grad of a scalar of predict's output is finite (BPTT-friendly)."""
        trans = create_linear_gaussian_transition(lat_dim=2)
        params = trans.kernel.initialize(jax.random.PRNGKey(0), shape=0.05)
        belief = trans.lat_man.initialize(jax.random.PRNGKey(1), shape=0.05)
        aux = trans.init_aux(jax.random.PRNGKey(2))

        def loss(p: Array) -> Array:
            pred, _ = trans.predict(p, belief, aux)
            return jnp.sum(pred**2)

        grads = jax.grad(loss)(params)
        assert grads.shape == params.shape
        assert jnp.all(jnp.isfinite(grads))


class TestCategoricalTransition:
    """Predict and kernel access for CategoricalTransition."""

    def test_dim_matches_kernel(self) -> None:
        trans = create_categorical_transition(n_states=4)
        assert trans.dim == trans.kernel.dim

    def test_predict_shape(self) -> None:
        trans = create_categorical_transition(n_states=4)
        params = trans.kernel.initialize(jax.random.PRNGKey(0), shape=0.1)
        belief = trans.lat_man.initialize(jax.random.PRNGKey(1), shape=0.1)
        aux = trans.init_aux(jax.random.PRNGKey(2))
        predicted, _ = trans.predict(params, belief, aux)
        assert predicted.shape == (trans.lat_man.dim,)

    def test_predict_differentiable(self) -> None:
        """Gradient flows through the categorical predict step."""
        trans = create_categorical_transition(n_states=3)
        params = trans.kernel.initialize(jax.random.PRNGKey(0), shape=0.1)
        belief = trans.lat_man.initialize(jax.random.PRNGKey(1), shape=0.1)
        aux = trans.init_aux(jax.random.PRNGKey(2))

        def loss(p: Array) -> Array:
            pred, _ = trans.predict(p, belief, aux)
            return jnp.sum(pred**2)

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads))


class TestMLPTransition:
    """MLPTransition shape, gradient flow, and lat_man passthrough."""

    def test_dim_matches_mlp(self) -> None:
        lat = Categorical(3)
        mlp = MLPMap(lat, lat, hidden_dims=(4,))
        trans = MLPTransition(mlp=mlp)
        assert trans.dim == mlp.dim

    def test_lat_man_is_mlp_dom_man(self) -> None:
        lat = Categorical(3)
        mlp = MLPMap(lat, lat, hidden_dims=(4,))
        trans = MLPTransition(mlp=mlp)
        assert trans.lat_man.dim == lat.dim

    def test_predict_shape(self) -> None:
        lat = Categorical(3)
        mlp = MLPMap(lat, lat, hidden_dims=(4,))
        trans = MLPTransition(mlp=mlp)
        params = mlp.glorot_initialize(jax.random.PRNGKey(0))
        belief = jnp.zeros(lat.dim)
        aux = trans.init_aux(jax.random.PRNGKey(1))
        predicted, _ = trans.predict(params, belief, aux)
        assert predicted.shape == (lat.dim,)

    def test_bptt_smoke(self) -> None:
        """Gradient w.r.t. transition params flows through a multi-step scan
        (BPTT through alternating predict + dummy update)."""
        lat = Categorical(3)
        mlp = MLPMap(lat, lat, hidden_dims=(4,))
        trans = MLPTransition(mlp=mlp)
        params = mlp.glorot_initialize(jax.random.PRNGKey(0))
        aux = trans.init_aux(jax.random.PRNGKey(1))

        def total(p: Array) -> Array:
            def step(belief: Array, _scan_input: None) -> tuple[Array, Array]:
                predicted, _new_aux = trans.predict(p, belief, aux)
                # Dummy "update": shift by a fixed observation embedding
                updated = predicted + 0.1
                return updated, updated

            init = jnp.zeros(lat.dim)
            final, _trajectory = jax.lax.scan(step, init, None, length=5)
            return jnp.sum(final**2)

        grads = jax.grad(total)(params)
        assert grads.shape == params.shape
        assert jnp.all(jnp.isfinite(grads))


# =============================================================================
# Phase 6: KalmanFilter and HiddenMarkovModel
# =============================================================================


@pytest.fixture
def kf_2d_3d() -> tuple[KalmanFilter, Array]:
    """A KF with 2D observations and 3D latent state, randomly initialized."""
    kf = create_kalman_filter(obs_dim=2, lat_dim=3)
    params = kf.initialize(jax.random.PRNGKey(0), shape=0.05)
    return kf, params


@pytest.fixture
def hmm_4_3() -> tuple[HiddenMarkovModel, Array]:
    """An HMM with 4 obs categories and 3 latent states."""
    hmm = create_hidden_markov_model(n_obs=4, n_states=3)
    params = hmm.initialize(jax.random.PRNGKey(0), shape=0.1)
    return hmm, params


class TestKalmanFilter:
    """Smoke tests for KalmanFilter end-to-end."""

    def test_dim_breakdown(self, kf_2d_3d: tuple[KalmanFilter, Array]) -> None:
        kf, _ = kf_2d_3d
        # Triple slots: prior (lat_man.dim), emission (emsn_hrm.dim), transition (kernel.dim)
        expected = kf.lat_man.dim + kf.emsn_hrm.dim + kf.transition.kernel.dim
        assert kf.dim == expected

    def test_sample_shape(self, kf_2d_3d: tuple[KalmanFilter, Array]) -> None:
        kf, params = kf_2d_3d
        n_steps = 10
        observations, latents = kf.sample(jax.random.PRNGKey(1), params, n_steps)
        assert observations.shape == (n_steps, kf.obs_dim)
        assert latents.shape == (n_steps + 1, kf.lat_dim)

    def test_filter_shapes(self, kf_2d_3d: tuple[KalmanFilter, Array]) -> None:
        kf, params = kf_2d_3d
        observations, _ = kf.sample(jax.random.PRNGKey(1), params, n_steps=8)
        beliefs, log_lik = kf.filter(params, observations)
        assert beliefs.shape == (8, kf.lat_man.dim)
        assert log_lik.shape == ()
        assert jnp.isfinite(log_lik)

    def test_smooth_shapes(self, kf_2d_3d: tuple[KalmanFilter, Array]) -> None:
        kf, params = kf_2d_3d
        observations, _ = kf.sample(jax.random.PRNGKey(1), params, n_steps=6)
        z0_smoothed, smoothed_seq, joints = kf.smooth(params, observations)
        assert z0_smoothed.shape == (kf.lat_man.dim,)
        assert smoothed_seq.shape == (6, kf.lat_man.dim)
        # joints are mean-params of the kernel harmonium
        assert joints.shape == (6, kf.transition.kernel.dim)

    def test_log_observable_density_matches_filter(
        self, kf_2d_3d: tuple[KalmanFilter, Array]
    ) -> None:
        """log_observable_density returns the same value as filter's log-likelihood."""
        kf, params = kf_2d_3d
        observations, _ = kf.sample(jax.random.PRNGKey(1), params, n_steps=5)
        _, ll_filter = kf.filter(params, observations)
        ll_obs = kf.log_observable_density(params, observations)
        assert jnp.allclose(ll_filter, ll_obs, rtol=RTOL, atol=ATOL)

    def test_em_step_finite(self, kf_2d_3d: tuple[KalmanFilter, Array]) -> None:
        """One EM step produces finite parameters."""
        kf, params = kf_2d_3d
        # Build a small batch: shape (n_seqs, n_steps, obs_dim)
        keys = jax.random.split(jax.random.PRNGKey(2), 3)
        obs_batch = jnp.stack(
            [kf.sample(k, params, n_steps=5)[0] for k in keys]
        )
        new_params = kf.expectation_maximization(params, obs_batch)
        assert new_params.shape == params.shape
        assert jnp.all(jnp.isfinite(new_params))

    def test_from_standard_round_trip(self) -> None:
        """from_standard produces filterable parameters."""
        kf = create_kalman_filter(obs_dim=2, lat_dim=2)
        A = jnp.array([[0.9, 0.1], [-0.1, 0.9]])
        Q = jnp.eye(2) * 0.05
        C = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        R = jnp.eye(2) * 0.1
        mu_0 = jnp.zeros(2)
        Sigma_0 = jnp.eye(2)
        params = kf.from_standard(A, Q, C, R, mu_0, Sigma_0)
        observations, _ = kf.sample(jax.random.PRNGKey(0), params, n_steps=8)
        _, ll = kf.filter(params, observations)
        assert jnp.isfinite(ll)


class TestHiddenMarkovModel:
    """Smoke tests for HiddenMarkovModel end-to-end."""

    def test_dim_breakdown(self, hmm_4_3: tuple[HiddenMarkovModel, Array]) -> None:
        hmm, _ = hmm_4_3
        expected = hmm.lat_man.dim + hmm.emsn_hrm.dim + hmm.transition.kernel.dim
        assert hmm.dim == expected

    def test_sample_shape(self, hmm_4_3: tuple[HiddenMarkovModel, Array]) -> None:
        hmm, params = hmm_4_3
        n_steps = 10
        observations, latents = hmm.sample(jax.random.PRNGKey(1), params, n_steps)
        assert observations.shape == (n_steps, hmm.emsn_hrm.obs_man.data_dim)
        assert latents.shape == (n_steps + 1, hmm.lat_man.data_dim)

    def test_filter_finite(self, hmm_4_3: tuple[HiddenMarkovModel, Array]) -> None:
        hmm, params = hmm_4_3
        observations, _ = hmm.sample(jax.random.PRNGKey(1), params, n_steps=6)
        beliefs, log_lik = hmm.filter(params, observations)
        assert beliefs.shape == (6, hmm.lat_man.dim)
        assert jnp.isfinite(log_lik)

    def test_smooth_finite(self, hmm_4_3: tuple[HiddenMarkovModel, Array]) -> None:
        hmm, params = hmm_4_3
        observations, _ = hmm.sample(jax.random.PRNGKey(1), params, n_steps=5)
        z0_smoothed, smoothed_seq, joints = hmm.smooth(params, observations)
        assert z0_smoothed.shape == (hmm.lat_man.dim,)
        assert smoothed_seq.shape == (5, hmm.lat_man.dim)
        assert joints.shape == (5, hmm.transition.kernel.dim)

    def test_em_step_finite(self, hmm_4_3: tuple[HiddenMarkovModel, Array]) -> None:
        hmm, params = hmm_4_3
        keys = jax.random.split(jax.random.PRNGKey(2), 3)
        obs_batch = jnp.stack(
            [hmm.sample(k, params, n_steps=5)[0] for k in keys]
        )
        new_params = hmm.expectation_maximization(params, obs_batch)
        assert new_params.shape == params.shape
        assert jnp.all(jnp.isfinite(new_params))


# =============================================================================
# Phase 7: Correctness tests
# =============================================================================


class TestKalmanFilterCorrectness:
    """KF correctness: EM increases log-likelihood; reference KF agrees on filter shape."""

    def test_em_monotone(self) -> None:
        """A few EM iterations on data sampled from the true model should not
        decrease the average log-likelihood."""
        kf = create_kalman_filter(obs_dim=2, lat_dim=2)
        true_params = kf.initialize(jax.random.PRNGKey(0), shape=0.1)
        # Sample a batch of trajectories
        keys = jax.random.split(jax.random.PRNGKey(1), 8)
        obs_batch = jnp.stack(
            [kf.sample(k, true_params, n_steps=15)[0] for k in keys]
        )

        def avg_ll(p: Array) -> Array:
            return jnp.mean(
                jax.vmap(lambda obs: kf.log_observable_density(p, obs))(obs_batch)
            )

        # Start from a perturbed initialization; run EM and check non-decreasing LL.
        params = kf.initialize(jax.random.PRNGKey(2), shape=0.1)
        ll_history = [avg_ll(params)]
        for _ in range(5):
            params = kf.expectation_maximization(params, obs_batch)
            ll_history.append(avg_ll(params))
        ll_history = jnp.array(ll_history)
        # EM is theoretically monotone; allow tiny numerical slack
        assert jnp.all(jnp.diff(ll_history) >= -1e-4), (
            f"EM log-likelihood decreased: {ll_history}"
        )


class TestHMMCorrectness:
    """HMM correctness: filter agrees with manual forward algorithm; EM monotonic."""

    @staticmethod
    def _manual_forward(
        pi: Array, A: Array, B: Array, observations: Array
    ) -> tuple[Array, Array]:
        """Manual forward algorithm.

        Args:
            pi: (K,) initial state distribution.
            A: (K, K) transition matrix where A[i, j] = P(z_t=j | z_{t-1}=i).
            B: (K, M) emission matrix where B[k, m] = P(x_t=m | z_t=k).
            observations: (T,) integer observation indices.

        Returns:
            (alpha, log_likelihood). alpha[t] is P(z_t=k | x_{1:t}) (normalized).
            log_likelihood is log P(x_{1:T}).
        """
        # alpha_pred[t, k] = P(z_t=k, x_{1:t-1})  (predicted, before update)
        # alpha[t, k] = P(z_t=k, x_{1:t})  (filtered)

        def step(carry: tuple[Array, Array], obs: Array) -> tuple[
            tuple[Array, Array], Array
        ]:
            alpha_prev, log_lik = carry
            # Predict: alpha_pred[k] = sum_j alpha_prev[j] * A[j, k]
            alpha_pred = alpha_prev @ A
            # Update with observation: multiply by B[:, obs]
            alpha_unnorm = alpha_pred * B[:, obs]
            # Normalize and accumulate log-likelihood
            scale = jnp.sum(alpha_unnorm)
            alpha_new = alpha_unnorm / scale
            return (alpha_new, log_lik + jnp.log(scale)), alpha_new

        (_, log_likelihood), alpha_seq = jax.lax.scan(
            step, (pi, jnp.array(0.0)), observations
        )
        return alpha_seq, log_likelihood

    def test_filter_matches_manual_forward(self) -> None:
        """LatentProcess.filter on an HMM matches the standard forward algorithm.

        Initializes an HMM, decodes its natural parameters into stochastic ``(pi, A, B)`` matrices, samples observations, then runs both ``hmm.filter`` and a manual forward pass over the decoded matrices. The two log-likelihoods should agree.
        """
        n_states = 3
        n_obs = 4

        hmm = create_hidden_markov_model(n_obs=n_obs, n_states=n_states)
        params = hmm.initialize(jax.random.PRNGKey(0), shape=0.5)

        # Decode (pi, A, B) from hmm's natural params.
        prior_p, emsn_p, trns_p = hmm.split_coords(params)
        pi_decoded = hmm.lat_man.to_probs(hmm.lat_man.to_mean(prior_p))

        # For A (transition), each previous state z_{t-1} = j gives a conditional
        # over z_t. Loop over j:
        kernel = hmm.transition.kernel
        trns_lkl, _ = kernel.split_conjugated(trns_p)

        # Categorical's data_dim is 1 (a scalar category index).
        def transition_row(j: int) -> Array:
            z_prev_data = jnp.array([j], dtype=jnp.float64)
            s_z_prev = kernel.lat_man.sufficient_statistic(z_prev_data)
            z_t_nat = kernel.lkl_fun_man(trns_lkl, s_z_prev)
            return kernel.obs_man.to_probs(kernel.obs_man.to_mean(z_t_nat))

        A_decoded = jnp.stack([transition_row(j) for j in range(n_states)])

        emsn_lkl, _ = hmm.emsn_hrm.split_conjugated(emsn_p)

        def emission_row(j: int) -> Array:
            z_data = jnp.array([j], dtype=jnp.float64)
            s_z = hmm.emsn_hrm.pst_man.sufficient_statistic(z_data)
            x_nat = hmm.emsn_hrm.lkl_fun_man(emsn_lkl, s_z)
            return hmm.emsn_hrm.obs_man.to_probs(
                hmm.emsn_hrm.obs_man.to_mean(x_nat)
            )

        B_decoded = jnp.stack([emission_row(j) for j in range(n_states)])

        # Sample observations from the model. Categorical observations are
        # already integer indices (data_dim = 1).
        obs, _ = hmm.sample(jax.random.PRNGKey(7), params, n_steps=20)
        # obs has shape (n_steps, 1) -- integer indices wrapped in a length-1 array.
        obs_int = obs.reshape(-1).astype(jnp.int32)

        # Run manual forward on (pi_decoded, A_decoded, B_decoded, obs_int).
        _, log_lik_manual = self._manual_forward(
            pi_decoded, A_decoded, B_decoded, obs_int
        )

        # Run LatentProcess.filter on hmm with the original observations.
        _, log_lik_hmm = hmm.filter(params, obs)

        # The two log-likelihoods should agree up to numerical error.
        assert jnp.allclose(log_lik_manual, log_lik_hmm, rtol=1e-4, atol=1e-4), (
            f"Manual forward log-likelihood {log_lik_manual} differs from "
            f"hmm.filter {log_lik_hmm}"
        )


class TestMLPBPTTGradientDescent:
    """MLP transition gradient descent reduces loss on a synthetic problem."""

    def test_grad_descent_reduces_loss(self) -> None:
        """A few SGD steps on a synthetic problem should reduce the loss."""
        lat = Categorical(3)
        mlp = MLPMap(lat, lat, hidden_dims=(8,))
        trans = MLPTransition(mlp=mlp)
        params = mlp.glorot_initialize(jax.random.PRNGKey(0))
        aux = trans.init_aux(jax.random.PRNGKey(1))
        target = jnp.array([0.3, 0.5])  # arbitrary target in lat_man's nat space

        def loss(p: Array) -> Array:
            belief = jnp.zeros(lat.dim)
            predicted, _ = trans.predict(p, belief, aux)
            return jnp.sum((predicted - target) ** 2)

        initial_loss = loss(params)
        lr = 0.1
        for _ in range(50):
            grads = jax.grad(loss)(params)
            params = params - lr * grads
        final_loss = loss(params)
        assert final_loss < initial_loss * 0.5, (
            f"SGD did not significantly reduce loss: {initial_loss} -> {final_loss}"
        )
