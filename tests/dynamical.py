"""Tests for geometry/exponential_family/dynamical.py and models/dynamical/.

Covers:
- ``AnalyticTransition`` over both kernel types: predict shape, BPTT gradient.
- ``KalmanFilter`` end-to-end (sample, filter, smooth, from_standard) and EM monotonicity.
- ``HiddenMarkovModel`` end-to-end (sample, filter, smooth, EM) and a manual-forward correctness cross-check.
- ``MultilayerPerceptron[L, L]`` as a transition map: BPTT gradient descent reduces loss.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from goal.geometry import MultilayerPerceptron, PositiveDefinite
from goal.geometry.exponential_family.dynamical import AnalyticTransition
from goal.models import (
    AnalyticMixture,
    Categorical,
    HiddenMarkovModel,
    KalmanFilter,
    NormalAnalyticLGM,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


_TRANSITIONS = [
    AnalyticTransition(
        kernel=NormalAnalyticLGM(obs_dim=2, obs_rep=PositiveDefinite(), lat_dim=2)
    ),
    AnalyticTransition(kernel=AnalyticMixture(Categorical(4), 4)),
]
_TRANSITION_IDS = ["gaussian", "categorical"]


class TestAnalyticTransition:
    """``AnalyticTransition`` parameterized over Gaussian and Categorical kernels."""

    @pytest.mark.parametrize("trans", _TRANSITIONS, ids=_TRANSITION_IDS)
    def test_dim_is_kernel_likelihood(self, trans) -> None:
        """``dim`` is the kernel's likelihood-only dim, not the full harmonium."""
        assert trans.dim == trans.kernel.lkl_fun_man.dim
        assert trans.dim < trans.kernel.dim

    @pytest.mark.parametrize("trans", _TRANSITIONS, ids=_TRANSITION_IDS)
    def test_predict_shape(self, trans) -> None:
        params = trans.initialize(jax.random.PRNGKey(0), shape=0.05)
        belief = trans.dom_man.initialize(jax.random.PRNGKey(1), shape=0.05)
        assert params.shape == (trans.dim,)
        assert trans(params, belief).shape == (trans.dom_man.dim,)

    @pytest.mark.parametrize("trans", _TRANSITIONS, ids=_TRANSITION_IDS)
    def test_predict_differentiable(self, trans) -> None:
        params = trans.initialize(jax.random.PRNGKey(0), shape=0.05)
        belief = trans.dom_man.initialize(jax.random.PRNGKey(1), shape=0.05)
        grads = jax.grad(lambda p: jnp.sum(trans(p, belief) ** 2))(params)
        assert jnp.all(jnp.isfinite(grads))


class TestKalmanFilter:
    """End-to-end smoke tests plus EM monotonicity and ``from_standard``."""

    def test_sample_shape(self) -> None:
        kf = KalmanFilter(obs_dim=2, lat_dim=3)
        params = kf.initialize(jax.random.PRNGKey(0), shape=0.05)
        obs, lat = kf.sample(jax.random.PRNGKey(1), params, n_steps=10)
        assert obs.shape == (10, kf.obs_dim)
        assert lat.shape == (11, kf.lat_dim)

    def test_filter_matches_log_observable(self) -> None:
        """``filter``'s log-likelihood agrees with ``log_observable_density``."""
        kf = KalmanFilter(obs_dim=2, lat_dim=3)
        params = kf.initialize(jax.random.PRNGKey(0), shape=0.05)
        obs, _ = kf.sample(jax.random.PRNGKey(1), params, n_steps=8)
        beliefs, ll_filter = kf.filter(params, obs)
        ll_obs = kf.log_observable_density(params, obs)
        assert beliefs.shape == (8, kf.lat_man.dim)
        assert jnp.allclose(ll_filter, ll_obs, rtol=1e-5, atol=1e-7)

    def test_smooth_shapes(self) -> None:
        kf = KalmanFilter(obs_dim=2, lat_dim=3)
        params = kf.initialize(jax.random.PRNGKey(0), shape=0.05)
        obs, _ = kf.sample(jax.random.PRNGKey(1), params, n_steps=6)
        z0, smoothed, joints = kf.smooth(params, obs)
        assert z0.shape == (kf.lat_man.dim,)
        assert smoothed.shape == (6, kf.lat_man.dim)
        assert joints.shape == (6, kf.trn_map.kernel.dim)

    def test_from_standard_filters(self) -> None:
        kf = KalmanFilter(obs_dim=2, lat_dim=2)
        A = jnp.array([[0.9, 0.1], [-0.1, 0.9]])
        Q = 0.05 * jnp.eye(2)
        C = jnp.eye(2)
        R = 0.1 * jnp.eye(2)
        params = kf.from_standard(A, Q, C, R, jnp.zeros(2), jnp.eye(2))
        obs, _ = kf.sample(jax.random.PRNGKey(0), params, n_steps=8)
        _, ll = kf.filter(params, obs)
        assert jnp.isfinite(ll)

    def test_em_monotone(self) -> None:
        """EM iterations on data sampled from the true model should not decrease average log-likelihood."""
        kf = KalmanFilter(obs_dim=2, lat_dim=2)
        true_params = kf.initialize(jax.random.PRNGKey(0), shape=0.1)
        keys = jax.random.split(jax.random.PRNGKey(1), 8)
        obs_batch = jnp.stack([kf.sample(k, true_params, n_steps=15)[0] for k in keys])

        def avg_ll(p: Array) -> Array:
            return jnp.mean(
                jax.vmap(lambda obs: kf.log_observable_density(p, obs))(obs_batch)
            )

        params = kf.initialize(jax.random.PRNGKey(2), shape=0.1)
        ll = [avg_ll(params)]
        for _ in range(5):
            params = kf.expectation_maximization(params, obs_batch)
            ll.append(avg_ll(params))
        ll = jnp.array(ll)
        assert jnp.all(jnp.diff(ll) >= -1e-4), f"EM log-likelihood decreased: {ll}"


class TestHiddenMarkovModel:
    """End-to-end smoke tests plus filter-vs-manual-forward correctness."""

    def test_sample_shape(self) -> None:
        hmm = HiddenMarkovModel(n_obs=4, n_states=3)
        params = hmm.initialize(jax.random.PRNGKey(0), shape=0.1)
        obs, lat = hmm.sample(jax.random.PRNGKey(1), params, n_steps=10)
        assert obs.shape == (10, hmm.ems_hrm.obs_man.data_dim)
        assert lat.shape == (11, hmm.lat_man.data_dim)

    def test_filter_finite(self) -> None:
        hmm = HiddenMarkovModel(n_obs=4, n_states=3)
        params = hmm.initialize(jax.random.PRNGKey(0), shape=0.1)
        obs, _ = hmm.sample(jax.random.PRNGKey(1), params, n_steps=6)
        beliefs, ll = hmm.filter(params, obs)
        assert beliefs.shape == (6, hmm.lat_man.dim)
        assert jnp.isfinite(ll)

    def test_smooth_shapes(self) -> None:
        hmm = HiddenMarkovModel(n_obs=4, n_states=3)
        params = hmm.initialize(jax.random.PRNGKey(0), shape=0.1)
        obs, _ = hmm.sample(jax.random.PRNGKey(1), params, n_steps=5)
        z0, smoothed, joints = hmm.smooth(params, obs)
        assert z0.shape == (hmm.lat_man.dim,)
        assert smoothed.shape == (5, hmm.lat_man.dim)
        assert joints.shape == (5, hmm.trn_map.kernel.dim)

    def test_em_step_finite(self) -> None:
        hmm = HiddenMarkovModel(n_obs=4, n_states=3)
        params = hmm.initialize(jax.random.PRNGKey(0), shape=0.1)
        keys = jax.random.split(jax.random.PRNGKey(2), 3)
        obs_batch = jnp.stack([hmm.sample(k, params, n_steps=5)[0] for k in keys])
        new_params = hmm.expectation_maximization(params, obs_batch)
        assert jnp.all(jnp.isfinite(new_params))

    def test_filter_matches_manual_forward(self) -> None:
        """``hmm.filter`` matches the standard forward algorithm.

        Decodes $(\\pi, A, B)$ from natural parameters, samples observations, runs both ``hmm.filter`` and a manual forward pass, and compares log-likelihoods.
        """
        n_states, n_obs = 3, 4
        hmm = HiddenMarkovModel(n_obs=n_obs, n_states=n_states)
        params = hmm.initialize(jax.random.PRNGKey(0), shape=0.5)

        prior_p, ems_p, trns_p = hmm.split_coords(params)
        pi = hmm.lat_man.to_probs(hmm.lat_man.to_mean(prior_p))

        def decoded_rows(lkl_params: Array, hrm) -> Array:
            def row(j: int) -> Array:
                s = hrm.lat_man.sufficient_statistic(
                    jnp.array([j], dtype=jnp.float64)
                )
                nat = hrm.lkl_fun_man(lkl_params, s)
                return hrm.obs_man.to_probs(hrm.obs_man.to_mean(nat))

            return jnp.stack([row(j) for j in range(n_states)])

        A = decoded_rows(trns_p, hmm.trn_map.kernel)
        B = decoded_rows(ems_p, hmm.ems_hrm)

        obs, _ = hmm.sample(jax.random.PRNGKey(7), params, n_steps=20)
        obs_int = obs.reshape(-1).astype(jnp.int32)

        def step(
            carry: tuple[Array, Array], o: Array
        ) -> tuple[tuple[Array, Array], None]:
            alpha_prev, ll = carry
            alpha_unnorm = (alpha_prev @ A) * B[:, o]
            scale = jnp.sum(alpha_unnorm)
            return (alpha_unnorm / scale, ll + jnp.log(scale)), None

        (_, ll_manual), _ = jax.lax.scan(step, (pi, jnp.array(0.0)), obs_int)
        _, ll_hmm = hmm.filter(params, obs)
        assert jnp.allclose(ll_manual, ll_hmm, rtol=1e-4, atol=1e-4)


class TestMLPTransition:
    """``MultilayerPerceptron[L, L]`` plugged into a transition slot drives loss down under SGD (BPTT-friendly)."""

    def test_grad_descent_reduces_loss(self) -> None:
        lat = Categorical(3)
        trans = MultilayerPerceptron(
            lat, lat, hidden_dims=(8,), activation=jax.nn.relu
        )
        params = trans.glorot_initialize(jax.random.PRNGKey(0))
        target = jnp.array([0.3, 0.5])

        def loss(p: Array) -> Array:
            return jnp.sum((trans(p, jnp.zeros(lat.dim)) - target) ** 2)

        initial = loss(params)
        for _ in range(50):
            params = params - 0.1 * jax.grad(loss)(params)
        assert loss(params) < initial * 0.5
