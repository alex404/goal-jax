"""Tests for geometry/exponential_family/dynamical.py and models/dynamical.

The forward filter's log-likelihood is checked against independent ground truth: for the Kalman filter, the dense joint normal of the whole observation sequence built from the standard parameters $(A, Q, C, R, \\mu_0, \\Sigma_0)$; for the hidden Markov model, the forward algorithm on the decoded probability tables. EM does not decrease the log-likelihood of either.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.scipy import stats

from goal.models import HiddenMarkovModel, KalmanFilter

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def sequence_moments(
    transition: Array,
    process_noise: Array,
    emission: Array,
    observation_noise: Array,
    prior_mean: Array,
    prior_covariance: Array,
    n_steps: int,
) -> tuple[Array, Array]:
    """Mean and covariance of $x_{1:T}$, flattened, for $z_t = A z_{t-1} + w_t$ and $x_t = C z_t + v_t$.

    With $P_t = A P_{t-1} A^\\top + Q$ the covariance of $z_t$, $\\mathrm{Cov}(z_t, z_s) = A^{t-s} P_s$ for $t \\geq s$.
    """
    means, covs = [], []
    mean, cov = prior_mean, prior_covariance
    for _ in range(n_steps):
        mean = transition @ mean
        cov = transition @ cov @ transition.T + process_noise
        means.append(mean)
        covs.append(cov)

    lat_cross = [[jnp.zeros_like(prior_covariance)] * n_steps for _ in range(n_steps)]
    for s in range(n_steps):
        block = covs[s]
        for t in range(s, n_steps):
            lat_cross[t][s] = block
            lat_cross[s][t] = block.T
            block = transition @ block

    obs_mean = jnp.concatenate([emission @ m for m in means])
    obs_cov = jnp.block(
        [
            [
                emission @ lat_cross[t][s] @ emission.T
                + (observation_noise if t == s else 0.0)
                for s in range(n_steps)
            ]
            for t in range(n_steps)
        ]
    )
    return obs_mean, obs_cov


class TestKalmanFilter:
    def test_log_likelihood_is_the_joint_normal(self) -> None:
        """The filter's $\\log p(x_{1:T})$ equals the log-density of the dense joint normal of the sequence."""
        kf = KalmanFilter(obs_dim=2, lat_dim=2)
        standard = (
            jnp.array([[0.9, 0.2], [-0.3, 0.8]]),
            jnp.array([[0.2, 0.05], [0.05, 0.1]]),
            jnp.array([[1.0, 0.5], [-0.4, 1.2]]),
            jnp.array([[0.3, 0.1], [0.1, 0.2]]),
            jnp.array([0.5, -1.0]),
            jnp.array([[1.0, 0.3], [0.3, 0.5]]),
        )
        params = kf.from_standard(*standard)
        n_steps = 6
        obs, _ = kf.sample(jax.random.PRNGKey(0), params, n_steps=n_steps)

        mean, cov = sequence_moments(*standard, n_steps)
        expected = stats.multivariate_normal.logpdf(obs.ravel(), mean, cov)
        assert jnp.allclose(
            kf.log_observable_density(params, obs), expected, rtol=1e-5, atol=1e-7
        )


class TestHiddenMarkovModel:
    def test_log_likelihood_is_the_forward_algorithm(self) -> None:
        """The filter's $\\log p(x_{1:T})$ equals the forward algorithm on $(\\pi, A, B)$ decoded from the natural parameters."""
        n_states, n_obs = 3, 4
        hmm = HiddenMarkovModel(n_obs=n_obs, n_states=n_states)
        params = hmm.initialize(jax.random.PRNGKey(0), shape=0.5)

        prior_params, ems_params, trns_params = hmm.split_coords(params)
        pi = hmm.lat_man.to_probs(hmm.lat_man.to_mean(prior_params))

        def decoded_rows(lkl_params: Array, hrm: Any) -> Array:
            def row(j: int) -> Array:
                s = hrm.lat_man.sufficient_statistic(jnp.array([j], dtype=jnp.float64))
                nat = hrm.lkl_fun_man(lkl_params, s)
                return hrm.obs_man.to_probs(hrm.obs_man.to_mean(nat))

            return jnp.stack([row(j) for j in range(n_states)])

        transition = decoded_rows(trns_params, hmm.trn_map.kernel)
        emission = decoded_rows(ems_params, hmm.ems_hrm)

        obs, _ = hmm.sample(jax.random.PRNGKey(7), params, n_steps=20)

        def step(
            carry: tuple[Array, Array], o: Array
        ) -> tuple[tuple[Array, Array], None]:
            alpha_prev, ll = carry
            alpha = (alpha_prev @ transition) * emission[:, o]
            scale = jnp.sum(alpha)
            return (alpha / scale, ll + jnp.log(scale)), None

        (_, expected), _ = jax.lax.scan(
            step, (pi, jnp.array(0.0)), obs.reshape(-1).astype(jnp.int32)
        )
        assert jnp.allclose(
            hmm.log_observable_density(params, obs), expected, rtol=1e-5, atol=1e-7
        )


@pytest.mark.parametrize(
    "model",
    [KalmanFilter(obs_dim=2, lat_dim=2), HiddenMarkovModel(n_obs=4, n_states=3)],
    ids=["kalman_filter", "hmm"],
)
def test_em_does_not_decrease_log_likelihood(
    model: KalmanFilter | HiddenMarkovModel,
) -> None:
    true_params = model.initialize(jax.random.PRNGKey(0), shape=0.5)
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    obs_batch = jnp.stack([model.sample(k, true_params, n_steps=15)[0] for k in keys])

    def average_log_likelihood(params: Array) -> Array:
        return jnp.mean(
            jax.vmap(model.log_observable_density, (None, 0))(params, obs_batch)
        )

    params = model.initialize(jax.random.PRNGKey(2), shape=0.5)
    lls = [average_log_likelihood(params)]
    for _ in range(3):
        params = model.expectation_maximization(params, obs_batch)
        lls.append(average_log_likelihood(params))
    assert jnp.all(jnp.diff(jnp.stack(lls)) >= -1e-8), lls
