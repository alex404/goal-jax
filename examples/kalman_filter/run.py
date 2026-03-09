"""Kalman Filter Example with Gradient-Based and EM Learning.

Demonstrates:
1. Creating a Kalman filter from a damped oscillator
2. Sampling trajectories from the model
3. Forward filtering and backward smoothing with known parameters
4. Gradient ascent on log p(x_{1:T}) to learn model parameters
5. EM (expectation maximization) to learn model parameters
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.models import KalmanFilter, create_kalman_filter

from ..shared import example_paths, jax_cli
from .types import KalmanFilterResults

jax.config.update("jax_enable_x64", True)

# Model configuration
OBS_DIM = 1
LAT_DIM = 2
N_STEPS = 50
N_SEQUENCES = 20
N_GRAD_STEPS = 200
N_EM_STEPS = 200


def create_ground_truth() -> tuple[KalmanFilter, Array]:
    """Create ground truth damped oscillator Kalman filter."""
    model = create_kalman_filter(obs_dim=OBS_DIM, lat_dim=LAT_DIM)

    damping = 0.98
    omega = 0.4
    A = damping * jnp.array(
        [[jnp.cos(omega), -jnp.sin(omega)], [jnp.sin(omega), jnp.cos(omega)]]
    )
    Q = 0.02 * jnp.eye(2)
    C = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.15]])
    mu0 = jnp.array([1.0, 0.0])
    Sigma0 = 0.1 * jnp.eye(2)

    return model, model.from_standard(A, Q, C, R, mu0, Sigma0)


def avg_log_lik(model: KalmanFilter, params: Array, observations_batch: Array) -> Array:
    """Average log-likelihood across batch."""
    return jnp.mean(
        jax.vmap(lambda obs: model.log_observable_density(params, obs))(
            observations_batch
        )
    )


def fit_gradient(
    key: Array,
    model: KalmanFilter,
    observations_batch: Array,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    """Fit via gradient ascent on log p(x_{1:T})."""
    init_params = model.initialize_from_sample(key, observations_batch)

    def obj(p: Array) -> Array:
        return avg_log_lik(model, p, observations_batch)

    optimizer = optax.adam(learning_rate=1e-2)

    def grad_step(carry: tuple[Array, Any], _: None) -> tuple[tuple[Array, Any], Array]:
        params, opt_state = carry
        ll = obj(params)
        grads = jax.grad(obj)(params)
        updates, new_opt_state = optimizer.update(
            jax.tree.map(jnp.negative, grads), opt_state
        )
        new_params = jnp.asarray(optax.apply_updates(params, updates))
        return (new_params, new_opt_state), ll

    opt_state = optimizer.init(init_params)
    (final_params, _), lls = jax.lax.scan(
        grad_step, (init_params, opt_state), None, length=n_steps
    )
    return lls.ravel(), init_params, final_params


fit_gradient = jax.jit(fit_gradient, static_argnames=["model", "n_steps"])


def fit_em(
    key: Array,
    model: KalmanFilter,
    observations_batch: Array,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    """Fit via expectation maximization."""
    init_params = model.initialize_from_sample(key, observations_batch)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = avg_log_lik(model, params, observations_batch)
        return model.expectation_maximization(params, observations_batch), ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_em = jax.jit(fit_em, static_argnames=["model", "n_steps"])


def extract_inference(
    model: KalmanFilter, params: Array, observations: Array
) -> tuple[Array, Array, Array, Array]:
    """Run filtering and smoothing, return means and stds."""

    def extract_mean_std(nat_params: Array) -> tuple[Array, Array]:
        mean_params = model.lat_man.to_mean(nat_params)
        mean, cov = model.lat_man.split_mean_covariance(mean_params)
        cov_mat = model.lat_man.cov_man.to_matrix(cov)
        std = jnp.sqrt(jnp.diag(cov_mat))
        return mean, std

    filtered, _ = model.filtering(params, observations)
    _, smoothed, _ = model.smoothing(params, observations)

    filt_means, filt_stds = jax.vmap(extract_mean_std)(filtered)
    smth_means, smth_stds = jax.vmap(extract_mean_std)(smoothed)
    return filt_means, filt_stds, smth_means, smth_stds


def main() -> None:
    """Run Kalman filter example."""
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)
    key_sample, key_train = jax.random.split(key)

    # Ground truth
    model, true_params = create_ground_truth()

    # Sample training data
    keys_sample = jax.random.split(key_sample, N_SEQUENCES)
    observations_batch, latents_batch = jax.vmap(
        lambda k: model.sample(k, true_params, N_STEPS)
    )(keys_sample)

    observations = observations_batch[0]
    true_latents = latents_batch[0]

    # Oracle inference
    true_filt_means, true_filt_stds, true_smth_means, true_smth_stds = (
        extract_inference(model, true_params, observations)
    )

    true_avg_ll = float(avg_log_lik(model, true_params, observations_batch))
    init_params = model.initialize_from_sample(key_train, observations_batch)
    initial_ll = float(avg_log_lik(model, init_params, observations_batch))

    # Training
    grad_lls, _, grad_final = fit_gradient(
        key_train, model, observations_batch, N_GRAD_STEPS
    )
    em_lls, _, em_final = fit_em(key_train, model, observations_batch, N_EM_STEPS)

    # Learned inference
    learned: dict[str, tuple[Array, Array, Array, Array]] = {
        "Gradient": extract_inference(model, grad_final, observations),
        "EM": extract_inference(model, em_final, observations),
    }

    results = KalmanFilterResults(
        obs_dim=OBS_DIM,
        lat_dim=LAT_DIM,
        n_steps=N_STEPS,
        n_sequences=N_SEQUENCES,
        observations=observations.tolist(),
        true_latents=true_latents.tolist(),
        filtered_means=true_filt_means.tolist(),
        filtered_stds=true_filt_stds.tolist(),
        smoothed_means=true_smth_means.tolist(),
        smoothed_stds=true_smth_stds.tolist(),
        log_likelihoods={
            "Gradient": grad_lls.tolist(),
            "EM": em_lls.tolist(),
        },
        learned_filtered_means={k: v[0].tolist() for k, v in learned.items()},
        learned_filtered_stds={k: v[1].tolist() for k, v in learned.items()},
        learned_smoothed_means={k: v[2].tolist() for k, v in learned.items()},
        learned_smoothed_stds={k: v[3].tolist() for k, v in learned.items()},
        true_log_lik=true_avg_ll,
        initial_log_lik=initial_ll,
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
