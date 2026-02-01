"""Gaussian-Boltzmann harmonium for modeling multi-modal distributions."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import PositiveDefinite
from goal.models import BoltzmannLGM

from ..shared import example_paths, jax_cli
from .types import GaussianBoltzmannResults

# Model configuration
n_latents = 8
obs_dim = 2
radii = [0.3, 1.5, 3.0]
noise_stds = [0.15, 0.2, 0.25]
n_obs = 1000
ratios = [0.1, 0.3, 0.6]
learning_rate = 3e-3
n_steps_per_epoch = 200
n_epochs = 1000
plot_range = (-5.0, 5.0)
plot_res = 100


def generate_data(key: Array) -> Array:
    """Generate samples from concentric noisy circles."""
    n_per_circle = [int(ratio * n_obs) for ratio in ratios]
    keys = jax.random.split(key, len(radii))
    circles = []
    for k, r, std, n in zip(keys, radii, noise_stds, n_per_circle):
        angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        points = jnp.stack([r * jnp.cos(angles), r * jnp.sin(angles)], axis=1)
        noise = std * jax.random.normal(k, (n, obs_dim))
        circles.append(points + noise)
    return jnp.concatenate(circles, axis=0)


def train_model(key: Array, model: BoltzmannLGM, data: Array) -> tuple[Array, Array]:
    """Train Boltzmann-Gaussian model."""
    params = model.initialize(key, location=0.0, shape=1.0)
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(p: Array) -> Array:
        return -model.average_log_observable_density(p, data)

    def step(state: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), -loss

    total_steps = n_epochs * n_steps_per_epoch
    (_, final_params), lls = jax.lax.scan(
        step, (opt_state, params), None, length=total_steps
    )

    # Epoch averages
    epoch_lls = lls.reshape(n_epochs, n_steps_per_epoch).mean(axis=1)
    return final_params, epoch_lls


def compute_boltzmann_moments(model: BoltzmannLGM, params: Array) -> Array:
    """Compute E[zz^T] for Boltzmann prior."""
    states = model.lat_man.states

    def log_prob(z: Array) -> Array:
        return jnp.dot(params, model.lat_man.sufficient_statistic(z))

    log_probs = jax.vmap(log_prob)(states)
    log_probs = log_probs - jax.nn.logsumexp(log_probs)
    probs = jnp.exp(log_probs)

    return jnp.sum(jax.vmap(lambda p, z: p * jnp.outer(z, z))(probs, states), axis=0)


def compute_ellipse(
    model: BoltzmannLGM, lkl_params: Array, latent_state: Array, n_points: int = 1000
) -> Array:
    """Compute confidence ellipse for p(x|z)."""
    cond_params = model.lkl_fun_man(lkl_params, latent_state)
    means = model.obs_man.to_mean(cond_params)
    mean, cov = model.obs_man.split_mean_covariance(means)
    cov_mat = model.obs_man.cov_man.to_matrix(cov)

    angles = jnp.linspace(0, 2 * jnp.pi, n_points)
    eigvals, eigvecs = jnp.linalg.eigh(cov_mat)
    unit_circle = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    scaled = unit_circle * jnp.sqrt(eigvals)[None, :]
    return (eigvecs @ scaled.T).T + mean


def main():
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)
    key_data, key_train = jax.random.split(key)

    data = generate_data(key_data)
    model = BoltzmannLGM(obs_dim=obs_dim, obs_rep=PositiveDefinite(), lat_dim=n_latents)

    final_params, lls = train_model(key_train, model, data)

    # Grid densities
    x_range = jnp.linspace(*plot_range, plot_res)
    y_range = jnp.linspace(*plot_range, plot_res)
    xx, yy = jnp.meshgrid(x_range, y_range)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    densities = jax.vmap(model.observable_density, in_axes=(None, 0))(
        final_params, grid_points
    )

    # Analysis
    lkl_params, prior_params = model.split_conjugated(final_params)
    prior_moment = compute_boltzmann_moments(model, prior_params)

    # Confidence ellipses for single-spike states
    single_spike_states = jnp.eye(n_latents)
    ellipses = [compute_ellipse(model, lkl_params, s) for s in single_spike_states]

    # Posterior moments for representative observations
    n_per_circle = [int(ratio * n_obs) for ratio in ratios]
    cumsum = jnp.cumsum(jnp.array([0] + n_per_circle))
    indices = [
        cumsum[0] + n_per_circle[0] // 2,
        cumsum[1],
        cumsum[1] + n_per_circle[1] // 2,
        cumsum[2] + n_per_circle[2] // 2,
    ]
    posterior_obs = data[jnp.array(indices)]

    def posterior_moment(obs: Array) -> Array:
        post_params = model.posterior_at(final_params, obs)
        return compute_boltzmann_moments(model, post_params)

    posterior_moments = [posterior_moment(obs) for obs in posterior_obs]

    results = GaussianBoltzmannResults(
        observations=data.tolist(),
        plot_range_x=x_range.tolist(),
        plot_range_y=y_range.tolist(),
        learned_density=densities.reshape(xx.shape).tolist(),
        log_likelihoods=lls.tolist(),
        component_confidence_ellipses=[e.tolist() for e in ellipses],
        prior_moment_matrix=prior_moment.tolist(),
        posterior_observations=posterior_obs.tolist(),
        posterior_moment_matrices=[m.tolist() for m in posterior_moments],
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
