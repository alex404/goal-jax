"""Validate Contrastive Divergence by comparing to exact training.

Uses small latent dimension where exact computation is tractable,
trains the same model with both exact gradients and CD gradients,
then compares NLL trajectories and learned densities.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import PositiveDefinite
from goal.models import DifferentiableBoltzmannLGM

from ..shared import example_paths, jax_cli
from .types import BoltzmannCDResults

# Model configuration - small scale for exact comparison
n_latents = 8  # Small enough for exact 2^n computation
obs_dim = 2
n_obs = 500
learning_rate = 1e-2
n_epochs = 200
n_steps_per_epoch = 50

# Data: simple mixture of Gaussians
n_clusters = 3
cluster_std = 0.5

# Density grid
plot_range = (-3.5, 3.5)
plot_res = 80


def generate_data(key: Array) -> Array:
    """Generate samples from a mixture of Gaussians."""
    keys = jax.random.split(key, 2)

    # Cluster centers arranged in a triangle
    centers = (
        jnp.array(
            [
                [0.0, 1.0],
                [-0.866, -0.5],
                [0.866, -0.5],
            ]
        )
        * 2.0
    )

    # Assign points to clusters
    n_per_cluster = n_obs // n_clusters
    cluster_assignments = jnp.repeat(jnp.arange(n_clusters), n_per_cluster)

    # Generate points
    noise = cluster_std * jax.random.normal(
        keys[0], (n_clusters * n_per_cluster, obs_dim)
    )
    points = centers[cluster_assignments] + noise

    # Shuffle
    perm = jax.random.permutation(keys[1], len(points))
    return points[perm]


def train_exact(
    model: DifferentiableBoltzmannLGM[PositiveDefinite], data: Array, init_params: Array
) -> tuple[Array, Array]:
    """Train with exact log-likelihood gradients, tracking NLL."""
    params = init_params
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(p: Array) -> Array:
        return -model.average_log_observable_density(p, data)

    def step(state: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), loss

    total_steps = n_epochs * n_steps_per_epoch
    (_, final_params), nlls = jax.lax.scan(
        step, (opt_state, params), None, length=total_steps
    )

    # Epoch averages
    epoch_nlls = nlls.reshape(n_epochs, n_steps_per_epoch).mean(axis=1)
    return final_params, epoch_nlls


def train_cd(
    key: Array,
    model: DifferentiableBoltzmannLGM[PositiveDefinite],
    data: Array,
    init_params: Array,
    cd_steps: int = 1,
) -> tuple[Array, Array]:
    """Train with contrastive divergence, tracking exact NLL for comparison."""
    params = init_params
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    def step(
        state: tuple[Any, Any, Array], _: Any
    ) -> tuple[tuple[Any, Any, Array], Array]:
        opt_state, params, step_key = state
        step_key, subkey = jax.random.split(step_key)

        # Compute exact NLL for tracking (we can do this with 8 neurons)
        nll = -model.average_log_observable_density(params, data)

        # Update with CD gradient
        cd_grad = model.mean_contrastive_divergence_gradient(
            subkey, params, data, k=cd_steps
        )
        updates, opt_state = optimizer.update(cd_grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        return (opt_state, params, step_key), nll

    total_steps = n_epochs * n_steps_per_epoch
    (_, final_params, _), nlls = jax.lax.scan(
        step, (opt_state, params, key), None, length=total_steps
    )

    # Epoch averages
    epoch_nlls = nlls.reshape(n_epochs, n_steps_per_epoch).mean(axis=1)
    return final_params, epoch_nlls


def compute_density_grid(
    model: DifferentiableBoltzmannLGM[PositiveDefinite], params: Array
) -> tuple[Array, Array, Array]:
    """Compute density on a grid."""
    x_range = jnp.linspace(*plot_range, plot_res)
    y_range = jnp.linspace(*plot_range, plot_res)
    xx, yy = jnp.meshgrid(x_range, y_range)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    densities = jax.vmap(model.observable_density, in_axes=(None, 0))(
        params, grid_points
    ).reshape(xx.shape)

    return x_range, y_range, densities


def main():
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)
    key_data, key_init, _, key_cd = jax.random.split(key, 4)

    # Generate data
    data = generate_data(key_data)
    print(f"Generated {len(data)} observations")

    # Create model
    model = DifferentiableBoltzmannLGM(
        obs_dim=obs_dim, obs_rep=PositiveDefinite(), lat_dim=n_latents
    )
    print(f"Model: {obs_dim}D observations, {n_latents} latent neurons")

    # Same initialization for both methods
    init_params = model.initialize(key_init, location=0.0, shape=0.1)
    init_nll = -model.average_log_observable_density(init_params, data)
    print(f"Initial NLL: {init_nll:.4f}")

    # Compute initial density
    x_range, y_range, init_density = compute_density_grid(model, init_params)

    # Train with exact gradients
    print(f"\nTraining with exact gradients ({n_epochs} epochs)...")
    params_exact, nlls_exact = train_exact(model, data, init_params)
    print(f"  Final NLL: {nlls_exact[-1]:.4f}")

    # Compute exact-trained density
    _, _, exact_density = compute_density_grid(model, params_exact)

    # Train with CD
    cd_k = 1
    print(f"\nTraining with CD-{cd_k} ({n_epochs} epochs)...")
    params_cd, nlls_cd = train_cd(key_cd, model, data, init_params, cd_steps=cd_k)
    print(f"  Final NLL: {nlls_cd[-1]:.4f}")

    # Compute CD-trained density
    _, _, cd_density = compute_density_grid(model, params_cd)

    # Save results
    results: BoltzmannCDResults = {
        "observations": data.tolist(),
        "plot_range_x": x_range.tolist(),
        "plot_range_y": y_range.tolist(),
        "exact_nlls": nlls_exact.tolist(),
        "cd_nlls": nlls_cd.tolist(),
        "initial_density": init_density.tolist(),
        "exact_density": exact_density.tolist(),
        "cd_density": cd_density.tolist(),
    }
    paths.save_analysis(results)
    print("\nResults saved.")


if __name__ == "__main__":
    main()
