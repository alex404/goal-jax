"""Gaussian Markov Chain MLE via average sufficient statistics.

For a fully observed Markov chain, maximum likelihood is closed-form:
    hat_theta = to_natural(average_sufficient_statistic(data))

This example uses a 1D chain so that trajectories are directly plottable
as time series.
"""

import jax
import jax.numpy as jnp

from goal.models import create_gaussian_markov_chain

from ..shared import example_paths, jax_cli
from .types import GaussianMarkovChainResults


def main() -> None:
    jax_cli()
    jax.config.update("jax_enable_x64", True)
    paths = example_paths(__file__)

    lat_dim = 1
    n_steps = 5
    n_sequences = 200
    n_plot_trajectories = 5

    model = create_gaussian_markov_chain(lat_dim=lat_dim, n_steps=n_steps)
    key = jax.random.PRNGKey(42)
    key_init, key_sample, key_plot = jax.random.split(key, 3)

    true_params = model.initialize(key_init, location=0.0, shape=0.5)
    data = model.sample(key_sample, true_params, n=n_sequences)

    # Closed-form MLE: average sufficient statistics -> to_natural
    learned_params = model.to_natural(model.average_sufficient_statistic(data))

    true_ll = float(
        jnp.mean(
            jax.vmap(model.log_density, in_axes=(None, 0))(true_params, data)
        )
    )
    learned_ll = float(
        jnp.mean(
            jax.vmap(model.log_density, in_axes=(None, 0))(learned_params, data)
        )
    )
    param_error = float(
        jnp.max(jnp.abs(model.to_mean(true_params) - model.to_mean(learned_params)))
    )

    # Sample trajectories for plotting from both true and learned models
    true_trajectories = model.sample(key_plot, true_params, n=n_plot_trajectories)
    learned_trajectories = model.sample(
        jax.random.fold_in(key_plot, 1), learned_params, n=n_plot_trajectories
    )

    # Reshape flat trajectories into (n_traj, n_steps+1) — 1D states
    n_states = n_steps + 1
    true_traj_1d = true_trajectories.reshape(n_plot_trajectories, n_states)
    learned_traj_1d = learned_trajectories.reshape(n_plot_trajectories, n_states)

    # Extract all (x_t, x_{t+1}) consecutive pairs from training data for scatter
    data_2d = data.reshape(n_sequences, n_states)
    pairs_x = data_2d[:, :-1].reshape(-1)
    pairs_y = data_2d[:, 1:].reshape(-1)

    print(f"True LL:    {true_ll:.4f}")
    print(f"Learned LL: {learned_ll:.4f}  (should be >= true LL on training data)")
    print(f"Max param error: {param_error:.6f}")

    results = GaussianMarkovChainResults(
        lat_dim=lat_dim,
        n_steps=n_steps,
        n_sequences=n_sequences,
        true_log_likelihood=true_ll,
        learned_log_likelihood=learned_ll,
        param_error=param_error,
        true_trajectories=true_traj_1d.tolist(),
        learned_trajectories=learned_traj_1d.tolist(),
        transition_pairs_x=pairs_x.tolist(),
        transition_pairs_y=pairs_y.tolist(),
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
