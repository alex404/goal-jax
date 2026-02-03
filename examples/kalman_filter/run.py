"""Kalman Filter Example with EM Learning.

Demonstrates:
1. Creating a Kalman filter model
2. Sampling trajectories from the model
3. Forward filtering to compute p(z_t | x_{1:t})
4. Backward smoothing to compute p(z_t | x_{1:T})
5. EM learning to fit model parameters from data
"""

import jax
import jax.numpy as jnp

from goal.models import (
    conjugated_filtering,
    conjugated_smoothing,
    create_kalman_filter,
    latent_process_expectation_maximization,
    latent_process_log_observable_density,
    sample_latent_process,
)

from ..shared import example_paths, jax_cli
from .types import KalmanFilterResults

# Enable float64 for better numerical precision
jax.config.update("jax_enable_x64", True)


def main() -> None:
    """Run Kalman filter example with EM learning."""
    jax_cli()
    paths = example_paths(__file__)

    # Model configuration
    obs_dim = 2
    lat_dim = 3
    n_steps = 30
    n_sequences = 10  # Number of sequences for EM
    n_em_steps = 20

    print("=" * 60)
    print("Kalman Filter with EM Learning")
    print("=" * 60)

    # Create model
    model = create_kalman_filter(obs_dim=obs_dim, lat_dim=lat_dim)
    print(f"Observable dimension: {obs_dim}")
    print(f"Latent dimension: {lat_dim}")
    print(f"Model parameter dimension: {model.dim}")
    print()

    # Initialize "true" parameters (what we'll try to recover)
    key = jax.random.PRNGKey(42)
    key_true, key_init, key_sample = jax.random.split(key, 3)
    true_params = model.initialize(key_true, location=0.0, shape=0.3)

    # Generate training data: multiple sequences
    print(f"Generating {n_sequences} training sequences of length {n_steps}...")
    keys_sample = jax.random.split(key_sample, n_sequences)

    def sample_one(k):
        return sample_latent_process(model, true_params, k, n_steps)

    observations_batch, latents_batch = jax.vmap(sample_one)(keys_sample)
    print(f"Observations batch shape: {observations_batch.shape}")
    print()

    # Initialize parameters for learning (different from true)
    init_params = model.initialize(key_init, location=0.0, shape=0.5)

    # Compute initial log-likelihood
    initial_log_lik = float(
        jnp.mean(
            jax.vmap(
                lambda obs: latent_process_log_observable_density(model, init_params, obs)
            )(observations_batch)
        )
    )
    print(f"Initial average log-likelihood: {initial_log_lik:.4f}")

    # EM learning
    print(f"\nRunning {n_em_steps} EM iterations...")
    params = init_params
    log_likelihoods = [initial_log_lik]

    for i in range(n_em_steps):
        # EM step
        params = latent_process_expectation_maximization(
            model, params, observations_batch
        )

        # Compute log-likelihood
        avg_ll = float(
            jnp.mean(
                jax.vmap(
                    lambda obs: latent_process_log_observable_density(model, params, obs)
                )(observations_batch)
            )
        )
        log_likelihoods.append(avg_ll)

        if (i + 1) % 5 == 0:
            print(f"  Step {i + 1}: avg log-lik = {avg_ll:.4f}")

    final_log_lik = log_likelihoods[-1]
    print(f"\nFinal average log-likelihood: {final_log_lik:.4f}")
    print(f"Improvement: {final_log_lik - initial_log_lik:.4f}")
    print()

    # Use first sequence for visualization
    observations = observations_batch[0]
    true_latents = latents_batch[0]

    # Run filtering and smoothing with learned parameters
    print("Running inference with learned parameters...")
    filtered, _ = conjugated_filtering(model, params, observations)
    smoothed = conjugated_smoothing(model, params, observations)

    # Extract means and standard deviations from natural parameters
    def extract_mean_std(nat_params):
        """Extract mean and standard deviation from natural parameters."""
        mean_params = model.lat_man.to_mean(nat_params)
        mean, cov = model.lat_man.split_mean_covariance(mean_params)
        # Extract diagonal of covariance for standard deviations
        cov_mat = model.lat_man.cov_man.to_matrix(cov)
        std = jnp.sqrt(jnp.diag(cov_mat))
        return mean, std

    filtered_stats = jax.vmap(extract_mean_std)(filtered)
    smoothed_stats = jax.vmap(extract_mean_std)(smoothed)

    filtered_means = filtered_stats[0]
    filtered_stds = filtered_stats[1]
    smoothed_means = smoothed_stats[0]
    smoothed_stds = smoothed_stats[1]

    print(f"Filtered means shape: {filtered_means.shape}")
    print(f"Smoothed means shape: {smoothed_means.shape}")
    print()

    # Compute RMSE between smoothed and true latents
    # Note: smoothed has T time points (z_1 to z_T), true_latents has T+1 (z_0 to z_T)
    rmse = float(jnp.sqrt(jnp.mean((smoothed_means - true_latents[1:, :lat_dim]) ** 2)))
    print(f"RMSE (smoothed vs true latents): {rmse:.4f}")

    # Save results
    results = KalmanFilterResults(
        obs_dim=obs_dim,
        lat_dim=lat_dim,
        n_steps=n_steps,
        n_sequences=n_sequences,
        n_em_steps=n_em_steps,
        observations=observations.tolist(),
        true_latents=true_latents.tolist(),
        filtered_means=filtered_means.tolist(),
        filtered_stds=filtered_stds.tolist(),
        smoothed_means=smoothed_means.tolist(),
        smoothed_stds=smoothed_stds.tolist(),
        log_likelihoods=log_likelihoods,
        initial_log_lik=initial_log_lik,
        final_log_lik=final_log_lik,
    )
    paths.save_analysis(results)

    print("=" * 60)
    print("Example completed! Results saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()
