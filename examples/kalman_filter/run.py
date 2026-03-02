"""Kalman Filter Example with EM Learning.

Demonstrates:
1. Creating a Kalman filter from a damped oscillator
2. Sampling trajectories from the model
3. Forward filtering and backward smoothing with known parameters
4. EM learning to fit model parameters from data
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
    obs_dim = 1
    lat_dim = 2
    n_steps = 50
    n_sequences = 20
    n_em_steps = 30

    print("=" * 60)
    print("Kalman Filter with EM Learning")
    print("=" * 60)

    # Create model
    model = create_kalman_filter(obs_dim=obs_dim, lat_dim=lat_dim)
    print(f"Observable dimension: {obs_dim}")
    print(f"Latent dimension: {lat_dim}")
    print(f"Model parameter dimension: {model.dim}")
    print()

    # Design a damped oscillator as ground truth
    # Latent state: [position, velocity]
    # Transition: damped rotation
    damping = 0.98
    omega = 0.4  # angular frequency
    A = damping * jnp.array(
        [[jnp.cos(omega), -jnp.sin(omega)], [jnp.sin(omega), jnp.cos(omega)]]
    )
    Q = 0.02 * jnp.eye(2)  # small process noise

    # Emission: observe noisy position
    C = jnp.array([[1.0, 0.0]])  # (1, 2) — observe position only
    R = jnp.array([[0.15]])  # moderate observation noise

    # Prior: start near [1, 0] with low uncertainty
    mu0 = jnp.array([1.0, 0.0])
    Sigma0 = 0.1 * jnp.eye(2)

    true_params = model.from_standard(A, Q, C, R, mu0, Sigma0)

    # Generate training data: multiple sequences
    key = jax.random.PRNGKey(42)
    key_init, key_sample = jax.random.split(key)

    print(f"Generating {n_sequences} training sequences of length {n_steps}...")
    keys_sample = jax.random.split(key_sample, n_sequences)

    def sample_one(k):
        return sample_latent_process(model, true_params, k, n_steps)

    observations_batch, latents_batch = jax.vmap(sample_one)(keys_sample)
    print(f"Observations batch shape: {observations_batch.shape}")
    print()

    # Use first sequence for visualization
    observations = observations_batch[0]
    true_latents = latents_batch[0]

    # --- Inference with TRUE parameters (oracle) ---
    print("Running inference with true parameters...")
    true_filtered, true_log_lik = conjugated_filtering(
        model, true_params, observations
    )
    true_smoothed = conjugated_smoothing(model, true_params, observations)

    def extract_mean_std(nat_params):
        """Extract mean and standard deviation from natural parameters."""
        mean_params = model.lat_man.to_mean(nat_params)
        mean, cov = model.lat_man.split_mean_covariance(mean_params)
        cov_mat = model.lat_man.cov_man.to_matrix(cov)
        std = jnp.sqrt(jnp.diag(cov_mat))
        return mean, std

    true_filt_stats = jax.vmap(extract_mean_std)(true_filtered)
    true_smth_stats = jax.vmap(extract_mean_std)(true_smoothed)

    true_filt_means, true_filt_stds = true_filt_stats
    true_smth_means, true_smth_stds = true_smth_stats

    filt_rmse = float(
        jnp.sqrt(jnp.mean((true_filt_means - true_latents[1:, :lat_dim]) ** 2))
    )
    smth_rmse = float(
        jnp.sqrt(jnp.mean((true_smth_means - true_latents[1:, :lat_dim]) ** 2))
    )
    print(f"  True-param log-likelihood: {float(true_log_lik):.4f}")
    print(f"  Filtered RMSE: {filt_rmse:.4f}")
    print(f"  Smoothed RMSE: {smth_rmse:.4f}")
    print()

    # --- EM learning ---
    init_params = model.initialize(key_init, location=0.0, shape=1.0)

    initial_log_lik = float(
        jnp.mean(
            jax.vmap(
                lambda obs: latent_process_log_observable_density(
                    model, init_params, obs
                )
            )(observations_batch)
        )
    )
    print(f"Initial average log-likelihood: {initial_log_lik:.4f}")

    print(f"\nRunning {n_em_steps} EM iterations...")
    params = init_params
    log_likelihoods = [initial_log_lik]

    for i in range(n_em_steps):
        params = latent_process_expectation_maximization(
            model, params, observations_batch
        )

        avg_ll = float(
            jnp.mean(
                jax.vmap(
                    lambda obs, p=params: latent_process_log_observable_density(
                        model, p, obs
                    )
                )(observations_batch)
            )
        )
        log_likelihoods.append(avg_ll)

        if (i + 1) % 5 == 0:
            print(f"  Step {i + 1}: avg log-lik = {avg_ll:.4f}")

    final_log_lik = log_likelihoods[-1]
    print(f"\nFinal average log-likelihood: {final_log_lik:.4f}")
    print(f"Improvement: {final_log_lik - initial_log_lik:.4f}")

    # True-param log-likelihood for reference
    true_avg_ll = float(
        jnp.mean(
            jax.vmap(
                lambda obs: latent_process_log_observable_density(
                    model, true_params, obs
                )
            )(observations_batch)
        )
    )
    print(f"True-param average log-likelihood: {true_avg_ll:.4f}")
    print()

    # Save results
    results = KalmanFilterResults(
        obs_dim=obs_dim,
        lat_dim=lat_dim,
        n_steps=n_steps,
        n_sequences=n_sequences,
        n_em_steps=n_em_steps,
        observations=observations.tolist(),
        true_latents=true_latents.tolist(),
        filtered_means=true_filt_means.tolist(),
        filtered_stds=true_filt_stds.tolist(),
        smoothed_means=true_smth_means.tolist(),
        smoothed_stds=true_smth_stds.tolist(),
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
