"""Latent trajectory recovery with Factor Analysis.

Demonstrates dimensionality reduction by:
1. Generating a smooth trajectory in 2D latent space (figure-8 curve)
2. Projecting to high-dimensional observations through a random linear map
3. Adding observation noise
4. Fitting Factor Analysis to recover the latent structure
5. Showing that FA recovers the trajectory shape (up to rotation)

Usage:
    python -m examples.dimensionality_reduction.run
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.models import FactorAnalysis

from ..shared import example_paths, jax_cli
from .types import TrajectoryResults

# Configuration
OBS_DIM = 20  # High-dimensional observation space
LATENT_DIM = 2  # True latent dimensionality
N_POINTS = 200  # Points along the trajectory
NOISE_STD = 0.5  # Observation noise level
N_EM_STEPS = 100  # EM iterations


def generate_figure_8(n_points: int) -> Array:
    """Generate a figure-8 (lemniscate) trajectory in 2D.

    Returns points with coordinates roughly in [-1, 1].
    """
    t = jnp.linspace(0, 2 * jnp.pi, n_points)
    x = jnp.sin(t)
    y = jnp.sin(t) * jnp.cos(t)
    return jnp.stack([x, y], axis=1)


def generate_observations(
    key: Array,
    latents: Array,
    obs_dim: int,
    noise_std: float,
) -> tuple[Array, Array]:
    """Project latents to observation space and add noise.

    Returns:
        observations: (n_points, obs_dim) noisy observations
        projection: (latent_dim, obs_dim) the true projection matrix
    """
    key_proj, key_noise = jax.random.split(key)
    latent_dim = latents.shape[1]

    # Random orthogonal-ish projection (scaled for reasonable variance)
    projection = jax.random.normal(key_proj, (latent_dim, obs_dim)) / jnp.sqrt(latent_dim)

    # Project and add noise
    clean = latents @ projection
    noise = jax.random.normal(key_noise, clean.shape) * noise_std
    observations = clean + noise

    return observations, projection


def procrustes_align(source: Array, target: Array) -> Array:
    """Align source to target using Procrustes analysis (rotation + reflection + scale).

    Finds the orthogonal transformation that best aligns source to target.
    """
    # Center both
    source_centered = source - jnp.mean(source, axis=0)
    target_centered = target - jnp.mean(target, axis=0)

    # SVD of cross-covariance
    H = source_centered.T @ target_centered
    U, _, Vt = jnp.linalg.svd(H)

    # Optimal rotation (allowing reflection)
    R = Vt.T @ U.T

    # Scale factor
    scale = jnp.trace(target_centered.T @ (source_centered @ R.T)) / jnp.trace(
        source_centered.T @ source_centered
    )

    # Apply transformation
    aligned = scale * (source_centered @ R.T) + jnp.mean(target, axis=0)
    return aligned


def fit_factor_analysis(
    key: Array,
    observations: Array,
    latent_dim: int,
    n_steps: int,
) -> tuple[Array, Array, FactorAnalysis]:
    """Fit Factor Analysis model via EM.

    Returns:
        lls: Log-likelihoods at each step
        final_params: Learned model parameters
        model: The FA model instance
    """
    obs_dim = observations.shape[1]
    model = FactorAnalysis(obs_dim, latent_dim)

    init_params = model.initialize(key)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = model.average_log_observable_density(params, observations)
        new_params = model.expectation_maximization(params, observations)
        return new_params, ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls, final_params, model


def infer_latents(model: FactorAnalysis, params: Array, observations: Array) -> Array:
    """Infer latent positions for each observation using posterior mean."""

    def posterior_mean(x: Array) -> Array:
        # Get posterior distribution parameters
        posterior_params = model.posterior_at(params, x)
        # Convert to mean parameters and extract mean
        posterior_means = model.lat_man.to_mean(posterior_params)
        mean, _ = model.lat_man.split_mean_covariance(posterior_means)
        return mean

    return jax.vmap(posterior_mean)(observations)


def reconstruct_observations(
    model: FactorAnalysis, params: Array, observations: Array
) -> Array:
    """Reconstruct observations via posterior mean projection."""
    # Get likelihood parameters (contains observation bias and interaction)
    lkl_params, _ = model.split_conjugated(params)
    obs_bias, int_params = model.lkl_fun_man.split_coords(lkl_params)

    # Observation mean from bias
    obs_means = model.obs_man.to_mean(obs_bias)
    obs_mean, _ = model.obs_man.split_mean_covariance(obs_means)

    # Get interaction matrix using the model's int_man
    # The matrix has shape (obs_dim, lat_dim) for W @ z
    int_matrix = model.int_man.to_matrix(int_params)

    def reconstruct_one(x: Array) -> Array:
        # Posterior mean of latent
        posterior_params = model.posterior_at(params, x)
        posterior_means = model.lat_man.to_mean(posterior_params)
        z_mean, _ = model.lat_man.split_mean_covariance(posterior_means)
        # Reconstruct: obs_mean + W @ z_mean
        return obs_mean + int_matrix @ z_mean

    return jax.vmap(reconstruct_one)(observations)


def main():
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    print(f"Generating figure-8 trajectory with {N_POINTS} points...")
    true_latents = generate_figure_8(N_POINTS)

    print(f"Projecting to {OBS_DIM}D observation space (noise_std={NOISE_STD})...")
    key, obs_key = jax.random.split(key)
    observations, _ = generate_observations(obs_key, true_latents, OBS_DIM, NOISE_STD)

    print(f"Fitting Factor Analysis ({LATENT_DIM}D latent, {N_EM_STEPS} EM steps)...")
    key, fit_key = jax.random.split(key)
    lls, final_params, model = fit_factor_analysis(
        fit_key, observations, LATENT_DIM, N_EM_STEPS
    )
    print(f"  Initial LL: {lls[0]:.2f}, Final LL: {lls[-1]:.2f}")

    print("Inferring latent positions...")
    recovered_latents = infer_latents(model, final_params, observations)

    print("Aligning recovered latents to ground truth (Procrustes)...")
    aligned_latents = procrustes_align(recovered_latents, true_latents)

    print("Computing reconstructions...")
    reconstructions = reconstruct_observations(model, final_params, observations)
    recon_errors = jnp.sqrt(jnp.sum((observations - reconstructions) ** 2, axis=1))
    mean_error = float(jnp.mean(recon_errors))
    print(f"  Mean reconstruction error: {mean_error:.4f}")

    # Compute alignment quality
    alignment_error = jnp.sqrt(jnp.mean(jnp.sum((aligned_latents - true_latents) ** 2, axis=1)))
    print(f"  Latent alignment RMSE: {alignment_error:.4f}")

    results = TrajectoryResults(
        true_latents=true_latents.tolist(),
        observations=observations.tolist(),
        training_lls=lls.tolist(),
        recovered_latents=recovered_latents.tolist(),
        aligned_latents=aligned_latents.tolist(),
        reconstructions=reconstructions.tolist(),
        reconstruction_errors=recon_errors.tolist(),
        obs_dim=OBS_DIM,
        latent_dim=LATENT_DIM,
        n_points=N_POINTS,
        noise_std=NOISE_STD,
    )
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
