"""Gaussian-Boltzmann harmonium example.

This example demonstrates training a Boltzmann machine with binary latent variables
and Gaussian observables. The model learns to represent a two-circle distribution
using a discrete latent representation with continuous observations.
"""

from typing import Any, TypedDict

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    AffineMap,
    Natural,
    Optimizer,
    OptState,
    Point,
    PositiveDefinite,
    Rectangular,
)
from goal.models import (
    Boltzmann,
    BoltzmannDifferentiableLinearGaussianModel,
    Euclidean,
    Normal,
)

from ..shared import example_paths, initialize_jax
from .types import GaussianBoltzmannResults

# Type aliases for readability
Model = BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]
ModelParams = Point[
    Natural, BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]
]
BoltzmannParams = Point[Natural, Boltzmann]
LikelihoodParams = Point[
    Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[PositiveDefinite]]
]


### Constants ###

# Model architecture
N_LATENTS = 12  # Number of binary latent variables (2^12 = 4096 states)
OBS_DIM = 2  # Dimensionality of observations

# Data generation parameters
RADII = [0.3, 1.5, 3.0]  # Radii of three concentric circles
NOISE_STDS = [0.15, 0.2, 0.25]  # Different noise levels for each circle
N_OBS = 1000  # Total number of observations
RATIOS = [
    1 / 10,
    3 / 10,
    6 / 10,
]  # Proportion of samples from each circle (100, 200, 300)

# Training hyperparameters
LEARNING_RATE = 3e-3
N_STEPS_PER_EPOCH = 200
N_EPOCHS = 1000

# Plotting parameters
PLOT_RES = 100  # Grid resolution
PLOT_MIN = -5.0
PLOT_MAX = 5.0


### Data Generation ###


def circle_2d(radius: float, theta: Array | float) -> Array:
    """Generate a point on a circle at the given angle."""
    return jnp.array([radius * jnp.cos(theta), radius * jnp.sin(theta)])


def generate_noisy_circle(
    key: Array, radius: float, noise_std: float, n_samples: int
) -> Array:
    """Generate points uniformly distributed around a circle with Gaussian noise."""
    angles = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)

    def point_at_angle(theta: Array) -> Array:
        return circle_2d(radius, theta)

    points = jax.vmap(point_at_angle)(angles)
    noise = noise_std * jax.random.normal(key, (n_samples, OBS_DIM))
    return points + noise


def generate_data(key: Array) -> Array:
    """Generate the full dataset consisting of two noisy circles."""
    n_samples_per_circle = [int(ratio * N_OBS) for ratio in RATIOS]

    keys = jax.random.split(key, len(RADII))
    circles = [
        generate_noisy_circle(k, r, std, n)
        for k, r, std, n in zip(keys, RADII, NOISE_STDS, n_samples_per_circle)
    ]

    return jnp.concatenate(circles, axis=0)


### Training ###


def train_model(key: Array, model: Model, data: Array) -> tuple[ModelParams, Array]:
    """Train the Boltzmann-Gaussian model using AdamW optimizer.

    Args:
        key: Random key for initialization
        model: The Boltzmann-Gaussian harmonium model
        data: Training observations

    Returns:
        Tuple of (trained_parameters, epoch_log_likelihoods)
    """
    # Initialize parameters and optimizer
    params = model.initialize(key, location=0.0, shape=1.0)
    optimizer = Optimizer.adamw(man=model, learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    # Define loss function (negative log-likelihood)
    def loss_fn(p: ModelParams) -> Array:
        return -model.average_log_observable_density(p, data)

    # Define single gradient step
    def grad_step(
        state: tuple[OptState, ModelParams], _: Any
    ) -> tuple[tuple[OptState, ModelParams], Array]:
        opt_state, params = state
        loss_val, grads = model.value_and_grad(loss_fn, params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), -loss_val  # Return log-likelihood

    # Run training loop using scan for efficiency
    total_steps = N_EPOCHS * N_STEPS_PER_EPOCH
    (_, final_params), log_likelihoods = jax.lax.scan(
        grad_step, (opt_state, params), None, length=total_steps
    )

    # Compute per-epoch average log-likelihoods
    epoch_log_likelihoods = log_likelihoods.reshape(N_EPOCHS, N_STEPS_PER_EPOCH).mean(
        axis=1
    )

    # Print progress periodically
    for epoch in range(0, N_EPOCHS, 100):
        print(
            f"Epoch {epoch}: Average Log-Likelihood = {epoch_log_likelihoods[epoch]:.6f}"
        )

    return final_params, epoch_log_likelihoods


### Analysis Functions ###


def compute_boltzmann_moment_matrix(
    model: Model, boltzmann_params: BoltzmannParams
) -> Array:
    """Compute the second moment matrix E[zz^T] for a Boltzmann distribution.

    Args:
        model: The Boltzmann-Gaussian model
        boltzmann_params: Parameters of the Boltzmann distribution

    Returns:
        Moment matrix of shape (n_latents, n_latents)
    """
    states = model.pst_lat_man.states  # All 2^n binary states

    # Compute log probabilities for each state
    def log_prob_at_state(z: Array) -> Array:
        return model.pst_lat_man.dot(
            boltzmann_params, model.pst_lat_man.sufficient_statistic(z)
        )

    log_probs = jax.vmap(log_prob_at_state)(states)
    log_probs = log_probs - jax.nn.logsumexp(log_probs)  # Normalize
    probs = jnp.exp(log_probs)

    # Compute E[zz^T] as weighted sum of outer products
    def weighted_outer_product(p: Array, z: Array) -> Array:
        return p * jnp.outer(z, z)

    return jnp.sum(jax.vmap(weighted_outer_product)(probs, states), axis=0)


def compute_confidence_ellipse(
    model: Model,
    likelihood_params: LikelihoodParams,
    latent_state: Array,
    n_points: int = 1000,
    n_std: float = 1.0,
) -> Array:
    """Compute confidence ellipse for the observable distribution given a latent state.

    Args:
        model: The Boltzmann-Gaussian model
        likelihood_params: Likelihood parameters p(x|z)
        latent_state: Binary latent state vector
        n_points: Number of points on ellipse
        n_std: Number of standard deviations for ellipse

    Returns:
        Array of shape (n_points, obs_dim) representing the ellipse
    """
    # Get conditional distribution p(x|z)
    z_location = model.pst_lat_man.loc_man.mean_point(latent_state)
    conditional_obs_params = model.lkl_man(likelihood_params, z_location)

    # Extract mean and covariance
    conditional_mean_params = model.obs_man.to_mean(conditional_obs_params)
    obs_mean, obs_cov = model.obs_man.split_mean_covariance(conditional_mean_params)
    cov_matrix = model.obs_man.cov_man.to_dense(obs_cov)

    # Generate ellipse via eigendecomposition
    angles = jnp.linspace(0, 2 * jnp.pi, n_points)
    eigvals, eigvecs = jnp.linalg.eigh(cov_matrix)

    # Transform unit circle to ellipse
    unit_circle = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    scaled_circle = unit_circle * jnp.sqrt(eigvals)[None, :] * n_std
    return (eigvecs @ scaled_circle.T).T + obs_mean.array


def compute_observable_density_grid(
    model: Model, params: ModelParams, grid_points: Array
) -> Array:
    """Compute observable densities p(x) on a grid of points.

    Args:
        model: The Boltzmann-Gaussian model
        params: Model parameters
        grid_points: Grid points of shape (n_points, obs_dim)

    Returns:
        Densities of shape (n_points,)
    """
    return jax.vmap(model.observable_density, in_axes=(None, 0))(params, grid_points)


def compute_posterior_moment_matrix(
    model: Model, params: ModelParams, obs: Array
) -> Array:
    """Compute the posterior moment matrix E[zz^T | x] for a given observation.

    Args:
        model: The Boltzmann-Gaussian model
        params: Model parameters
        obs: Single observation

    Returns:
        Posterior moment matrix of shape (n_latents, n_latents)
    """
    posterior_params = model.posterior_at(params, obs)
    return compute_boltzmann_moment_matrix(model, posterior_params)


class AnalysisResults(TypedDict):
    """Results from model analysis."""

    density_grid: Array
    x_range: Array
    y_range: Array
    prior_moment: Array
    confidence_ellipses: list[Array]
    posterior_obs: Array
    posterior_moments: list[Array]


def analyze_model(model: Model, params: ModelParams, data: Array) -> AnalysisResults:
    """Perform comprehensive analysis of the trained model.

    Args:
        model: The trained Boltzmann-Gaussian model
        params: Trained model parameters
        data: Training data

    Returns:
        Dictionary containing analysis results
    """
    # 1. Compute observable density on a grid
    x_range = jnp.linspace(PLOT_MIN, PLOT_MAX, PLOT_RES)
    y_range = jnp.linspace(PLOT_MIN, PLOT_MAX, PLOT_RES)
    xx, yy = jnp.meshgrid(x_range, y_range)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    densities = compute_observable_density_grid(model, params, grid_points)
    density_grid = densities.reshape(xx.shape)

    # 2. Split into likelihood and prior parameters
    likelihood_params, prior_params = model.split_conjugated(params)

    # 3. Compute prior moment matrix
    prior_moment = compute_boltzmann_moment_matrix(model, prior_params)

    # 4. Generate confidence ellipses for single-spike latent states
    single_spike_states = jnp.eye(N_LATENTS)
    confidence_ellipses = [
        compute_confidence_ellipse(model, likelihood_params, state)
        for state in single_spike_states
    ]

    # 5. Select representative observations for posterior analysis
    posterior_obs_indices = _select_posterior_observation_indices()
    posterior_obs = data[jnp.array(posterior_obs_indices)]

    # 6. Compute posterior moment matrices for selected observations
    def posterior_moment_at_obs(obs: Array) -> Array:
        return compute_posterior_moment_matrix(model, params, obs)

    posterior_moments: Array = jax.vmap(posterior_moment_at_obs)(posterior_obs)
    posterior_moments_list: list[Array] = list(posterior_moments)

    return {
        "density_grid": density_grid,
        "x_range": x_range,
        "y_range": y_range,
        "prior_moment": prior_moment,
        "confidence_ellipses": confidence_ellipses,
        "posterior_obs": posterior_obs,
        "posterior_moments": posterior_moments_list,
    }


def _select_posterior_observation_indices() -> list[int]:
    """Select 4 representative observations for posterior analysis (2x2 grid)."""
    n_samples_per_circle = [int(ratio * N_OBS) for ratio in RATIOS]
    indices: list[int] = []
    cumulative_idx = 0

    # Select observations from different circles: 1 from first, 2 from second, 1 from third
    for i, n_samples in enumerate(n_samples_per_circle):
        if i == 0:  # First circle: middle observation
            indices.append(cumulative_idx + n_samples // 2)
        elif i == 1:  # Second circle: first and middle observations
            indices.append(cumulative_idx)
            indices.append(cumulative_idx + n_samples // 2)
        elif i == 2:  # Third circle: middle observation
            indices.append(cumulative_idx + n_samples // 2)
        cumulative_idx += n_samples

    return indices


### Main Execution ###


def compute_gaussian_boltzmann_results(key: Array) -> GaussianBoltzmannResults:
    """Run the complete Gaussian-Boltzmann harmonium example.

    Args:
        key: Random key for reproducibility

    Returns:
        Complete results ready for serialization and plotting
    """
    # Generate synthetic data
    print("Generating data...")
    key_data, key_train = jax.random.split(key)
    data = generate_data(key_data)
    print(f"Generated {len(data)} observations")

    # Create model
    print("\nCreating model...")
    model = BoltzmannDifferentiableLinearGaussianModel(
        obs_dim=OBS_DIM,
        obs_rep=PositiveDefinite,
        lat_dim=N_LATENTS,
    )
    print(
        f"Model: {OBS_DIM}D observables, {N_LATENTS} binary latents (2^{N_LATENTS} = {2**N_LATENTS} states)"
    )

    # Train model
    print(f"\nTraining for {N_EPOCHS} epochs...")
    final_params, log_likelihoods = train_model(key_train, model, data)
    print(f"Final log-likelihood: {log_likelihoods[-1]:.6f}")

    # Analyze trained model
    print("\nAnalyzing model...")
    analysis = analyze_model(model, final_params, data)

    # Convert to serializable format
    return GaussianBoltzmannResults(
        observations=data.tolist(),
        plot_range_x=analysis["x_range"].tolist(),
        plot_range_y=analysis["y_range"].tolist(),
        learned_density=analysis["density_grid"].tolist(),
        log_likelihoods=log_likelihoods.tolist(),
        component_confidence_ellipses=[
            ellipse.tolist() for ellipse in analysis["confidence_ellipses"]
        ],
        prior_moment_matrix=analysis["prior_moment"].tolist(),
        posterior_observations=analysis["posterior_obs"].tolist(),
        posterior_moment_matrices=[
            moment.tolist() for moment in analysis["posterior_moments"]
        ],
    )


def main():
    """Run Gaussian-Boltzmann harmonium example."""
    initialize_jax(device="gpu")
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    results = compute_gaussian_boltzmann_results(key)

    print("\nSaving results...")
    paths.save_analysis(results)
    print(f"Results saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
