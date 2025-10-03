"""Gaussian-Boltzmann harmonium example."""

from typing import Any

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

### Constants ###

# Model parameters
N_LATENTS = 9
OBS_DIM = 2

# Data generation
RADII = [0.3, 3.0]
NOISE_STDS = [0.2, 0.2]
N_OBS = 1000
RATIOS = [0.2, 0.8]  # Proportion of samples from each circle

# Training
LEARNING_RATE = 3e-4
N_STEPS_PER_EPOCH = 500
N_EPOCHS = 1000

# Plotting
PLOT_RES = 100
PLOT_MIN = -5.0
PLOT_MAX = 5.0


### Data Generation ###


def circle_2d(radius: float, theta: float) -> Array:
    """Generate point on circle at given angle."""
    return jnp.array([radius * jnp.cos(theta), radius * jnp.sin(theta)])


def generate_noisy_circle(
    key: Array, radius: float, noise_std: float, n_samples: int
) -> Array:
    """Generate noisy circle data."""
    # Generate angles equally spaced around circle
    angles = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)

    # Generate points on circle
    points = jax.vmap(lambda theta: circle_2d(radius, theta))(angles)

    # Add Gaussian noise
    noise = noise_std * jax.random.normal(key, (n_samples, 2))
    return points + noise


def generate_data(key: Array) -> Array:
    """Generate full dataset with two noisy circles."""
    n_samples_list = [int(ratio * N_OBS) for ratio in RATIOS]

    keys = jax.random.split(key, len(RADII))
    circles = [
        generate_noisy_circle(k, r, std, n)
        for k, r, std, n in zip(keys, RADII, NOISE_STDS, n_samples_list)
    ]

    return jnp.concatenate(circles, axis=0)


### Training ###


def train_model(
    key: Array,
    model: BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite],
    data: Array,
) -> tuple[
    Point[Natural, BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]], Array
]:
    """Train model using Adam optimizer with JIT-compiled training loop."""

    # Initialize parameters and optimizer
    params = model.initialize(key, location=0.0, shape=1.0)
    optimizer: Optimizer[
        Natural, BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]
    ] = Optimizer.adamw(man=model, learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    # Define loss function
    def cross_entropy_loss(
        p: Point[Natural, BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]],
    ) -> Array:
        return -model.average_log_observable_density(p, data)

    # Define single gradient step
    def grad_step(
        opt_state_and_params: tuple[
            OptState,
            Point[
                Natural, BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]
            ],
        ],
        _: Any,
    ) -> tuple[
        tuple[
            OptState,
            Point[
                Natural, BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]
            ],
        ],
        Array,
    ]:
        opt_state, params = opt_state_and_params
        loss_val, grads = model.value_and_grad(cross_entropy_loss, params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), -loss_val  # Return log likelihood

    # Run training loop with scan
    total_steps = N_EPOCHS * N_STEPS_PER_EPOCH
    (_, final_params), lls = jax.lax.scan(
        grad_step, (opt_state, params), None, length=total_steps
    )

    # Reshape to get per-epoch averages
    epoch_lls = lls.reshape(N_EPOCHS, N_STEPS_PER_EPOCH).mean(axis=1)

    # Print progress every 100 epochs
    for epoch in range(0, N_EPOCHS, 100):
        print(f"Epoch {epoch}: Average Log-Likelihood = {epoch_lls[epoch]:.6f}")

    return final_params, epoch_lls


### Analysis ###


def compute_moment_matrix(
    model: BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite],
    boltzmann_params: Point[Natural, Boltzmann],
) -> Array:
    """Compute moment matrix E[zz^T] for Boltzmann distribution."""
    # Get all binary states
    states = model.lat_man.states  # Shape: (2^n, n)

    # Compute probabilities for each state
    log_probs = jax.vmap(
        lambda z: model.lat_man.dot(
            boltzmann_params, model.lat_man.sufficient_statistic(z)
        )
    )(states)
    log_probs = log_probs - jax.nn.logsumexp(log_probs)
    probs = jnp.exp(log_probs)

    # Compute weighted outer products
    moment_matrix = jnp.sum(
        jax.vmap(lambda p, z: p * jnp.outer(z, z))(probs, states), axis=0
    )

    return moment_matrix


def compute_confidence_ellipse(
    model: BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite],
    lkl_params: Point[
        Natural, AffineMap[Rectangular, Euclidean, Euclidean, Normal[PositiveDefinite]]
    ],
    latent_state: Array,
    n_points: int = 1000,
    n_std: float = 1.0,
) -> Array:
    """Compute confidence ellipse for observable given latent state."""
    # Evaluate likelihood at latent state to get conditional p(x|z)
    # For Boltzmann, the location is just the binary vector
    z_loc = model.lat_man.loc_man.point(latent_state)
    conditional_obs_nat_params = model.lkl_man(lkl_params, z_loc)

    # Convert to mean parameters
    conditional_mean_params = model.obs_man.to_mean(conditional_obs_nat_params)
    obs_mean, obs_cov = model.obs_man.split_mean_covariance(conditional_mean_params)

    # Convert covariance to dense matrix
    cov_dense = model.obs_man.cov_man.to_dense(obs_cov)

    # Generate ellipse points
    angles = jnp.linspace(0, 2 * jnp.pi, n_points)

    # Compute eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(cov_dense)

    # Generate unit circle and transform
    unit_circle = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    scaled_circle = unit_circle * jnp.sqrt(eigvals)[None, :] * n_std
    ellipse = (eigvecs @ scaled_circle.T).T + obs_mean.array

    return ellipse


def analyze_model(
    model: BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite],
    params: Point[
        Natural, BoltzmannDifferentiableLinearGaussianModel[PositiveDefinite]
    ],
    data: Array,
) -> dict:
    """Perform analysis of trained model."""
    # Compute observable densities on grid
    x_range = jnp.linspace(PLOT_MIN, PLOT_MAX, PLOT_RES)
    y_range = jnp.linspace(PLOT_MIN, PLOT_MAX, PLOT_RES)
    xx, yy = jnp.meshgrid(x_range, y_range)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    densities = jax.vmap(model.observable_density, in_axes=(None, 0))(
        params, grid_points
    )
    density_grid = densities.reshape(xx.shape)

    # Split parameters
    lkl_params, prior_params = model.split_conjugated(params)

    # Compute prior moment matrix
    prior_moment = compute_moment_matrix(model, prior_params)

    # Generate confidence ellipses for each latent state (single spikes)
    n_states = N_LATENTS
    single_spike_states = jnp.eye(n_states)

    confidence_ellipses = [
        compute_confidence_ellipse(model, lkl_params, state)
        for state in single_spike_states
    ]

    # Select posterior observations (first and middle of each group)
    n_samples_list = [int(ratio * N_OBS) for ratio in RATIOS]
    posterior_obs_indices = []
    cumsum = 0
    for n_samples in n_samples_list:
        posterior_obs_indices.append(cumsum)
        posterior_obs_indices.append(cumsum + n_samples // 2)
        cumsum += n_samples

    posterior_obs = data[jnp.array(posterior_obs_indices)]

    # Compute posterior moment matrices
    posterior_moments = []
    for obs in posterior_obs:
        post_params = model.posterior_at(params, obs)
        post_moment = compute_moment_matrix(model, post_params)
        posterior_moments.append(post_moment)

    return {
        "density_grid": density_grid,
        "x_range": x_range,
        "y_range": y_range,
        "prior_moment": prior_moment,
        "confidence_ellipses": confidence_ellipses,
        "posterior_obs": posterior_obs,
        "posterior_moments": posterior_moments,
    }


### Main ###


def compute_gaussian_boltzmann_results(key: Array) -> GaussianBoltzmannResults:
    """Run full Gaussian-Boltzmann harmonium example."""
    print("Generating data...")
    key_data, key_train = jax.random.split(key)
    data = generate_data(key_data)
    print(f"Generated {len(data)} observations")

    print("\nCreating model...")
    model = BoltzmannDifferentiableLinearGaussianModel(
        obs_dim=OBS_DIM,
        obs_rep=PositiveDefinite,
        lat_dim=N_LATENTS,
    )
    print(f"Model: {OBS_DIM}D observables, {N_LATENTS} binary latents")

    print(f"\nTraining for {N_EPOCHS} epochs...")
    final_params, log_likelihoods = train_model(key_train, model, data)
    print(f"Final log-likelihood: {log_likelihoods[-1]:.6f}")

    print("\nAnalyzing model...")
    analysis = analyze_model(model, final_params, data)

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
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    results = compute_gaussian_boltzmann_results(key)

    print("\nSaving results...")
    paths.save_analysis(results)
    print(f"Results saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
