"""Boltzmann HMoG on Two Moons: Comparing Gaussian vs Binary Latent Models.

This example demonstrates how Boltzmann HMoG models can learn nonlinear structure
that Gaussian mixture models struggle with. The two moons dataset has curved
geometry that requires either many Gaussian components or nonlinear features.

Models compared:
1. Gaussian Mixture Model (GMM) - baseline, struggles with curved geometry
2. SymmetricBoltzmannHMoG - binary latents can encode nonlinear features
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Optimizer, OptState, PositiveDefinite
from goal.models import (
    Normal,
    SymmetricBoltzmannHMoG,
    symmetric_boltzmann_hmog,
)
from goal.models.harmonium.mixture import CompleteMixture

from ..shared import example_paths, initialize_jax
from .types import BoltzmannHMoGMoonsResults

# Configuration
n_samples = 500
noise = 0.1
n_neurons = 8  # Binary latent dimensions
n_components = 2  # Mixture components
n_gmm_components = 2  # More components for GMM to approximate curves
learning_rate = 3e-3  # Match gaussian_boltzmann example
n_steps_per_epoch = 200  # Match gaussian_boltzmann example
n_epochs = 1000  # Total steps = 200,000
plot_resolution = 80


def generate_two_moons(key: Array, n_samples: int, noise: float) -> tuple[Array, Array]:
    """Generate two moons dataset.

    Returns:
        samples: Array of shape (n_samples, 2)
        labels: Array of shape (n_samples,) with 0 or 1 for each moon
    """
    n_per_moon = n_samples // 2

    # First moon: upper arc
    key1, key2, key = jax.random.split(key, 3)
    theta1 = jnp.linspace(0, jnp.pi, n_per_moon)
    x1 = jnp.cos(theta1)
    y1 = jnp.sin(theta1)
    moon1 = jnp.stack([x1, y1], axis=1)
    moon1 = moon1 + noise * jax.random.normal(key1, moon1.shape)

    # Second moon: lower arc, shifted
    theta2 = jnp.linspace(0, jnp.pi, n_per_moon)
    x2 = 1 - jnp.cos(theta2)
    y2 = -jnp.sin(theta2) + 0.5
    moon2 = jnp.stack([x2, y2], axis=1)
    moon2 = moon2 + noise * jax.random.normal(key2, moon2.shape)

    samples = jnp.concatenate([moon1, moon2], axis=0)
    labels = jnp.concatenate([jnp.zeros(n_per_moon), jnp.ones(n_per_moon)])

    # Shuffle
    perm = jax.random.permutation(key, n_samples)
    return samples[perm], labels[perm].astype(jnp.int32)


def train_gmm(
    key: Array,
    model: CompleteMixture[Normal],
    data: Array,
) -> tuple[Array, list[float]]:
    """Train Gaussian mixture model via gradient descent."""
    params = model.initialize_from_sample(key, data)
    optimizer: Optimizer[CompleteMixture[Normal]] = Optimizer.adamw(
        man=model, learning_rate=learning_rate
    )
    opt_state = optimizer.init(params)

    def loss_fn(p: Array) -> Array:
        return -model.average_log_observable_density(p, data)

    def step(
        state: tuple[OptState, Array], _: Any
    ) -> tuple[tuple[OptState, Array], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), -loss

    total_steps = n_steps_per_epoch * n_epochs
    (_, final_params), lls = jax.lax.scan(
        step, (opt_state, params), None, length=total_steps
    )
    # Return epoch-averaged log likelihoods
    epoch_lls = lls.reshape(n_epochs, n_steps_per_epoch).mean(axis=1)
    return final_params, epoch_lls.tolist()


def train_symmetric_hmog(
    key: Array,
    model: SymmetricBoltzmannHMoG,
    data: Array,
) -> tuple[Array, list[float]]:
    """Train SymmetricBoltzmannHMoG via gradient descent."""
    params = model.initialize(
        key, location=0.0, shape=1.0
    )  # shape=1.0 like gaussian_boltzmann
    optimizer: Optimizer[SymmetricBoltzmannHMoG] = Optimizer.adamw(
        man=model, learning_rate=learning_rate
    )
    opt_state = optimizer.init(params)

    def loss_fn(p: Array) -> Array:
        return -model.average_log_observable_density(p, data)

    def step(
        state: tuple[OptState, Array], _: Any
    ) -> tuple[tuple[OptState, Array], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), -loss

    total_steps = n_steps_per_epoch * n_epochs
    (_, final_params), lls = jax.lax.scan(
        step, (opt_state, params), None, length=total_steps
    )
    # Return epoch-averaged log likelihoods
    epoch_lls = lls.reshape(n_epochs, n_steps_per_epoch).mean(axis=1)
    return final_params, epoch_lls.tolist()


def compute_density_grid(
    model: Any, params: Array, x_range: Array, y_range: Array
) -> Array:
    """Compute observable density on a grid."""
    xx, yy = jnp.meshgrid(x_range, y_range)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)
    densities = jax.vmap(model.observable_density, in_axes=(None, 0))(
        params, grid_points
    )
    return densities.reshape(xx.shape)


def compute_gmm_assignments(
    model: CompleteMixture[Normal], params: Array, data: Array
) -> Array:
    """Compute soft cluster assignments for GMM.

    Manually computes p(z=k|x) ∝ π_k * p(x|z=k) for each component k.
    """
    # Get component parameters and prior
    comp_params, cat_params = model.split_natural_mixture(params)
    probs = model.lat_man.to_probs(model.lat_man.to_mean(cat_params))
    log_probs = jnp.log(probs)  # [n_components]

    # Convert components to 2D: [n_components, obs_dim]
    comp_2d = model.cmp_man.to_2d(comp_params)

    # Compute log density for each (data, component) pair
    def log_density(comp_params_k: Array, x: Array) -> Array:
        return model.obs_man.log_density(comp_params_k, x)

    # vmap over components, then over data
    def log_densities_at_x(x: Array) -> Array:
        return jax.vmap(log_density, in_axes=(0, None))(comp_2d, x)

    log_densities = jax.vmap(log_densities_at_x)(data)  # [n_obs, n_components]

    # Add log prior and compute softmax for posterior
    log_posterior = log_densities + log_probs
    return jax.nn.softmax(log_posterior, axis=1)


def compute_hmog_assignments(
    model: SymmetricBoltzmannHMoG,
    params: Array,
    data: Array,
) -> Array:
    """Compute soft cluster assignments for HMoG."""
    return jax.vmap(model.posterior_soft_assignments, in_axes=(None, 0))(params, data)


def compute_binary_activations(
    model: SymmetricBoltzmannHMoG, params: Array, data: Array
) -> Array:
    """Compute posterior binary activations (marginal E[x_i]) for each observation."""

    def get_activation(x: Array) -> Array:
        # Get posterior Boltzmann parameters
        boltz_params = model.posterior_boltzmann(params, x)
        # Convert to mean parameters (full moment matrix)
        means = model.lwr_hrm.lat_man.to_mean(boltz_params)
        # Extract just the marginal activations E[x_i] (diagonal of moment matrix)
        marginals, _ = model.lwr_hrm.lat_man.split_mean_second_moment(means)
        return marginals

    return jax.vmap(get_activation)(data)


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)
    key_data, key_gmm, key_boltz = jax.random.split(key, 3)

    print("Generating two moons data...")
    data, labels = generate_two_moons(key_data, n_samples, noise)

    # Create models
    print("Creating models...")
    obs_rep = PositiveDefinite()

    gmm = CompleteMixture(Normal(2, obs_rep), n_gmm_components)

    boltz_hmog = symmetric_boltzmann_hmog(
        obs_dim=2,
        obs_rep=obs_rep,
        boltz_dim=n_neurons,
        n_components=n_components,
    )

    # Train models
    print("Training GMM...")
    gmm_params, gmm_lls = train_gmm(key_gmm, gmm, data)

    print("Training SymmetricBoltzmannHMoG...")
    boltz_params, boltz_lls = train_symmetric_hmog(key_boltz, boltz_hmog, data)

    # Compute densities
    print("Computing densities...")
    x_range = jnp.linspace(-1.5, 2.5, plot_resolution)
    y_range = jnp.linspace(-1.0, 1.5, plot_resolution)

    gmm_density = compute_density_grid(gmm, gmm_params, x_range, y_range)
    boltz_density = compute_density_grid(boltz_hmog, boltz_params, x_range, y_range)

    # Compute assignments
    print("Computing cluster assignments...")
    gmm_assignments = compute_gmm_assignments(gmm, gmm_params, data)
    boltz_assignments = compute_hmog_assignments(boltz_hmog, boltz_params, data)

    # Compute binary activations
    print("Computing binary activations...")
    boltz_activations = compute_binary_activations(boltz_hmog, boltz_params, data)

    # Save results
    print("Saving results...")
    results = BoltzmannHMoGMoonsResults(
        observations=data.tolist(),
        moon_labels=labels.tolist(),
        plot_range_x=x_range.tolist(),
        plot_range_y=y_range.tolist(),
        gmm_density=gmm_density.tolist(),
        boltzmann_density=boltz_density.tolist(),
        gmm_log_likelihoods=gmm_lls,
        boltzmann_log_likelihoods=boltz_lls,
        gmm_assignments=gmm_assignments.tolist(),
        boltzmann_assignments=boltz_assignments.tolist(),
        boltzmann_binary_activations=boltz_activations.tolist(),
    )
    paths.save_analysis(results)
    print(f"Results saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
