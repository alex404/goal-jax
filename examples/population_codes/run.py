"""Population code example: inference of circular stimulus from neural responses.

This example demonstrates:
1. Creating a population code with cosine tuning curves
2. Simulating neural responses to various stimuli
3. Inferring the stimulus from neural responses using approximate Bayesian inference
"""

import jax
import jax.numpy as jnp

from goal.models import VonMisesPopulationCode

from ..shared import example_paths, initialize_jax
from .types import PopulationCodeResults

# Model configuration
n_neurons = 8
n_test_stimuli = 20
n_grid_points = 100

# Tuning curve parameters
baseline_rate = 1.0  # Baseline firing rate
max_rate = 10.0  # Maximum firing rate at preferred direction


def create_population_code() -> tuple[VonMisesPopulationCode, jax.Array]:
    """Create a population code with evenly spaced tuning curves."""
    model = VonMisesPopulationCode(n_neurons)

    # Evenly spaced preferred directions
    preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)

    # Gains to achieve max_rate at preferred direction
    # log(rate) = baseline + gain * cos(0) = baseline + gain
    # So gain = log(max_rate) - baseline
    baselines = jnp.ones(n_neurons) * jnp.log(baseline_rate)
    gains = jnp.ones(n_neurons) * (jnp.log(max_rate) - jnp.log(baseline_rate))

    # Uniform prior (low concentration)
    prior_params = model.lat_man.join_mean_concentration(0.0, 0.1)

    params = model.join_tuning_parameters(gains, preferred, baselines, prior_params)

    return model, params


def compute_tuning_curves(
    model: VonMisesPopulationCode, params: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute tuning curves over a grid of stimuli."""
    grid = jnp.linspace(0, 2 * jnp.pi, n_grid_points, endpoint=False)

    # Compute firing rates at each grid point
    rates = jax.vmap(lambda z: model.firing_rates(params, z))(grid)

    return grid, rates


def run_inference(
    model: VonMisesPopulationCode, params: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample from joint distribution and infer posterior."""
    # Sample (spike_counts, stimuli) pairs from the joint distribution
    spike_counts, stimuli = model.sample_joint(key, params, n=n_test_stimuli)

    # Compute posterior for each observation
    posterior_means = jax.vmap(lambda x: model.posterior_mean(params, x))(spike_counts)
    posterior_kappas = jax.vmap(lambda x: model.posterior_concentration(params, x))(
        spike_counts
    )

    return stimuli, spike_counts, posterior_means, posterior_kappas


def circular_error(estimate: jax.Array, true_val: jax.Array) -> jax.Array:
    """Compute circular distance between two angles."""
    return jnp.abs(jnp.angle(jnp.exp(1j * (estimate - true_val))))


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    print(f"Creating population code with {n_neurons} neurons...")
    model, params = create_population_code()

    # Get preferred directions
    _, preferred, _, _ = model.split_tuning_parameters(params)

    # Compute tuning curves
    print("Computing tuning curves...")
    grid, tuning_curves = compute_tuning_curves(model, params)

    # Get regression diagnostics
    print("Computing regression diagnostics...")
    reg_grid, reg_actual, reg_fitted = model.regression_diagnostics(params)

    # Run inference
    print(f"Running inference on {n_test_stimuli} test stimuli...")
    stimuli, spikes, post_means, post_kappas = run_inference(model, params, key)

    # Compute errors
    errors = circular_error(post_means, stimuli)
    mean_error = float(jnp.mean(errors))
    print(f"Mean circular error: {mean_error:.4f} radians ({jnp.degrees(mean_error):.2f} degrees)")

    # Save results
    results: PopulationCodeResults = {
        "tuning_curve_grid": grid.tolist(),
        "tuning_curves": tuning_curves.T.tolist(),  # Transpose for (n_neurons, n_grid)
        "preferred_directions": preferred.tolist(),
        "regression_grid": reg_grid.tolist(),
        "regression_actual": reg_actual.tolist(),
        "regression_fitted": reg_fitted.tolist(),
        "true_stimuli": stimuli.tolist(),
        "spike_counts": spikes.astype(int).tolist(),
        "posterior_means": post_means.tolist(),
        "posterior_concentrations": post_kappas.tolist(),
        "mean_circular_error": mean_error,
    }
    paths.save_analysis(results)
    print(f"Results saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
