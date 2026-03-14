"""Population code example: inference of circular stimulus from neural responses.

This example demonstrates:
1. Creating a population code with cosine tuning curves
2. Simulating neural responses to various stimuli
3. Inferring the stimulus from neural responses using approximate Bayesian inference
"""

import jax
import jax.numpy as jnp

from goal.models import PoissonVonMisesHarmonium, VonMisesPopulationCode

from ..shared import example_paths, jax_cli
from .types import PopulationCodeResults

# Model configuration
n_neurons = 8
n_test_stimuli = 20
n_grid_points = 100

# Tuning curve parameters
baseline_rate = 1.0  # Baseline firing rate
max_rate = 10.0  # Maximum firing rate at preferred direction


def create_population_code(key: jax.Array) -> tuple[VonMisesPopulationCode, jax.Array]:
    """Create a population code with evenly spaced tuning curves."""
    model = VonMisesPopulationCode(hrm=PoissonVonMisesHarmonium(n_neurons, 1))
    preferred = jnp.linspace(0, 2 * jnp.pi, n_neurons, endpoint=False)
    params = model.initialize_from_tuning_curves(
        key=key,
        gains=jnp.ones(n_neurons) * (jnp.log(max_rate) - jnp.log(baseline_rate)),
        preferred=preferred,
        baselines=jnp.ones(n_neurons) * jnp.log(baseline_rate),
        prior_mean=0.0,
        prior_concentration=0.1,
    )
    return model, params


def compute_tuning_curves(
    model: VonMisesPopulationCode, params: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute tuning curves over a grid of stimuli."""
    grid = jnp.linspace(0, 2 * jnp.pi, n_grid_points, endpoint=False)
    _, hrm_params = model.split_coords(params)

    def rates_at(z: jax.Array) -> jax.Array:
        lkl_params = model.hrm.likelihood_at(hrm_params, z)
        return model.obs_man.to_mean(lkl_params)

    rates = jax.vmap(rates_at)(grid)
    return grid, rates


def compute_regression_diagnostics(
    model: VonMisesPopulationCode, params: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute regression fit diagnostics for visualization."""
    _, hrm_params = model.split_coords(params)
    obs_params, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(n_neurons, 2)
    vm = model.pst_man.rep_man  # underlying VonMises

    grid = jnp.linspace(0, 2 * jnp.pi, n_grid_points, endpoint=False)

    def log_partition_at(z: jax.Array) -> jax.Array:
        suff = vm.sufficient_statistic(z)
        log_rates = obs_params + int_matrix @ suff
        return jnp.sum(jnp.exp(log_rates))

    actual = jax.vmap(log_partition_at)(grid)
    suff_stats = jax.vmap(vm.sufficient_statistic)(grid)
    design = jnp.column_stack([jnp.ones(n_grid_points), suff_stats])
    coeffs, _, _, _ = jnp.linalg.lstsq(design, actual, rcond=None)
    fitted = design @ coeffs
    return grid, actual, fitted


def run_inference(
    model: VonMisesPopulationCode, params: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample from generative model and infer posterior."""
    vm = model.pst_man.rep_man  # underlying VonMises

    # Sample joint (x, z) from generative model
    key, sample_key = jax.random.split(key)
    joint_samples = model.sample(sample_key, params, n_test_stimuli)
    spike_counts = joint_samples[:, : model.obs_man.data_dim]
    stimuli = joint_samples[:, model.obs_man.data_dim :]

    # Compute approximate posterior for each observation
    def posterior_stats(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        q_params = model.approximate_posterior_at(params, x)
        # VonMisesProduct(1) has 2 natural params; extract mean/concentration from underlying VonMises
        mu, kappa = vm.split_mean_concentration(q_params)
        return mu, kappa

    posterior_means, posterior_kappas = jax.vmap(posterior_stats)(spike_counts)

    # stimuli are angles from VonMisesProduct(1) — shape (n, 1), squeeze to (n,)
    return stimuli.squeeze(-1), spike_counts, posterior_means, posterior_kappas


def circular_error(estimate: jax.Array, true_val: jax.Array) -> jax.Array:
    """Compute circular distance between two angles."""
    return jnp.abs(jnp.angle(jnp.exp(1j * (estimate - true_val))))


def main():
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(42)

    print(f"Creating population code with {n_neurons} neurons...")
    key, init_key = jax.random.split(key)
    model, params = create_population_code(init_key)

    # Extract preferred directions from interaction weights
    _, hrm_params = model.split_coords(params)
    _, int_params, _ = model.hrm.split_coords(hrm_params)
    int_matrix = int_params.reshape(n_neurons, 2)
    preferred = jnp.arctan2(int_matrix[:, 1], int_matrix[:, 0])
    preferred = jnp.mod(preferred, 2 * jnp.pi)

    # Compute tuning curves
    print("Computing tuning curves...")
    grid, tuning_curves = compute_tuning_curves(model, params)

    # Get regression diagnostics
    print("Computing regression diagnostics...")
    reg_grid, reg_actual, reg_fitted = compute_regression_diagnostics(model, params)

    # Run inference
    print(f"Running inference on {n_test_stimuli} test stimuli...")
    stimuli, spikes, post_means, post_kappas = run_inference(model, params, key)

    # Compute errors
    errors = circular_error(post_means, stimuli)
    mean_error = float(jnp.mean(errors))
    print(
        f"Mean circular error: {mean_error:.4f} radians ({jnp.degrees(mean_error):.2f} degrees)"
    )

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
