"""Test script for comparing Factor Analysis and PCA."""

import json
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    LinearMap,
    Natural,
    Point,
    Rectangular,
)
from goal.models import (
    Euclidean,
    FactorAnalysis,
    LinearGaussianModel,
    PrincipalComponentAnalysis,
)

from .common import LGMResults, analysis_path


def ground_truth(
    key: Array,
    sample_size: int,
) -> tuple[FactorAnalysis, Point[Natural, FactorAnalysis], Array]:
    """Generate synthetic data from a ground truth FA model.

    Creates data by:
    1. Sampling latent z ~ N(0,1)
    2. Mapping to a line: x = z * [1, slope]
    3. Adding different noise scales parallel/perpendicular to line
    """
    fan_man = FactorAnalysis(2, 1)

    with fan_man.obs_man as om:
        obs_mean = om.loc_man.mean_point(jnp.asarray([-0.5, 1.5]))
        obs_cov = om.cov_man.mean_point(jnp.asarray([2, 0.5]))
        obs_means = om.join_mean_covariance(obs_mean, obs_cov)
        obs_params = om.to_natural(obs_means)

    prr_params = fan_man.lat_man.to_natural(fan_man.lat_man.standard_normal())

    int_params: Point[Natural, LinearMap[Rectangular, Euclidean, Euclidean]] = Point(
        jnp.asarray([1.0, 0.5])
    )
    lkl_params = fan_man.lkl_man.join_params(obs_params, int_params)
    fan_params = fan_man.join_conjugated(lkl_params, prr_params)
    sample = fan_man.observable_sample(key, fan_params, sample_size)

    return fan_man, fan_params, sample


def get_plot_bounds(
    sample: Array, margin: float = 0.2
) -> tuple[float, float, float, float]:
    """Compute bounds for plotting with margin."""
    min_vals = jnp.min(sample, axis=0)
    max_vals = jnp.max(sample, axis=0)
    range_vals = max_vals - min_vals
    x_min = float(min_vals[0] - margin * range_vals[0])
    x_max = float(max_vals[0] + margin * range_vals[0])
    y_min = float(min_vals[1] - margin * range_vals[1])
    y_max = float(max_vals[1] + margin * range_vals[1])
    return (x_min, x_max, y_min, y_max)


def fit_model[Model: LinearGaussianModel[Any]](
    key: Array,
    model: Model,
    n_steps: int,
    sample: Array,
) -> tuple[Array, Point[Natural, Model], Point[Natural, Model]]:
    """Fit model using EM algorithm."""
    params0 = model.shape_initialize(key)

    def em_step(
        carry: Point[Natural, Model], _: Any
    ) -> tuple[Point[Natural, Model], Array]:
        params = carry
        ll = model.average_log_observable_density(params, sample)
        next_params = model.expectation_maximization(params, sample)
        return next_params, ll

    params1, lls = jax.lax.scan(em_step, params0, None, length=n_steps)
    return lls.ravel(), params0, params1


fit_model = jax.jit(fit_model, static_argnames=["model", "n_steps"])


def create_evaluation_grids(
    bounds: tuple[float, float, float, float], n_points: int = 50
) -> tuple[Array, Array]:
    """Create grids for evaluating densities in observable and latent space."""
    x_min, x_max, y_min, y_max = bounds
    x = jnp.linspace(x_min, x_max, n_points)
    y = jnp.linspace(y_min, y_max, n_points)
    xs, ys = jnp.meshgrid(x, y)
    grid_points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    # For latent space, use range of true latents
    z = jnp.linspace(x_min, x_max, n_points)
    return grid_points, z


def compute_lgm_results(
    key: Array,
    sample_size: int,
    n_steps: int,
) -> LGMResults:
    """Run complete LGM analysis."""
    # Generate synthetic data
    key_data, key_train = jax.random.split(key)
    gt_model, gt_params, sample = ground_truth(key_data, sample_size)

    # Create models
    fa_model = FactorAnalysis(2, 1)
    pca_model = PrincipalComponentAnalysis(2, 1)

    # Split training key
    key_fa, key_pca = jax.random.split(key_train)

    # Train both models
    fa_lls, fa_params0, fa_params1 = fit_model(key_fa, fa_model, n_steps, sample)
    pca_lls, pca_params0, pca_params1 = fit_model(key_pca, pca_model, n_steps, sample)

    # Create evaluation grids
    plot_bounds = get_plot_bounds(sample)
    obs_grid, lat_grid = create_evaluation_grids(plot_bounds)

    # Compute densities
    gt_obs_densities = jax.vmap(gt_model.observable_density, in_axes=(None, 0))(
        gt_params, obs_grid
    )
    fa_obs_densities0 = jax.vmap(fa_model.observable_density, in_axes=(None, 0))(
        fa_params0, obs_grid
    )
    pca_obs_densities0 = jax.vmap(pca_model.observable_density, in_axes=(None, 0))(
        pca_params0, obs_grid
    )

    fa_obs_densities1 = jax.vmap(fa_model.observable_density, in_axes=(None, 0))(
        fa_params1, obs_grid
    )
    pca_obs_densities1 = jax.vmap(pca_model.observable_density, in_axes=(None, 0))(
        pca_params1, obs_grid
    )

    # Structure results
    return LGMResults(
        sample=sample.tolist(),
        plot_bounds=plot_bounds,
        training_lls={
            "factor_analysis": fa_lls.tolist(),
            "pca": pca_lls.tolist(),
        },
        grid_points={
            "observable": obs_grid.tolist(),
            "latent": lat_grid.tolist(),
        },
        ground_truth_densities=gt_obs_densities.tolist(),
        initial_densities={
            "factor_analysis": fa_obs_densities0.tolist(),
            "pca": pca_obs_densities0.tolist(),
        },
        learned_densities={
            "factor_analysis": fa_obs_densities1.tolist(),
            "pca": pca_obs_densities1.tolist(),
        },
    )


def main() -> None:
    """Run LGM comparison tests."""
    # Configure JAX
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_disable_jit", False)

    # Set random seed
    key = jax.random.PRNGKey(0)

    # Generate sample and fit models
    results = compute_lgm_results(
        key,
        sample_size=500,
        n_steps=50,
    )

    # Save results
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
