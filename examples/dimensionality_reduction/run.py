"""Test script for comparing Factor Analysis and PCA."""

import json
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Diagonal, Natural, Point, Scale
from goal.models import (
    FactorAnalysis,
    LinearGaussianModel,
)

from .common import LGMResults, analysis_path


def generate_ground_truth_data(
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
    om = fan_man.obs_man
    obs_mean = om.loc_man.mean_point(jnp.asarray([-0.5, 1.5]))
    obs_cov = om.cov_man.mean_point(jnp.asarray([[2, 0.2, 0.5]]))
    obs_means = om.join_mean_covariance(obs_mean, obs_cov)
    obs_params = om.to_natural(obs_means)

    lm = fan_man.lat_man
    obs_mean = om.loc_man.shape_initialize(key)
    obs_cov = om.cov_man.shape_initialize(key)
    obs_means = om.join_mean_covariance(obs_mean, obs_cov)
    obs_params = om.to_natural(obs_means)

    lat_mu = fan_man.lat_man.standard_normal()

    return noisy_points, latents


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


def initialize_parameters[R: Scale | Diagonal](
    key: Array,
    model: LinearGaussianModel[R],
) -> Point[Natural, LinearGaussianModel[R]]:
    """Initialize model parameters for EM algorithm."""
    # Split keys for independent initialization
    key_obs_mu, key_obs_cov, key_lat_mu, key_lat_cov, key_int = jax.random.split(key, 5)

    om = model.obs_man
    obs_mu = om.loc_man.shape_initialize(key_obs_mu)
    obs_cov = om.cov_man.shape_initialize(key_obs_cov)
    obs_bias = om.to_natural(om.join_mean_covariance(obs_mu, obs_cov))

    lm = model.lat_man
    lat_mu = lm.loc_man.shape_initialize(key_lat_mu)
    lat_cov = lm.cov_man.shape_initialize(key_lat_cov)
    lat_bias = lm.to_natural(lm.join_mean_covariance(lat_mu, lat_cov))

    # Initialize interaction matrix with small values
    int_mat = model.int_man.shape_initialize(key_int)

    return model.join_params(obs_bias, lat_bias, int_mat)


def fit_model[R: Scale | Diagonal](
    key: Array,
    model: LinearGaussianModel[R],
    n_steps: int,
    sample: Array,
) -> tuple[Array, Point[Natural, LinearGaussianModel[R]]]:
    """Fit model using EM algorithm."""
    params = initialize_parameters(key, model)

    def em_step(
        carry: Point[Natural, LinearGaussianModel[R]], _: Any
    ) -> tuple[Point[Natural, LinearGaussianModel[R]], Array]:
        params = carry
        ll = model.average_log_observable_density(params, sample)
        next_params = model.lgm_expectation_maximization(params, sample)
        return next_params, ll

    final_params, lls = jax.lax.scan(em_step, params, None, length=n_steps)
    return lls.ravel(), final_params


fit_model = jax.jit(fit_model, static_argnames=["model", "n_steps"])


def create_evaluation_grids(
    bounds: Bounds2D, n_points: int = 50
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
    true_params: TrueParameters,
    sample_size: int,
    n_steps: int,
) -> LGMResults:
    """Run complete LGM analysis."""
    # Generate synthetic data
    key_data, key_train = jax.random.split(key)
    sample, latents = generate_ground_truth_data(key_data, true_params, sample_size)

    # Create models
    fa_model = LinearGaussianModel(2, Diagonal, 1)
    pca_model = LinearGaussianModel(2, Scale, 1)

    # Split training key
    key_fa, key_pca = jax.random.split(key_train)

    # Train both models
    fa_lls, fa_params = fit_model(key_fa, fa_model, n_steps, sample)
    pca_lls, pca_params = fit_model(key_pca, pca_model, n_steps, sample)

    print(*zip(fa_lls.tolist(), pca_lls.tolist()))
    # Create evaluation grids
    plot_bounds = get_plot_bounds(sample)
    obs_grid, lat_grid = create_evaluation_grids(plot_bounds)

    # Compute densities
    fa_obs_densities = jax.vmap(fa_model.observable_density, in_axes=(None, 0))(
        fa_params, obs_grid
    )
    pca_obs_densities = jax.vmap(pca_model.observable_density, in_axes=(None, 0))(
        pca_params, obs_grid
    )

    # Compute latent densities using prior
    fa_prior = fa_model.prior(fa_params)
    pca_prior = pca_model.prior(pca_params)
    fa_lat_densities = jax.vmap(fa_model.lat_man.density, in_axes=(None, 0))(
        fa_prior, lat_grid[:, None]
    )
    pca_lat_densities = jax.vmap(pca_model.lat_man.density, in_axes=(None, 0))(
        pca_prior, lat_grid[:, None]
    )

    # Create direction vector
    direction = jnp.array([1.0, true_params.slope])
    direction = direction / jnp.sqrt(jnp.sum(jnp.square(direction)))

    # Structure results
    return LGMResults(
        sample=sample.tolist(),
        true_latents=latents.tolist(),
        plot_bounds=plot_bounds,
        training_lls={
            "factor_analysis": fa_lls.tolist(),
            "pca": pca_lls.tolist(),
        },
        grid_points={
            "observable": obs_grid.tolist(),
            "latent": lat_grid.tolist(),
        },
        observable_densities={
            "factor_analysis": fa_obs_densities.tolist(),
            "pca": pca_obs_densities.tolist(),
        },
        latent_densities={
            "factor_analysis": fa_lat_densities.tolist(),
            "pca": pca_lat_densities.tolist(),
        },
        true_direction=direction.tolist(),
    )


def main() -> None:
    """Run LGM comparison tests."""
    # Configure JAX
    jax.config.update("jax_platform_name", "cpu")

    # Set random seed
    key = jax.random.PRNGKey(0)

    # Define parameters
    true_params = TrueParameters()

    # Generate sample and fit models
    results = compute_lgm_results(
        key,
        true_params,
        sample_size=500,
        n_steps=100,
    )

    # Save results
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
