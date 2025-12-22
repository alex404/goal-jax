"""Bivariate Normal distributions with different covariance structures."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import multivariate_normal

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import Normal

from ..shared import create_grid, example_paths, get_normal_bounds, initialize_jax
from .types import BivariateResults


def compute_densities(model: Normal, params: Array, xs: Array, ys: Array) -> Array:
    """Compute densities on a grid."""
    points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)
    zs = jax.vmap(model.density, in_axes=(None, 0))(params, points)
    return zs.reshape(xs.shape)


def fit_and_evaluate(model: Normal, sample: Array, xs: Array, ys: Array) -> Array:
    """Fit model to sample and evaluate densities on grid."""
    mean_params = model.average_sufficient_statistic(sample)
    natural_params = model.to_natural(mean_params)
    return compute_densities(model, natural_params, xs, ys)


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)

    # Ground truth parameters
    mean = jnp.array([1.0, -0.5])
    cov = jnp.array([[2.0, 1.0], [1.0, 1.0]])
    sample_size = 100

    # Create grid
    bounds = get_normal_bounds(mean, cov, n_std=3.0)
    xs, ys = create_grid(bounds, n_points=50)

    # Models
    pod_model = Normal(2, PositiveDefinite())
    dia_model = Normal(2, Diagonal())
    iso_model = Normal(2, Scale())

    # Ground truth distribution
    gt_cov = pod_model.cov_man.from_matrix(cov)
    gt_mean_params = pod_model.join_mean_covariance(mean, gt_cov)
    gt_natural = pod_model.to_natural(gt_mean_params)

    # Generate sample
    sample = pod_model.sample(key, gt_natural, sample_size)

    # Compute densities
    points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)
    scipy_dens = jax.vmap(multivariate_normal.pdf, in_axes=(0, None, None))(
        points, mean, cov
    ).reshape(xs.shape)

    gt_dens = compute_densities(pod_model, gt_natural, xs, ys)
    pd_dens = fit_and_evaluate(pod_model, sample, xs, ys)
    dia_dens = fit_and_evaluate(dia_model, sample, xs, ys)
    scl_dens = fit_and_evaluate(iso_model, sample, xs, ys)

    results = BivariateResults(
        sample=sample.tolist(),
        plot_xs=xs.tolist(),
        plot_ys=ys.tolist(),
        scipy_densities=scipy_dens.tolist(),
        ground_truth_densities=gt_dens.tolist(),
        positive_definite_densities=pd_dens.tolist(),
        diagonal_densities=dia_dens.tolist(),
        scale_densities=scl_dens.tolist(),
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
