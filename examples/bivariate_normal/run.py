"""Test script for bivariate Normal distributions with different covariance structures."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import multivariate_normal

from goal.geometry import Diagonal, Mean, Natural, Point, PositiveDefinite, Scale
from goal.models import (
    Euclidean,
    Normal,
)

from ..shared import (
    create_grid,
    example_paths,
    get_normal_bounds,
    initialize_jax,
)
from .types import BivariateResults


def compute_densities[R: PositiveDefinite](
    model: Normal[R], natural_point: Point[Natural, Normal[R]], xs: Array, ys: Array
) -> Array:
    points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)
    zs = jax.vmap(model.density, in_axes=(None, 0))(natural_point, points)
    return zs.reshape(xs.shape)


def scipy_densities(mean: Array, cov: Array, xs: Array, ys: Array) -> Array:
    points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)
    return jax.vmap(multivariate_normal.pdf, in_axes=(0, None, None))(points, mean, cov)


def compute_gaussian_results(
    key: Array,
    mean: Array,
    covariance: Array,
    sample_size: int,
) -> BivariateResults:
    # Initialize grid for plotting
    bounds = get_normal_bounds(mean, covariance, n_std=3.0)
    xs, ys = create_grid(bounds, n_points=50)

    # Scipy densities
    sp_dens = scipy_densities(mean, covariance, xs, ys)
    sp_dens = sp_dens.reshape(xs.shape).tolist()

    # Models
    pod_model = Normal(2, PositiveDefinite)
    dia_model = Normal(2, Diagonal)
    iso_model = Normal(2, Scale)

    # Ground truth
    gt_cov = pod_model.cov_man.from_dense(covariance)
    mu: Point[Mean, Euclidean] = Point(mean)
    gt_mean_point = pod_model.join_mean_covariance(mu, gt_cov)

    gt_natural_point = pod_model.to_natural(gt_mean_point)
    gt_dens = compute_densities(pod_model, gt_natural_point, xs, ys)
    gt_dens = gt_dens.reshape(xs.shape).tolist()
    sample = pod_model.sample(key, gt_natural_point, sample_size)

    # Models
    def process_normal[R: PositiveDefinite](
        model: Normal[R],
        sample: Array,
    ) -> Array:
        mean_point = model.average_sufficient_statistic(sample)
        p_natural = model.to_natural(mean_point)
        densities = compute_densities(model, p_natural, xs, ys)
        return densities.reshape(xs.shape)

    pd_dens = process_normal(pod_model, sample).tolist()
    dia_dens = process_normal(dia_model, sample).tolist()
    scl_dens = process_normal(iso_model, sample).tolist()

    return BivariateResults(
        sample=sample.tolist(),
        plot_xs=xs.tolist(),
        plot_ys=ys.tolist(),
        scipy_densities=sp_dens,
        ground_truth_densities=gt_dens,
        positive_definite_densities=pd_dens,
        diagonal_densities=dia_dens,
        scale_densities=scl_dens,
    )


### Main ###


def main():
    initialize_jax()
    paths = example_paths(__file__)
    """Run bivariate Normal tests."""
    key = jax.random.PRNGKey(0)

    # Define ground truth parameters
    true_mean = jnp.array([1.0, -0.5])
    true_cov = jnp.array([[2.0, 1.0], [1.0, 1.0]])

    # Generate sample and fit models
    results = compute_gaussian_results(
        key,
        true_mean,
        true_cov,
        sample_size=100,
    )

    # Save results
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
