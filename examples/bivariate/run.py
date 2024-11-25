"""Test script for bivariate Gaussian distributions with different covariance structures."""

import json

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import multivariate_normal

from goal.distributions import Gaussian
from goal.exponential_family import Natural
from goal.linear import Diagonal, PositiveDefinite, Scale
from goal.manifold import Point

from .common import BivariateResults, analysis_path


def create_test_grid(
    mean: Array, cov: Array, n_points: int = 20
) -> tuple[Array, Array]:
    """Create grid of test points for density evaluation."""
    # Compute bounds based on covariance
    std_devs = jnp.sqrt(jnp.diag(cov))
    margin = 3.0  # Number of standard deviations for bounds

    x_min = mean[0] - margin * std_devs[0]
    x_max = mean[0] + margin * std_devs[0]
    y_min = mean[1] - margin * std_devs[1]
    y_max = mean[1] + margin * std_devs[1]

    # Create meshgrid
    x = jnp.linspace(x_min, x_max, n_points)
    y = jnp.linspace(y_min, y_max, n_points)
    xs, ys = jnp.meshgrid(x, y)

    return xs, ys


def sample_gaussian(
    key: Array, man: Gaussian, p: Point[Natural, Gaussian], n: int
) -> Array:
    """Sample from a Gaussian distribution."""
    return man.sample(key, p, n)


sample_gaussian = jax.jit(sample_gaussian, static_argnames=["man", "n"])


def compute_densities(
    man: Gaussian, natural_point: Point[Natural, Gaussian], xs: Array, ys: Array
) -> Array:
    """Compute densities for a Gaussian distribution."""

    points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    zs = jax.vmap(man.density, in_axes=(None, 0))(natural_point, points)

    return zs.reshape(xs.shape)


compute_densities = jax.jit(compute_densities, static_argnames=["man"])


def scipy_densities(mean: Array, cov: Array, xs: Array, ys: Array) -> Array:
    points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    return jax.vmap(multivariate_normal.pdf, in_axes=(0, None, None))(points, mean, cov)


scipy_densities = jax.jit(scipy_densities)


def compute_gaussian_results(
    key: Array,
    mean: Array,
    covariance: Array,
    sample_size: int,
) -> BivariateResults:
    """Run bivariate Gaussian computations with different covariance structures."""
    # Your existing computation code here, returning raw arrays
    pd_man = Gaussian(data_dim=2, covariance_shape=PositiveDefinite)

    gt_cov = PositiveDefinite.from_matrix(covariance)
    gt_mean_point = pd_man.from_mean_and_covariance(mean, gt_cov)
    print("Mean point:", gt_mean_point.params)

    gt_natural_point = pd_man.to_natural(gt_mean_point)
    print("Natural point:", gt_natural_point.params)

    # Debug the conversion back
    print("\nConverting back:")
    mean_point = pd_man.to_mean(gt_natural_point)
    print("Mean point params:", mean_point.params)

    # Look at intermediate steps in to_mean_and_covariance
    mean, cov = pd_man.to_mean_and_covariance(mean_point)
    print("Extracted mean:", mean)
    print("Covariance params:", cov.params)
    print("Covariance matrix:", cov.matrix)

    dia_man = Gaussian(data_dim=2, covariance_shape=Diagonal)
    scl_man = Gaussian(data_dim=2, covariance_shape=Scale)

    mans = [pd_man, dia_man, scl_man]

    gt_cov = PositiveDefinite.from_matrix(covariance)
    gt_mean_point = pd_man.from_mean_and_covariance(mean, gt_cov)
    gt_natural_point = pd_man.to_natural(gt_mean_point)

    xs, ys = create_test_grid(mean, covariance)
    gt_dens = compute_densities(pd_man, gt_natural_point, xs, ys)
    gt_dens = gt_dens.reshape(xs.shape).tolist()
    sp_dens = scipy_densities(mean, covariance, xs, ys)
    sp_dens = sp_dens.reshape(xs.shape).tolist()

    sample = sample_gaussian(key, pd_man, gt_natural_point, sample_size)

    model_results: list[list[list[float]]] = []

    for man in mans:
        mean_point = man.average_sufficient_statistic(sample)
        natural_point = man.to_natural(mean_point)

        densities = compute_densities(man, natural_point, xs, ys)
        zs = densities.reshape(xs.shape)

        model_results.append(
            zs.tolist(),
        )

    [pd_dens, dia_dens, scl_dens] = model_results

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


def main():
    """Run bivariate Gaussian tests."""
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
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
