"""Test script for mixture models with different covariance structures."""

import json

import jax
import jax.numpy as jnp
from jax import Array

from goal.exponential_family import Mean, Natural
from goal.exponential_family.distributions import (
    Categorical,
    FullNormal,
    Normal,
    diagonal_normal_manifold,
    full_normal_manifold,
    isotropic_normal_manifold,
)
from goal.harmonium.models import Mixture
from goal.manifold import Euclidean, Point, euclidean_point
from goal.transforms import PositiveDefinite

from .common import MixtureResults, analysis_path

### Constructors ###


def create_test_grid(
    bounds: tuple[float, float, float, float], n_points: int = 50
) -> tuple[Array, Array]:
    """Create a grid for density evaluation."""
    x_min, x_max, y_min, y_max = bounds
    x = jnp.linspace(x_min, x_max, n_points)
    y = jnp.linspace(y_min, y_max, n_points)
    xs, ys = jnp.meshgrid(x, y)
    return xs, ys


def create_ground_truth_parameters(
    mix_man: Mixture[FullNormal],
) -> Point[Natural, Mixture[FullNormal]]:
    """Create ground truth parameters for a mixture of 3 bivariate Gaussians.

    Returns:
        likelihood_params: Parameters of likelihood distribution p(x|z)
        prior_params: Parameters of categorical prior p(z)
    """
    normal_man = mix_man.obs_man
    # Two well-separated components and one overlapping component
    means = [
        jnp.array([1.0, 1.0]),  # First component
        jnp.array([-2.0, -1.0]),  # Second component
        jnp.array([0.0, 0.5]),  # Third component (overlapping)
    ]

    # Different covariance structures
    covs = [
        jnp.array([[1.0, 0.5], [0.5, 1.0]]),  # Standard covariance
        jnp.array([[2.0, -0.8], [-0.8, 0.5]]),  # More elongated
        jnp.array([[0.5, 0.0], [0.0, 0.5]]),  # Isotropic
    ]

    # Create component parameters
    components: list[Point[Mean, Normal[PositiveDefinite]]] = []
    for mean, cov in zip(means, covs):
        cov_mat = normal_man.cov_man.from_dense(cov)
        mean_point = euclidean_point(mean)
        mean_params = normal_man.from_mean_and_covariance(mean_point, cov_mat)
        components.append(mean_params)

    weights: Point[Mean, Categorical] = Point(jnp.array([0.35, 0.25]))

    mean_mix: Point[Mean, Mixture[FullNormal]] = mix_man.join_mean_mixture(
        components, weights
    )
    return mix_man.to_natural(mean_mix)


def initialize_mixture_parameters[R: PositiveDefinite](
    key: Array,
    mix_man: Mixture[Normal[R]],
    n_components: int,
    mean_scale: float = 1.0,
    cov_scale: float = 0.1,
) -> Point[Natural, Mixture[Normal[R]]]:
    # Split keys for independent initialization
    keys = jax.random.split(key, 3)

    # Initialize component means and covariances
    components = []
    for i in range(n_components):
        # Get fresh key for each component
        key_i = jax.random.fold_in(keys[0], i)
        keys_i = jax.random.split(key_i, 2)

        # Initialize mean - spread out within data scale
        mean: Point[Mean, Euclidean] = Point(
            mean_scale * jax.random.normal(keys_i[0], shape=(mix_man.obs_man.data_dim,))
        )

        # Initialize covariance - start near identity
        base_cov = jnp.eye(mix_man.obs_man.data_dim)
        noise = cov_scale * jax.random.normal(keys_i[1], shape=base_cov.shape)
        cov = base_cov + noise @ noise.T  # Ensure positive definite

        # Convert to natural parameters
        cov_mat = mix_man.obs_man.cov_man.from_dense(cov)
        mean_params = mix_man.obs_man.from_mean_and_covariance(mean, cov_mat)
        comp = mix_man.obs_man.to_natural(mean_params)
        components.append(comp)

    # Initialize weights close to uniform
    prior: Point[Natural, Categorical] = mix_man.lat_man.normal_initialize(
        keys[1], sigma=0.1
    )

    # Join into mixture parameters
    return mix_man.join_natural_mixture(components, prior)


### Fitting ###


def fit_mixture[R: PositiveDefinite](
    key: Array,
    mix_man: Mixture[Normal[R]],
    sample: Array,
    n_steps: int,
) -> tuple[list[float], Point[Natural, Mixture[Normal[R]]]]:
    params = initialize_mixture_parameters(key, mix_man, 3)

    # Run EM iterations
    log_lls: list[float] = []
    for _ in range(n_steps):
        # Compute current log likelihood
        log_ll = jnp.mean(
            jax.vmap(mix_man.log_observable_density, in_axes=(None, 0))(params, sample)
        )
        log_lls.append(float(log_ll))

        # Update parameters
        params = mix_man.expectation_maximization(params, sample)

    return log_lls, params


### Analysis ###


def compute_mixture_results(
    key: Array,
    sample_size: int,
    n_steps: int,
    bounds: tuple[float, float, float, float],
) -> MixtureResults:
    # Create manifolds
    pd_man = full_normal_manifold(2)
    dia_man = diagonal_normal_manifold(2)
    iso_man = isotropic_normal_manifold(2)

    pd_mix_man = Mixture(pd_man, 3)
    dia_mix_man = Mixture(dia_man, 3)
    iso_mix_man = Mixture(iso_man, 3)

    # Generate ground truth model
    gt_params = create_ground_truth_parameters(pd_mix_man)

    # Generate samples
    key_sample, key_train = jax.random.split(key)
    sample = pd_mix_man.observable_sample(key_sample, gt_params, sample_size)

    # Create evaluation grid
    xs, ys = create_test_grid(bounds)
    grid_points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    # Split training key
    keys_train = jax.random.split(key_train, 3)

    # Train all models
    pd_lls, pd_params = fit_mixture(keys_train[0], pd_mix_man, sample, n_steps)
    diag_lls, dia_params = fit_mixture(keys_train[1], dia_mix_man, sample, n_steps)
    iso_lls, iso_params = fit_mixture(keys_train[2], iso_mix_man, sample, n_steps)

    def compute_grid_density[R: PositiveDefinite](
        model: Mixture[Normal[R]], params: Point[Natural, Mixture[Normal[R]]]
    ) -> Array:
        return jax.vmap(model.observable_density, in_axes=(None, 0))(
            params, grid_points
        ).reshape(xs.shape)

    gt_dens = compute_grid_density(pd_mix_man, gt_params)
    pd_dens = compute_grid_density(pd_mix_man, pd_params)
    diag_dens = compute_grid_density(dia_mix_man, dia_params)
    iso_dens = compute_grid_density(iso_mix_man, iso_params)

    # Compute ground truth log likelihood
    gt_ll = jnp.mean(
        jax.vmap(pd_mix_man.log_observable_density, in_axes=(None, 0))(
            gt_params, sample
        )
    )

    return MixtureResults(
        sample=sample.tolist(),
        plot_xs=xs.tolist(),
        plot_ys=ys.tolist(),
        ground_truth_densities=gt_dens.tolist(),
        positive_definite_densities=pd_dens.tolist(),
        diagonal_densities=diag_dens.tolist(),
        isotropic_densities=iso_dens.tolist(),
        ground_truth_ll=float(gt_ll),
        positive_definite_lls=pd_lls,
        diagonal_lls=diag_lls,
        isotropic_lls=iso_lls,
    )


### Main ###


def main():
    """Run mixture model tests."""
    key = jax.random.PRNGKey(0)

    # Set bounds based on expected component locations
    bounds = (-4.0, 4.0, -4.0, 4.0)

    # Generate sample and fit models
    results = compute_mixture_results(
        key,
        sample_size=500,
        n_steps=100,
        bounds=bounds,
    )

    # Save results
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
