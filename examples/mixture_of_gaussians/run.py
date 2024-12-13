"""Test script for mixture models with different covariance structures."""

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.mixture import GaussianMixture

from goal.geometry import Diagonal, Mean, Natural, Point, PositiveDefinite, Scale
from goal.models import (
    Categorical,
    Euclidean,
    FullNormal,
    Mixture,
    Normal,
)

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
        mean_point: Point[Mean, Euclidean] = Point(mean)
        mean_params = normal_man.join_mean_covariance(mean_point, cov_mat)
        components.append(mean_params)

    weights: Point[Mean, Categorical] = Point(jnp.array([0.35, 0.25]))

    mean_mix: Point[Mean, Mixture[FullNormal]] = mix_man.join_mean_mixture(
        components, weights
    )
    return mix_man.to_natural(mean_mix)


# def initialize_mixture_parameters[R: PositiveDefinite](
#     key: Array,
#     mix_man: Mixture[Normal[R]],
#     n_components: int,
#     mean_scale: float = 1.0,
#     cov_scale: float = 0.1,
# ) -> Point[Natural, Mixture[Normal[R]]]:
#     # Split keys for independent initialization
#     keys = jax.random.split(key, 3)
#
#     # Initialize component means and covariances
#     components = []
#     for i in range(n_components):
#         # Get fresh key for each component
#         key_i = jax.random.fold_in(keys[0], i)
#         keys_i = jax.random.split(key_i, 2)
#
#         # Initialize mean - spread out within data scale
#         mean: Point[Mean, Euclidean] = Point(
#             mean_scale * jax.random.normal(keys_i[0], shape=(mix_man.obs_man.data_dim,))
#         )
#
#         # Initialize covariance - start near identity
#         base_cov = jnp.eye(mix_man.obs_man.data_dim)
#         noise = cov_scale * jax.random.normal(keys_i[1], shape=base_cov.shape)
#         cov = base_cov + noise @ noise.T  # Ensure positive definite
#
#         # Convert to natural parameters
#         cov_mat = mix_man.obs_man.cov_man.from_dense(cov)
#         mean_params = mix_man.obs_man.join_mean_covariance(mean, cov_mat)
#         comp = mix_man.obs_man.to_natural(mean_params)
#         components.append(comp)
#
#     # Initialize weights close to uniform
#     prior: Point[Natural, Categorical] = mix_man.lat_man.shape_initialize(
#         keys[1], shp=0.1
#     )
#
#     # Join into mixture parameters
#     return mix_man.join_natural_mixture(components, prior)


def goal_to_sklearn_mixture(
    mix_man: Mixture[FullNormal],
    goal_mixture: Point[Natural, Mixture[FullNormal]],
) -> GaussianMixture:
    # Convert to mean coordinates first
    mean_mixture = mix_man.to_mean(goal_mixture)

    # Split into components and weights
    components, weights = mix_man.split_mean_mixture(mean_mixture)

    # Get component means and covariances
    means = []
    covariances: list[Array] = []
    for comp in components:
        mean, cov = mix_man.obs_man.split_mean_covariance(comp)
        means.append(mean.params)
        covariances.append(mix_man.obs_man.cov_man.to_dense(cov))

    # Get mixing weights
    mixing_weights = mix_man.lat_man.to_probs(weights)

    # Create and configure sklearn model
    n_components = len(components)

    sklearn_mixture = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        weights_init=mixing_weights,
        means_init=np.array(means),
        precisions_init=np.array([np.linalg.inv(cov) for cov in covariances]),
    )

    # Fake the fitting by setting parameters directly
    sklearn_mixture.weights_ = mixing_weights
    sklearn_mixture.means_ = np.array(means)
    sklearn_mixture.covariances_ = np.array(covariances)
    sklearn_mixture.precisions_cholesky_ = np.array(
        [np.linalg.cholesky(np.linalg.inv(cov)) for cov in covariances]
    )

    return sklearn_mixture


### Fitting ###


def fit_mixture[R: PositiveDefinite](
    key: Array,
    mix_man: Mixture[Normal[R]],
    n_steps: int,
    sample: Array,
) -> tuple[Array, Point[Natural, Mixture[Normal[R]]]]:
    params = mix_man.shape_initialize(key)

    def em_step(
        carry: Point[Natural, Mixture[Normal[R]]], _: Any
    ) -> tuple[Point[Natural, Mixture[Normal[R]]], Array]:
        params = carry
        ll = mix_man.average_log_observable_density(params, sample)
        next_params = mix_man.expectation_maximization(params, sample)
        return next_params, ll

    final_params, lls = jax.lax.scan(em_step, params, None, length=n_steps)
    return lls.ravel(), final_params


fit_mixture = jax.jit(fit_mixture, static_argnames=["mix_man", "n_steps"])

### Analysis ###


def compute_mixture_results(
    key: Array,
    sample_size: int,
    n_steps: int,
    bounds: tuple[float, float, float, float],
) -> MixtureResults:
    # Create manifolds
    pd_man = Normal(2, PositiveDefinite)
    dia_man = Normal(2, Diagonal)
    iso_man = Normal(2, Scale)

    pd_mix_man = Mixture(pd_man, 3)
    dia_mix_man = Mixture(dia_man, 3)
    iso_mix_man = Mixture(iso_man, 3)

    # Generate ground truth model
    gt_params = create_ground_truth_parameters(pd_mix_man)
    sklearn_mixture = goal_to_sklearn_mixture(pd_mix_man, gt_params)

    # Generate samples
    key_sample, key_train = jax.random.split(key)
    sample = pd_mix_man.observable_sample(key_sample, gt_params, sample_size)

    # Create evaluation grid
    xs, ys = create_test_grid(bounds)
    grid_points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    # Split training key
    keys_train = jax.random.split(key_train, 3)

    # Train all models
    pd_lls, pd_params = fit_mixture(keys_train[0], pd_mix_man, n_steps, sample)
    dia_lls, dia_params = fit_mixture(keys_train[1], dia_mix_man, n_steps, sample)
    iso_lls, iso_params = fit_mixture(keys_train[2], iso_mix_man, n_steps, sample)

    lls = {
        "positive_definite": pd_lls.tolist(),
        "diagonal": dia_lls.tolist(),
        "isotropic": iso_lls.tolist(),
    }

    def compute_grid_density[R: PositiveDefinite](
        model: Mixture[Normal[R]], params: Point[Natural, Mixture[Normal[R]]]
    ) -> Array:
        return jax.vmap(model.observable_density, in_axes=(None, 0))(
            params, grid_points
        ).reshape(xs.shape)

    gt_dens = compute_grid_density(pd_mix_man, gt_params)
    pd_dens = compute_grid_density(pd_mix_man, pd_params)
    dia_dens = compute_grid_density(dia_mix_man, dia_params)
    iso_dens = compute_grid_density(iso_mix_man, iso_params)

    # Compute ground truth log likelihood
    gt_ll = jnp.mean(
        jax.vmap(pd_mix_man.log_observable_density, in_axes=(None, 0))(
            gt_params, sample
        )
    )

    # Compute scipy density
    sklearn_dens = np.exp(sklearn_mixture.score_samples(grid_points)).reshape(xs.shape)

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
        scipy_densities=sklearn_dens.tolist(),
        ground_truth_densities=gt_dens.tolist(),
        positive_definite_densities=pd_dens.tolist(),
        diagonal_densities=dia_dens.tolist(),
        isotropic_densities=iso_dens.tolist(),
        ground_truth_ll=float(gt_ll),
        training_lls=lls,
    )


### Main ###


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_disable_jit", False)


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
