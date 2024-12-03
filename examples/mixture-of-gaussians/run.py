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


def create_test_grid(
    bounds: tuple[float, float, float, float], n_points: int = 50
) -> tuple[Array, Array]:
    """Create a grid for density evaluation."""
    x_min, x_max, y_min, y_max = bounds
    x = jnp.linspace(x_min, x_max, n_points)
    y = jnp.linspace(y_min, y_max, n_points)
    xs, ys = jnp.meshgrid(x, y)
    return xs, ys


def initialize_mixture[R: PositiveDefinite](
    key: Array,
    mix_man: Mixture[Normal[R]],
    n_components: int,
    mean_scale: float = 1.0,
    cov_scale: float = 0.1,
) -> Point[Natural, Mixture[Normal[R]]]:
    """Initialize mixture model with reasonable parameters.

    Args:
        key: PRNG key
        mix_man: Mixture model manifold
        n_components: Number of components
        mean_scale: Scale for initializing means
        cov_scale: Scale for initializing covariances

    Returns:
        Initial mixture parameters in natural coordinates
    """
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
            mean_scale
            * jax.random.normal(keys_i[0], shape=(mix_man.obs_man.data_dimension,))
        )

        # Initialize covariance - start near identity
        base_cov = jnp.eye(mix_man.obs_man.data_dimension)
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


def fit_mixture[R: PositiveDefinite](
    key: Array,
    mix_man: Mixture[Normal[R]],
    sample: Array,
    n_steps: int,
) -> tuple[list[float], Point[Natural, Mixture[Normal[R]]]]:
    # Initialize parameters using sklearn GMM for good starting point
    # n_components = mix_man.lat_man.dimension + 1  # Categorical dimension + 1
    # sk_gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    # sk_gmm.fit(sample)
    #
    # # Convert sklearn parameters to our format
    # sk_means = sk_gmm.means_
    # sk_covs = sk_gmm.covariances_
    # components = []
    # for mean, cov in zip(sk_means, sk_covs):
    #     cov_mat = mix_man.obs_man.cov_man.from_dense(cov)
    #     mean_point = Point(mean)
    #     mean_params = mix_man.obs_man.from_mean_and_covariance(mean_point, cov_mat)
    #     nat_params = mix_man.obs_man.to_natural(mean_params)
    #     components.append(nat_params)

    # Create initial likelihood parameters
    base_params = components[0]
    interaction_cols = [comp - base_params for comp in components[1:]]
    interaction_mat = mix_man.obs_man.inter_man.from_columns(interaction_cols)
    like_params = mix_man.like_man.join_params(base_params, interaction_mat)

    # Create initial prior parameters (log weights)
    prior_params = mix_man.lat_man.to_natural(Point(gmm.weights_))

    # Create initial model parameters
    params = mix_man.join_conjugated(like_params, prior_params)

    # Run EM iterations
    log_lls = []
    for _ in range(n_steps):
        # Compute current log likelihood
        log_ll = jnp.mean(
            jax.vmap(lambda x: mix_man.log_observable_density(params, x))(sample)
        )
        log_lls.append(float(log_ll))

        # Update parameters
        params = mix_man.expectation_maximization(params, sample)

    return log_lls, params


def compute_mixture_results(
    key: Array,
    sample_size: int,
    n_steps: int,
    bounds: tuple[float, float, float, float],
) -> MixtureResults:
    """Run mixture model analysis and return results."""
    # Create manifolds
    pd_man = full_normal_manifold(2)
    dia_man = diagonal_normal_manifold(2)
    iso_man = isotropic_normal_manifold(2)

    pd_mix_man = Mixture(pd_man, 3)
    dia_mix_man = Mixture(dia_man, 3)
    iso_mix_man = Mixture(iso_man, 3)

    # Generate ground truth model
    pd_mix = create_ground_truth_parameters(pd_mix_man)

    # Generate samples
    key_sample, key_train = jax.random.split(key)
    sample = pd_mix_man.sample(key_sample, pd_mix, sample_size)

    # Create evaluation grid
    xs, ys = create_test_grid(bounds)
    grid_points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    # Split training key
    keys_train = jax.random.split(key_train, 3)

    # Train all models
    pd_lls, pd_model = fit_mixture(keys_train[0], pd_man, sample, n_steps)
    diag_lls, diag_model = fit_mixture(keys_train[1], diag_man, sample, n_steps)
    scale_lls, scale_model = fit_mixture(keys_train[2], scale_man, sample, n_steps)

    # Compute densities on grid
    compute_grid_density = lambda model, points: jnp.array(
        [model.observable_density(points[i]) for i in range(len(points))]
    ).reshape(xs.shape)

    gt_dens = compute_grid_density(gt_model, grid_points)
    pd_dens = compute_grid_density(pd_model, grid_points)
    diag_dens = compute_grid_density(diag_model, grid_points)
    scale_dens = compute_grid_density(scale_model, grid_points)

    # Compute ground truth log likelihood
    gt_ll = jnp.mean(
        jax.vmap(lambda x: pd_man.log_observable_density(gt_model, x))(sample)
    )

    return MixtureResults(
        sample=sample.tolist(),
        plot_xs=xs.tolist(),
        plot_ys=ys.tolist(),
        ground_truth_densities=gt_dens.tolist(),
        positive_definite_densities=pd_dens.tolist(),
        diagonal_densities=diag_dens.tolist(),
        scale_densities=scale_dens.tolist(),
        ground_truth_ll=float(gt_ll),
        positive_definite_lls=pd_lls,
        diagonal_lls=diag_lls,
        scale_lls=scale_lls,
    )


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
