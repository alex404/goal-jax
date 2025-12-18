"""Test script for mixture models with different covariance structures."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.mixture import GaussianMixture  # pyright: ignore[reportMissingTypeStubs]

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import (
    AnalyticMixture,
    Normal,
)

from ..shared import create_grid, example_paths, initialize_jax
from .types import MixtureResults

### Constructors ###


def create_ground_truth_parameters(
    mix_man: AnalyticMixture[Normal],
) -> Array:
    """Create ground truth parameters for a mixture of 3 bivariate Gaussians.

    Returns:
        likelihood_params: Parameters of likelihood distribution p(x|z)
        prior_params: Parameters of categorical prior p(z)
    """
    means = [
        jnp.array([1.0, 1.0]),
        jnp.array([-2.0, -1.0]),
        jnp.array([0.0, 0.5]),
    ]
    covs = [
        jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        jnp.array([[2.0, -0.8], [-0.8, 0.5]]),
        jnp.array([[0.5, 0.0], [0.0, 0.5]]),
    ]

    # Create component parameters
    component_list: list[Array] = []
    for mean, cov in zip(means, covs):
        with mix_man.obs_man as nm:
            cov_mat = nm.cov_man.from_matrix(cov)
            mean_params = nm.join_mean_covariance(mean, cov_mat)
            component_list.append(mean_params)

    # Concatenate components into flat 1D array [n_categories * obs_dim]
    components = jnp.concatenate(component_list)

    weights = jnp.array([0.35, 0.25])

    mean_mix = mix_man.join_mean_mixture(components, weights)
    return mix_man.to_natural(mean_mix)


def goal_to_sklearn_mixture(
    mix_man: AnalyticMixture[Normal],
    goal_mixture: Array,
) -> GaussianMixture:
    # Convert to mean coordinates first
    mean_mixture = mix_man.to_mean(goal_mixture)

    # Split into components and weights (components is flat 1D)
    components, weights = mix_man.split_mean_mixture(mean_mixture)

    # Get component means and covariances
    means = []
    covariances: list[Array] = []
    for i in range(mix_man.cmp_man.n_reps):
        comp = mix_man.cmp_man.get_replicate(components, i)
        mean, cov = mix_man.obs_man.split_mean_covariance(comp)
        means.append(mean)
        covariances.append(mix_man.obs_man.cov_man.to_matrix(cov))

    # Get mixing weights
    mixing_weights = mix_man.lat_man.to_probs(weights)

    # Create and configure sklearn model
    n_components = mix_man.cmp_man.n_reps

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


def fit_model(
    key: Array,
    mix_man: AnalyticMixture[Normal],
    n_steps: int,
    sample: Array,
) -> tuple[
    Array,
    Array,
    Array,
]:
    init_params = mix_man.initialize(key)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = mix_man.average_log_observable_density(params, sample)
        next_params = mix_man.expectation_maximization(params, sample)
        return next_params, ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_model = jax.jit(fit_model, static_argnames=["mix_man", "n_steps"])

### Analysis ###


def compute_mixture_results(
    key: Array,
    sample_size: int,
    n_steps: int,
    bounds: tuple[float, float, float, float],
) -> MixtureResults:
    # Create manifolds
    pd_man = Normal(2, PositiveDefinite())
    dia_man = Normal(2, Diagonal())
    iso_man = Normal(2, Scale())

    pd_mix_man = AnalyticMixture(pd_man, 3)
    dia_mix_man = AnalyticMixture(dia_man, 3)
    iso_mix_man = AnalyticMixture(iso_man, 3)

    # Generate ground truth model
    gt_params = create_ground_truth_parameters(pd_mix_man)
    sklearn_mixture = goal_to_sklearn_mixture(pd_mix_man, gt_params)

    # Generate samples
    key_sample, key_train = jax.random.split(key)
    sample = pd_mix_man.observable_sample(key_sample, gt_params, sample_size)

    # Create evaluation grid
    xs, ys = create_grid(bounds)
    grid_points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    # Split training key
    keys_train = jax.random.split(key_train, 3)

    # Train all models
    pd_lls, pod_params0, pod_params1 = fit_model(
        keys_train[0], pd_mix_man, n_steps, sample
    )
    dia_lls, dia_params0, dia_params1 = fit_model(
        keys_train[1], dia_mix_man, n_steps, sample
    )
    iso_lls, iso_params0, iso_params1 = fit_model(
        keys_train[2], iso_mix_man, n_steps, sample
    )

    lls = {
        "positive_definite": pd_lls.tolist(),
        "diagonal": dia_lls.tolist(),
        "isotropic": iso_lls.tolist(),
    }

    def compute_grid_density(
        model: AnalyticMixture[Normal],
        params: Array,
    ) -> Array:
        return jax.vmap(model.observable_density, in_axes=(None, 0))(
            params, grid_points
        ).reshape(xs.shape)

    gt_dens = compute_grid_density(pd_mix_man, gt_params)
    pod_dens0 = compute_grid_density(pd_mix_man, pod_params0)
    dia_dens0 = compute_grid_density(dia_mix_man, dia_params0)
    iso_dens0 = compute_grid_density(iso_mix_man, iso_params0)
    pod_dens1 = compute_grid_density(pd_mix_man, pod_params1)
    dia_dens1 = compute_grid_density(dia_mix_man, dia_params1)
    iso_dens1 = compute_grid_density(iso_mix_man, iso_params1)

    # Compute ground truth log likelihood
    gt_ll = jnp.mean(
        jax.vmap(pd_mix_man.log_observable_density, in_axes=(None, 0))(
            gt_params, sample
        )
    )

    # Compute scipy density
    sklearn_dens = np.exp(sklearn_mixture.score_samples(grid_points)).reshape(xs.shape)

    return MixtureResults(
        sample=sample.tolist(),
        plot_xs=xs.tolist(),
        plot_ys=ys.tolist(),
        scipy_densities=sklearn_dens.tolist(),  # type: ignore
        ground_truth_densities=gt_dens.tolist(),
        init_positive_definite_densities=pod_dens0.tolist(),
        init_diagonal_densities=dia_dens0.tolist(),
        init_isotropic_densities=iso_dens0.tolist(),
        positive_definite_densities=pod_dens1.tolist(),
        diagonal_densities=dia_dens1.tolist(),
        isotropic_densities=iso_dens1.tolist(),
        ground_truth_ll=float(gt_ll),
        training_lls=lls,
    )


### Main ###


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_disable_jit", False)


def main():
    """Run mixture model tests."""
    initialize_jax()
    paths = example_paths(__file__)

    key = jax.random.PRNGKey(0)

    # Set bounds based on expected component locations
    bounds = (-4.0, 4.0, -4.0, 4.0)

    # Generate sample and fit models
    results = compute_mixture_results(
        key,
        sample_size=500,
        n_steps=50,
        bounds=bounds,
    )

    paths.save_analysis(results)


if __name__ == "__main__":
    main()
