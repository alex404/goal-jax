"""Mixture of Gaussians with different covariance structures."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from sklearn.mixture import GaussianMixture

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import AnalyticMixture, Normal

from ..shared import create_grid, example_paths, initialize_jax
from .types import MixtureResults


def create_ground_truth(mix_man: AnalyticMixture[Normal]) -> Array:
    """Create ground truth mixture parameters."""
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

    components = []
    for mean, cov in zip(means, covs):
        cov_params = mix_man.obs_man.cov_man.from_matrix(cov)
        means = mix_man.obs_man.join_mean_covariance(mean, cov_params)
        components.append(means)

    weights = jnp.array([0.35, 0.25])
    mean_mix = mix_man.join_mean_mixture(jnp.concatenate(components), weights)
    return mix_man.to_natural(mean_mix)


def to_sklearn(mix_man: AnalyticMixture[Normal], params: Array) -> GaussianMixture:
    """Convert GOAL mixture to sklearn GaussianMixture."""
    means = mix_man.to_mean(params)
    components, weights = mix_man.split_mean_mixture(means)

    means, covs = [], []
    for i in range(mix_man.cmp_man.n_reps):
        comp = mix_man.cmp_man.get_replicate(components, i)
        mean, cov = mix_man.obs_man.split_mean_covariance(comp)
        means.append(mean)
        covs.append(mix_man.obs_man.cov_man.to_matrix(cov))

    mixing = mix_man.lat_man.to_probs(weights)

    gmm = GaussianMixture(
        n_components=mix_man.cmp_man.n_reps,
        covariance_type="full",
        weights_init=mixing,
        means_init=np.array(means),
        precisions_init=np.array([np.linalg.inv(c) for c in covs]),
    )
    gmm.weights_ = mixing
    gmm.means_ = np.array(means)
    gmm.covariances_ = np.array(covs)
    gmm.precisions_cholesky_ = np.array(
        [np.linalg.cholesky(np.linalg.inv(c)) for c in covs]
    )
    return gmm


def fit_mixture(
    key: Array, mix_man: AnalyticMixture[Normal], sample: Array, n_steps: int
) -> tuple[Array, Array, Array]:
    """Fit mixture via EM, return (lls, init_params, final_params)."""
    init_params = mix_man.initialize(key)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = mix_man.average_log_observable_density(params, sample)
        return mix_man.expectation_maximization(params, sample), ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_mixture = jax.jit(fit_mixture, static_argnames=["mix_man", "n_steps"])


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)

    # Models
    pd_man = Normal(2, PositiveDefinite())
    dia_man = Normal(2, Diagonal())
    iso_man = Normal(2, Scale())

    pd_mix = AnalyticMixture(pd_man, 3)
    dia_mix = AnalyticMixture(dia_man, 3)
    iso_mix = AnalyticMixture(iso_man, 3)

    # Ground truth and sample
    gt_params = create_ground_truth(pd_mix)
    sklearn_gmm = to_sklearn(pd_mix, gt_params)

    key_sample, key_train = jax.random.split(key)
    sample = pd_mix.observable_sample(key_sample, gt_params, 500)

    # Grid
    bounds = (-4.0, 4.0, -4.0, 4.0)
    xs, ys = create_grid(bounds)
    grid_points = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    # Train
    keys = jax.random.split(key_train, 3)
    pd_lls, pd_init, pd_final = fit_mixture(keys[0], pd_mix, sample, 50)
    dia_lls, dia_init, dia_final = fit_mixture(keys[1], dia_mix, sample, 50)
    iso_lls, iso_init, iso_final = fit_mixture(keys[2], iso_mix, sample, 50)

    def grid_density(model: AnalyticMixture[Normal], params: Array) -> Array:
        return jax.vmap(model.observable_density, in_axes=(None, 0))(
            params, grid_points
        ).reshape(xs.shape)

    gt_ll = float(
        jnp.mean(
            jax.vmap(pd_mix.log_observable_density, in_axes=(None, 0))(
                gt_params, sample
            )
        )
    )
    sklearn_dens = np.exp(sklearn_gmm.score_samples(grid_points)).reshape(xs.shape)

    results = MixtureResults(
        sample=sample.tolist(),
        plot_xs=xs.tolist(),
        plot_ys=ys.tolist(),
        scipy_densities=sklearn_dens.tolist(),  # type: ignore
        ground_truth_densities=grid_density(pd_mix, gt_params).tolist(),
        init_positive_definite_densities=grid_density(pd_mix, pd_init).tolist(),
        init_diagonal_densities=grid_density(dia_mix, dia_init).tolist(),
        init_isotropic_densities=grid_density(iso_mix, iso_init).tolist(),
        positive_definite_densities=grid_density(pd_mix, pd_final).tolist(),
        diagonal_densities=grid_density(dia_mix, dia_final).tolist(),
        isotropic_densities=grid_density(iso_mix, iso_final).tolist(),
        ground_truth_ll=gt_ll,
        training_lls={
            "positive_definite": pd_lls.tolist(),
            "diagonal": dia_lls.tolist(),
            "isotropic": iso_lls.tolist(),
        },
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
