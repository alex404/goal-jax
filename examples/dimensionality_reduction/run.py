"""Comparing Factor Analysis and PCA for dimensionality reduction."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.models import FactorAnalysis, NormalAnalyticLGM, PrincipalComponentAnalysis

from ..shared import create_grid, example_paths, get_plot_bounds, initialize_jax
from .types import LGMResults


def create_ground_truth(key: Array, sample_size: int) -> tuple[FactorAnalysis, Array, Array]:
    """Generate ground truth FA model and samples."""
    fa = FactorAnalysis(2, 1)

    with fa.obs_man as om:
        obs_mean = jnp.asarray([-0.5, 1.5])
        obs_cov = jnp.asarray([2, 0.5])
        obs_means = om.join_mean_covariance(obs_mean, obs_cov)
        obs_params = om.to_natural(obs_means)

    prr_params = fa.lat_man.to_natural(fa.lat_man.standard_normal())
    int_params = jnp.asarray([1.0, 0.5])
    lkl_params = fa.lkl_fun_man.join_coords(obs_params, int_params)
    params = fa.join_conjugated(lkl_params, prr_params)
    sample = fa.observable_sample(key, params, sample_size)

    return fa, params, sample


def fit_model(
    key: Array, model: NormalAnalyticLGM, sample: Array, n_steps: int
) -> tuple[Array, Array, Array]:
    """Fit model via EM, return (lls, init_params, final_params)."""
    init_params = model.initialize(key)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = model.average_log_observable_density(params, sample)
        return model.expectation_maximization(params, sample), ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_model = jax.jit(fit_model, static_argnames=["model", "n_steps"])


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)

    # Generate data
    key_data, key_train = jax.random.split(key)
    gt_model, gt_params, sample = create_ground_truth(key_data, 500)

    # Models
    fa = FactorAnalysis(2, 1)
    pca = PrincipalComponentAnalysis(2, 1)

    # Train
    key_fa, key_pca = jax.random.split(key_train)
    fa_lls, fa_init, fa_final = fit_model(key_fa, fa, sample, 50)
    pca_lls, pca_init, pca_final = fit_model(key_pca, pca, sample, 50)

    # Grid
    bounds = get_plot_bounds(sample)
    obs_grid = jnp.stack([arr.ravel() for arr in create_grid(bounds)], axis=1)
    lat_grid = jnp.linspace(bounds[0], bounds[1], 50)

    def grid_density(model: NormalAnalyticLGM, params: Array) -> Array:
        return jax.vmap(model.observable_density, in_axes=(None, 0))(params, obs_grid)

    results = LGMResults(
        sample=sample.tolist(),
        plot_bounds=bounds,
        training_lls={"factor_analysis": fa_lls.tolist(), "pca": pca_lls.tolist()},
        grid_points={"observable": obs_grid.tolist(), "latent": lat_grid.tolist()},
        ground_truth_densities=grid_density(gt_model, gt_params).tolist(),
        initial_densities={
            "factor_analysis": grid_density(fa, fa_init).tolist(),
            "pca": grid_density(pca, pca_init).tolist(),
        },
        learned_densities={
            "factor_analysis": grid_density(fa, fa_final).tolist(),
            "pca": grid_density(pca, pca_final).tolist(),
        },
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
