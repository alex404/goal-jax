"""Hierarchical Mixture of Gaussians (HMoG) example."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Diagonal, Scale
from goal.models import AnalyticHMoG, analytic_hmog

from ..shared import example_paths, initialize_jax
from .types import HMoGResults

# Model configuration
width = 0.1
height = 8.0
sep = 3.0


def create_ground_truth() -> tuple[AnalyticHMoG, Array]:
    """Create ground truth HMoG model."""
    hmog = analytic_hmog(obs_dim=2, obs_rep=Diagonal(), lat_dim=1, n_components=2)

    with hmog.pst_man.prr_man as cm:
        cat_natural = jnp.array([0.5])
        cat_mean = cm.to_mean(cat_natural)

    with hmog.pst_man as um, um.obs_man as lm:
        y0_means = lm.join_mean_covariance(jnp.array([-sep / 2]), jnp.array([1.0]))
        y1_means = lm.join_mean_covariance(jnp.array([sep / 2]), jnp.array([1.0]))
        components = jnp.concatenate([y0_means, y1_means])
        mix_mean = um.join_mean_mixture(components, cat_mean)
        mix_natural = um.to_natural(mix_mean)

    with hmog.obs_man as om:
        obs_mean = om.join_mean_covariance(jnp.array([0.0, 0.0]), jnp.array([width, height]))
        obs_natural = om.to_natural(obs_mean)
        int_mat0 = jnp.array([1.0, 0.0])
        obs_prs = om.split_location_precision(obs_natural)[1]
        int_mat = hmog.int_man.from_matrix(om.cov_man.to_matrix(obs_prs) @ int_mat0)

    lkl_params = hmog.lkl_fun_man.join_coords(obs_natural, int_mat)
    return hmog, hmog.join_conjugated(lkl_params, mix_natural)


def fit_hmog(
    key: Array, hmog: AnalyticHMoG, sample: Array, n_steps: int
) -> tuple[Array, Array, Array]:
    """Fit HMoG via EM."""
    init_params = hmog.initialize_from_sample(key, sample, shape=2)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = hmog.average_log_observable_density(params, sample)
        return hmog.expectation_maximization(params, sample), ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_hmog = jax.jit(fit_hmog, static_argnames=["hmog", "n_steps"])


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)
    key_sample, key_train = jax.random.split(key)

    # Ground truth
    gt_hmog, gt_params = create_ground_truth()
    sample = gt_hmog.observable_sample(key_sample, gt_params, 400)

    # Models
    iso_hmog = analytic_hmog(2, Scale(), 1, 2)
    dia_hmog = analytic_hmog(2, Diagonal(), 1, 2)

    # Training
    iso_lls, iso_init, iso_final = fit_hmog(key_train, iso_hmog, sample, 1000)
    dia_lls, dia_init, dia_final = fit_hmog(key_train, dia_hmog, sample, 1000)

    # Grid for densities
    x = jnp.linspace(-10, 10, 100)
    y = jnp.linspace(-10, 10, 100)
    grid_points = jnp.stack([g.ravel() for g in jnp.meshgrid(x, y)], axis=1)
    lat_y = jnp.linspace(-6, 6, 100)

    def obs_density(model: AnalyticHMoG, params: Array) -> Array:
        return jax.vmap(model.observable_density, in_axes=(None, 0))(params, grid_points).reshape(100, 100)

    def mix_density(model: AnalyticHMoG, params: Array) -> Array:
        mix_params = model.split_conjugated(params)[1]
        return jax.vmap(model.pst_man.observable_density, in_axes=(None, 0))(mix_params, lat_y[:, None])

    results = HMoGResults(
        plot_range_x1=x.tolist(),
        plot_range_x2=y.tolist(),
        observations=sample.tolist(),
        log_likelihoods={"Isotropic": iso_lls.tolist(), "Diagonal": dia_lls.tolist()},
        observable_densities={
            "Ground Truth": obs_density(gt_hmog, gt_params).tolist(),
            "Isotropic Initial": obs_density(iso_hmog, iso_init).tolist(),
            "Isotropic Final": obs_density(iso_hmog, iso_final).tolist(),
            "Diagonal Initial": obs_density(dia_hmog, dia_init).tolist(),
            "Diagonal Final": obs_density(dia_hmog, dia_final).tolist(),
        },
        plot_range_y=lat_y.tolist(),
        mixture_densities={
            "Ground Truth": mix_density(gt_hmog, gt_params).tolist(),
            "Isotropic Initial": mix_density(iso_hmog, iso_init).tolist(),
            "Isotropic Final": mix_density(iso_hmog, iso_final).tolist(),
            "Diagonal Initial": mix_density(dia_hmog, dia_init).tolist(),
            "Diagonal Final": mix_density(dia_hmog, dia_final).tolist(),
        },
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
