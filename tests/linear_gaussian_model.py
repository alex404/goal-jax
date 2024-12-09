"""Tests for LinearGaussianModel conversions to Normal distributions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import LinearGaussianModel, Normal

jax.config.update("jax_platform_name", "cpu")

rtol = 1e-3
atol = 1e-5


def test_normal_conversion():
    obs_dim = 3
    lat_dim = 2

    # Normal Models
    nor_man = Normal(obs_dim + lat_dim, PositiveDefinite)

    # LGM Models
    pod_lgm_man = LinearGaussianModel(obs_dim, PositiveDefinite, lat_dim)
    dia_lgm_man = LinearGaussianModel(obs_dim, Diagonal, lat_dim)
    iso_lgm_man = LinearGaussianModel(obs_dim, Scale, lat_dim)

    # GT Model
    gt_nor_mean = nor_man.loc_man.mean_point(jnp.asarray([2, -1, 0, 3, -2]))
    gt_nor_cov = nor_man.cov_man.from_dense(
        jnp.asarray(
            [
                [4.0, 1.5, 0.0, -1.0, 0.5],
                [1.5, 3.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.5, 0.0],
                [-1.0, 0.0, 0.5, 3.0, 1.0],
                [0.5, 0.0, 0.0, 1.0, 2.0],
            ]
        )
    )
    gt_nor_means = nor_man.from_mean_and_covariance(gt_nor_mean, gt_nor_cov)
    key = jax.random.PRNGKey(0)
    sample = nor_man.sample(key, nor_man.to_natural(gt_nor_means), 1000)

    def test_models[Rep: PositiveDefinite](lgm_man: LinearGaussianModel[Rep]) -> None:
        # Gather parameterizations
        lgm_means = lgm_man.average_sufficient_statistic(sample)
        lgm_params = lgm_man.to_natural(lgm_means)
        lgm_remeans = lgm_man.to_mean(lgm_params)

        nor_means = lgm_man.to_normal(lgm_means)
        nor_params = nor_man.to_natural(nor_means)
        nor_remeans = nor_man.to_mean(nor_params)

        # Density comparison lgm vs normal
        assert jnp.allclose(
            lgm_man.average_log_density(lgm_params, sample),
            nor_man.average_log_density(nor_params, sample),
            rtol=rtol,
            atol=atol,
        )

        # Recovering mean parameters back and forth through natural space
        assert jnp.allclose(lgm_means.params, lgm_remeans.params, rtol=rtol, atol=atol)

        # Comparing backward transition of lgm vs normal
        assert jnp.allclose(
            lgm_man.to_normal(lgm_remeans).params,
            nor_remeans.params,
            rtol=rtol,
            atol=atol,
        )

    test_models(pod_lgm_man)
    test_models(dia_lgm_man)
    test_models(iso_lgm_man)
