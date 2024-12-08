"""Tests for LinearGaussianModel conversions to Normal distributions."""

from __future__ import annotations

import jax.numpy as jnp

from goal.geometry import Diagonal, PositiveDefinite, Scale
from goal.models import LinearGaussianModel, Normal


def test_normal_conversion():
    obs_dim = 3
    lat_dim = 2

    # LGM Models
    pod_lgm_man = LinearGaussianModel(obs_dim, PositiveDefinite, lat_dim)
    dia_lgm_man = LinearGaussianModel(obs_dim, Diagonal, lat_dim)
    iso_lgm_man = LinearGaussianModel(obs_dim, Scale, lat_dim)

    # Normal Models
    nor_man = Normal(obs_dim + lat_dim, PositiveDefinite)

    """Create test parameters for the models."""
    # Location parameters
    obs_loc = pod_lgm_man.obs_man.loc_man.mean_point(jnp.array([1.0, -0.5, 0.0]))
    lat_loc = pod_lgm_man.lat_man.loc_man.mean_point(jnp.array([0.5, -1.0]))

    # Shape parameters
    cov_array = jnp.array([[4.0, 1.2, 0.5], [1.2, 3.0, -0.8], [0.5, -0.8, 2.0]])

    pod_obs_cov = pod_lgm_man.obs_man.cov_man.from_dense(cov_array)
    dia_obs_cov = dia_lgm_man.obs_man.cov_man.from_dense(cov_array)
    iso_obs_cov = iso_lgm_man.obs_man.cov_man.from_dense(cov_array)
    lat_cov = pod_lgm_man.lat_man.cov_man.from_dense(
        jnp.array([[1.0, 0.3], [0.3, 0.5]])
    )
    int_mat = pod_lgm_man.int_man.from_dense(
        jnp.array([[0.5, -0.3], [0.2, 0.4], [0.4, -0.1]])
    )

    # Component Normals
    lat_nrm = pod_lgm_man.lat_man.from_mean_and_covariance(lat_loc, lat_cov)
    pod_obs_nrm = pod_lgm_man.obs_man.from_mean_and_covariance(obs_loc, pod_obs_cov)
    dia_obs_nrm = dia_lgm_man.obs_man.from_mean_and_covariance(obs_loc, dia_obs_cov)
    iso_obs_nrm = iso_lgm_man.obs_man.from_mean_and_covariance(obs_loc, iso_obs_cov)

    # LGM Parmeters
    pod_lgm_params = pod_lgm_man.join_params(pod_obs_nrm, lat_nrm, int_mat)
    dia_lgm_params = dia_lgm_man.join_params(dia_obs_nrm, lat_nrm, int_mat)
    iso_lgm_params = iso_lgm_man.join_params(iso_obs_nrm, lat_nrm, int_mat)

    # Big normals
    pod_nrm_params = pod_lgm_man.to_normal(pod_lgm_params)
    dia_nrm_params = dia_lgm_man.to_normal(dia_lgm_params)
    iso_nrm_params = iso_lgm_man.to_normal(iso_lgm_params)

    ### To Natural Tests ###

    pod_lgm_means = pod_lgm_man.to_natural(pod_lgm_params)
    pod_nrm_means = nor_man.to_natural(pod_nrm_params)
    assert jnp.allclose(pod_lgm_means.params, pod_nrm_means.params, rtol=1e-5)

    dia_lgm_means = dia_lgm_man.to_natural(dia_lgm_params)
    dia_nrm_means = nor_man.to_natural(dia_nrm_params)
    assert jnp.allclose(dia_lgm_means.params, dia_nrm_means.params, rtol=1e-5)

    iso_lgm_means = iso_lgm_man.to_natural(iso_lgm_params)
    iso_nrm_means = nor_man.to_natural(iso_nrm_params)
    assert jnp.allclose(iso_lgm_means.params, iso_nrm_means.params, rtol=1e-5)
    # # Test mean coordinate recovery
    # for model, mean_params in [
    #     (pod_man, pod_lgm_params),
    #     (dia_man, dia_lgm_params),
    #     (iso_man, iso_lgm_params),
    # ]:
    #     # Compare mean parameters
    #     nat_params = model.to_natural(mean_params)
    #     remean_params = model.to_natural(nat_params)
    #
    #     assert jnp.allclose(nat_params.params, renat_params.params, rtol=1e-5)
