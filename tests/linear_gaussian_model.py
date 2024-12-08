"""Tests for LinearGaussianModel conversions to Normal distributions."""

from __future__ import annotations

import jax.numpy as jnp

from goal.harmonium.models import LinearGaussianModel
from goal.transforms import Diagonal, PositiveDefinite, Scale


def test_normal_conversion():
    obs_dim = 2
    lat_dim = 2

    # LGM Models
    pod_man = LinearGaussianModel(obs_dim, PositiveDefinite, lat_dim)
    dia_man = LinearGaussianModel(obs_dim, Diagonal, lat_dim)
    iso_man = LinearGaussianModel(obs_dim, Scale, lat_dim)

    """Create test parameters for the models."""
    # Location parameters
    obs_loc = pod_man.obs_man.loc_man.mean_point(jnp.array([1.0, -0.5]))
    lat_loc = pod_man.lat_man.loc_man.mean_point(jnp.array([0.5, -1.0]))

    # Shape parameters
    cov_array = jnp.array([[2.0, 0.5], [0.5, 1.5]])

    pod_obs_cov = pod_man.obs_man.cov_man.from_dense(cov_array)
    dia_obs_cov = dia_man.obs_man.cov_man.from_dense(cov_array)
    iso_obs_cov = iso_man.obs_man.cov_man.from_dense(cov_array)
    lat_cov = pod_man.lat_man.cov_man.from_dense(jnp.array([[1.0, 0.3], [0.3, 0.5]]))
    int_mat = pod_man.int_man.from_dense(jnp.array([[0.5, -0.3], [0.2, 0.4]]))

    # Normals
    lat_nrm = pod_man.lat_man.from_mean_and_covariance(lat_loc, lat_cov)
    pod_obs_nrm = pod_man.obs_man.from_mean_and_covariance(obs_loc, pod_obs_cov)
    dia_obs_nrm = dia_man.obs_man.from_mean_and_covariance(obs_loc, dia_obs_cov)
    iso_obs_nrm = iso_man.obs_man.from_mean_and_covariance(obs_loc, iso_obs_cov)

    # LGM Parmeters
    pod_lgm_params = pod_man.join_params(pod_obs_nrm, lat_nrm, int_mat)
    dia_lgm_params = dia_man.join_params(dia_obs_nrm, lat_nrm, int_mat)
    iso_lgm_params = iso_man.join_params(iso_obs_nrm, lat_nrm, int_mat)

    repod_lgm_params = pod_man.to_mean(pod_man.to_natural(pod_lgm_params))
    assert jnp.allclose(pod_lgm_params.params, repod_lgm_params.params, rtol=1e-5)

    redia_lgm_params = dia_man.to_mean(dia_man.to_natural(dia_lgm_params))
    assert jnp.allclose(dia_lgm_params.params, redia_lgm_params.params, rtol=1e-5)

    reiso_lgm_params = iso_man.to_mean(iso_man.to_natural(iso_lgm_params))
    assert jnp.allclose(iso_lgm_params.params, reiso_lgm_params.params, rtol=1e-5)
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
