"""Test script for analyzing Linear Gaussian Model natural parameter conversion."""

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Diagonal, PositiveDefinite, Scale, expand_dual
from goal.models import LinearGaussianModel
from goal.models.linear_gaussian_model import _change_of_basis, _dual_composition

jax.config.update("jax_platform_name", "cpu")


"""Compare steps of to_natural_likelihood conversion across different representations."""


def generate_sample_data(
    obs_dim: int = 3, lat_dim: int = 2, n_samples: int = 1000
) -> Array:
    """Generate sample data from a ground truth joint distribution."""
    key = jax.random.PRNGKey(0)

    # Create ground truth parameters for joint distribution
    mean = jnp.concatenate(
        [
            jnp.array([1.0, -0.5, 0.3])[:obs_dim],  # observable mean
            jnp.array([0.2, -0.1])[:lat_dim],  # latent mean
        ]
    )

    # Create block covariance matrix
    obs_cov = jnp.eye(obs_dim) + 0.1 * jnp.ones((obs_dim, obs_dim))
    lat_cov = jnp.eye(lat_dim) + 0.05 * jnp.ones((lat_dim, lat_dim))
    cross_cov = 0.1 * jnp.ones((obs_dim, lat_dim))

    cov = jnp.block([[obs_cov, cross_cov], [cross_cov.T, lat_cov]])

    # Sample from ground truth
    return jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(n_samples,))


def main():
    # Set dimensions
    obs_dim = 3
    lat_dim = 2

    # Generate sample data
    sample_data = generate_sample_data(obs_dim, lat_dim)

    # Create models with different representations
    iso_model = LinearGaussianModel(obs_dim, Scale, lat_dim)
    dia_model = LinearGaussianModel(obs_dim, Diagonal, lat_dim)
    pod_model = LinearGaussianModel(obs_dim, PositiveDefinite, lat_dim)

    # Get isotropic mean parameters
    iso_means0 = iso_model.average_sufficient_statistic(sample_data)
    iso_params = iso_model.to_natural(iso_means0)
    dia_params = iso_model.transform_observable_rep(dia_model, iso_params)
    dia_means = dia_model.to_mean(dia_params)
    pod_params = iso_model.transform_observable_rep(pod_model, iso_params)
    pod_means = pod_model.to_mean(pod_params)
    iso_means = iso_model.to_mean(iso_params)

    """Compare each step of natural likelihood conversion."""
    print("\n=== Step 0: Deconstruct Parameters ===")
    pod_params = pod_model.to_natural(pod_means)
    (obs_params, _, _) = pod_model.split_params(pod_params)
    (_, obs_prs) = pod_model.obs_man.split_location_precision(obs_params)
    print(
        "PD Observable Covariance:",
        pod_model.obs_man.cov_man.to_dense(obs_prs),
    )

    print("\n=== Step 1: Deconstruct Parameters ===")

    # Split parameters for each model
    print("\nIsotropic Model:")
    iso_obs_means, iso_lat_means, iso_int_means = iso_model.split_params(iso_means)
    iso_obs_mean, iso_obs_cov = iso_model.obs_man.split_mean_covariance(iso_obs_means)
    iso_lat_mean, iso_lat_cov = iso_model.lat_man.split_mean_covariance(iso_lat_means)
    iso_int_cov = iso_int_means - iso_model.int_man.outer_product(
        iso_obs_mean, iso_lat_mean
    )
    print("obs_mean:", iso_obs_mean.params)
    print("obs_cov:\n", iso_model.obs_man.cov_man.to_dense(iso_obs_cov))
    print("lat_mean:", iso_lat_mean.params)
    print("lat_cov:\n", iso_model.lat_man.cov_man.to_dense(iso_lat_cov))
    print("int_cov:\n", iso_model.int_man.to_dense(iso_int_cov))

    print("\nDiagonal Model:")
    dia_obs_means, dia_lat_means, dia_int_means = dia_model.split_params(dia_means)
    dia_obs_mean, dia_obs_cov = dia_model.obs_man.split_mean_covariance(dia_obs_means)
    dia_lat_mean, dia_lat_cov = dia_model.lat_man.split_mean_covariance(dia_lat_means)
    dia_int_cov = dia_int_means - dia_model.int_man.outer_product(
        dia_obs_mean, dia_lat_mean
    )
    print("obs_mean:", dia_obs_mean.params)
    print("obs_cov:\n", dia_model.obs_man.cov_man.to_dense(dia_obs_cov))
    print("lat_mean:", dia_lat_mean.params)
    print("lat_cov:\n", dia_model.lat_man.cov_man.to_dense(dia_lat_cov))
    print("int_cov:\n", dia_model.int_man.to_dense(dia_int_cov))

    print("\nPositive Definite Model:")
    pd_obs_means, pd_lat_means, pd_int_means = pod_model.split_params(pod_means)
    pd_obs_mean, pd_obs_cov = pod_model.obs_man.split_mean_covariance(pd_obs_means)
    pd_lat_mean, pd_lat_cov = pod_model.lat_man.split_mean_covariance(pd_lat_means)
    pd_int_cov = pd_int_means - pod_model.int_man.outer_product(
        pd_obs_mean, pd_lat_mean
    )
    print("obs_mean:", pd_obs_mean.params)
    print("obs_cov:\n", pod_model.obs_man.cov_man.to_dense(pd_obs_cov))
    print("lat_mean:", pd_lat_mean.params)
    print("lat_cov:\n", pod_model.lat_man.cov_man.to_dense(pd_lat_cov))
    print("int_cov:\n", pod_model.int_man.to_dense(pd_int_cov))

    print("\n=== Step 2: Construct Precisions ===")

    # Get latent precisions
    print("\nLatent Precisions:")
    iso_lat_prs = iso_model.lat_man.cov_man.inverse(iso_lat_cov)
    dia_lat_prs = dia_model.lat_man.cov_man.inverse(dia_lat_cov)
    pd_lat_prs = pod_model.lat_man.cov_man.inverse(pd_lat_cov)
    print("Isotropic:\n", iso_model.lat_man.cov_man.to_dense(iso_lat_prs))
    print("Diagonal:\n", dia_model.lat_man.cov_man.to_dense(dia_lat_prs))
    print("PositiveDefinite:\n", pod_model.lat_man.cov_man.to_dense(pd_lat_prs))

    # Get interaction transpose
    print("\nInteraction Transpose:")
    iso_int_man_t = iso_model.int_man.transpose_manifold()
    dia_int_man_t = dia_model.int_man.transpose_manifold()
    pd_int_man_t = pod_model.int_man.transpose_manifold()

    iso_int_cov_t = iso_model.int_man.transpose(iso_int_cov)
    dia_int_cov_t = dia_model.int_man.transpose(dia_int_cov)
    pd_int_cov_t = pod_model.int_man.transpose(pd_int_cov)
    print("Isotropic:\n", iso_model.int_man.to_dense(iso_int_cov_t))
    print("Diagonal:\n", dia_model.int_man.to_dense(dia_int_cov_t))
    print("PositiveDefinite:\n", pod_model.int_man.to_dense(pd_int_cov_t))

    # Get change of basis
    print("\nChange of Basis:")
    # Note: _change_of_basis is internal so we replicate its logic here
    iso_cob_man, iso_cob = _change_of_basis(
        iso_int_man_t, iso_int_cov_t, iso_model.lat_man.cov_man, iso_lat_prs
    )
    dia_cob_man, dia_cob = _change_of_basis(
        dia_int_man_t, dia_int_cov_t, dia_model.lat_man.cov_man, dia_lat_prs
    )
    pd_cob_man, pd_cob = _change_of_basis(
        pd_int_man_t, pd_int_cov_t, pod_model.lat_man.cov_man, pd_lat_prs
    )
    print("Isotropic:\n", iso_cob_man.to_dense(iso_cob))
    print("Diagonal:\n", dia_cob_man.to_dense(dia_cob))
    print("PositiveDefinite:\n", pd_cob_man.to_dense(pd_cob))

    # Get observable covariance adjusted by change of basis
    print("\nObservable Covariance minus Change of Basis:")
    iso_shaped_cob = iso_model.obs_man.cov_man.from_dense(iso_cob_man.to_dense(iso_cob))
    dia_shaped_cob = dia_model.obs_man.cov_man.from_dense(dia_cob_man.to_dense(dia_cob))
    pd_shaped_cob = pod_model.obs_man.cov_man.from_dense(pd_cob_man.to_dense(pd_cob))

    print("Isotropic shaped COB:\n", iso_model.obs_man.cov_man.to_dense(iso_shaped_cob))
    print("Diagonal shaped COB:\n", dia_model.obs_man.cov_man.to_dense(dia_shaped_cob))
    print(
        "PositiveDefinite shaped COB:\n",
        pod_model.obs_man.cov_man.to_dense(pd_shaped_cob),
    )

    print("\nAdjusted Observable Covariance:")
    print(
        "Isotropic:\n", iso_model.obs_man.cov_man.to_dense(iso_obs_cov - iso_shaped_cob)
    )
    print(
        "Diagonal:\n", dia_model.obs_man.cov_man.to_dense(dia_obs_cov - dia_shaped_cob)
    )
    print(
        "PositiveDefinite:\n",
        pod_model.obs_man.cov_man.to_dense(pd_obs_cov - pd_shaped_cob),
    )

    # Get observable precisions
    print("\nObservable Precisions:")
    iso_obs_prs = iso_model.obs_man.cov_man.inverse(iso_obs_cov - iso_shaped_cob)
    dia_obs_prs = dia_model.obs_man.cov_man.inverse(dia_obs_cov - dia_shaped_cob)
    pd_obs_prs = pod_model.obs_man.cov_man.inverse(pd_obs_cov - pd_shaped_cob)
    print("Isotropic:\n", iso_model.obs_man.cov_man.to_dense(iso_obs_prs))
    print("Diagonal:\n", dia_model.obs_man.cov_man.to_dense(dia_obs_prs))
    print("PositiveDefinite:\n", pod_model.obs_man.cov_man.to_dense(pd_obs_prs))

    print("\n=== Step 3: Construct Location Parameters ===")

    # Get interaction parameters from dual composition
    print("\nInteraction Parameters:")
    _, iso_int_params = _dual_composition(
        iso_model.obs_man.cov_man,
        iso_obs_prs,
        iso_model.int_man,
        expand_dual(iso_int_cov),
        iso_model.lat_man.cov_man,
        iso_lat_prs,
    )
    _, dia_int_params = _dual_composition(
        dia_model.obs_man.cov_man,
        dia_obs_prs,
        dia_model.int_man,
        expand_dual(dia_int_cov),
        dia_model.lat_man.cov_man,
        dia_lat_prs,
    )
    _, pd_int_params = _dual_composition(
        pod_model.obs_man.cov_man,
        pd_obs_prs,
        pod_model.int_man,
        expand_dual(pd_int_cov),
        pod_model.lat_man.cov_man,
        pd_lat_prs,
    )
    print("Isotropic:\n", iso_model.int_man.to_dense(iso_int_params))
    print("Diagonal:\n", dia_model.int_man.to_dense(dia_int_params))
    print("PositiveDefinite:\n", pod_model.int_man.to_dense(pd_int_params))


if __name__ == "__main__":
    main()
