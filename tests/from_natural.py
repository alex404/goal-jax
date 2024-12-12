"""Test script for analyzing Linear Gaussian Model natural parameter conversion."""

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Diagonal,
    Natural,
    Point,
    PositiveDefinite,
    Scale,
    expand_dual,
)
from goal.models import LinearGaussianModel
from goal.models.linear_gaussian_model import _change_of_basis

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


# Break down dot products
def print_dot_components[Rep: PositiveDefinite](
    model: LinearGaussianModel[Rep],
    params: Point[Natural, LinearGaussianModel[Rep]],
    sample: Array,
):
    means = model.to_mean(params)
    obs_params, lat_params, int_params = model.split_params(params)
    obs_means, lat_means, int_means = model.split_params(means)

    dot = model.dot(means, params)
    obs_dot = model.obs_man.dot(obs_means, obs_params)
    lat_dot = model.lat_man.dot(lat_means, lat_params)
    int_dot = model.int_man.dot(int_means, int_params)

    print("Params:", params.params)
    print("Means:", means.params)
    print("Reparams:", model.to_natural(means).params)
    print("dot product:", dot)
    print("log likelihood:", model.average_log_density(params, sample))

    print("log partition:", model.log_partition_function(params))
    print(f"Observable params: {obs_params.params}")
    print(f"\nObservable means: {obs_means.params}")
    print(f"Observable dot: {obs_dot}")
    print(f"Latent params: {lat_params.params}")
    print(f"\nLatent means: {lat_means.params}")
    print(f"Latent dot: {lat_dot}")
    print(f"Interaction params: {int_params.params}")
    print(f"\nInteraction means: {int_means.params}")
    print(f"Interaction dot: {int_dot}")


def compare_conjugation_steps(
    iso_model: LinearGaussianModel[Scale],
    dia_model: LinearGaussianModel[Diagonal],
    pod_model: LinearGaussianModel[PositiveDefinite],
    iso_params: Point[Natural, LinearGaussianModel[Scale]],
    dia_params: Point[Natural, LinearGaussianModel[Diagonal]],
    pod_params: Point[Natural, LinearGaussianModel[PositiveDefinite]],
) -> None:
    """Compare each step of conjugation parameter computation."""
    print("\n=== Step 1: Split Natural Parameters ===")

    # Split parameters for isotropic
    print("\nIsotropic Model:")
    iso_lkl_params = iso_model.likelihood_function(iso_params)
    obs_bias_iso, int_mat_iso = iso_model.lkl_man.split_params(iso_lkl_params)
    obs_loc_iso, obs_prec_iso = iso_model.obs_man.split_location_precision(obs_bias_iso)
    print("obs_loc:", obs_loc_iso.params)
    print("obs_prec:\n", iso_model.obs_man.cov_man.to_dense(obs_prec_iso))
    print("int_mat:\n", iso_model.int_man.to_dense(int_mat_iso))

    # Split parameters for diagonal
    print("\nDiagonal Model:")
    dia_lkl_params = dia_model.likelihood_function(dia_params)
    obs_bias_dia, int_mat_dia = dia_model.lkl_man.split_params(dia_lkl_params)
    obs_loc_dia, obs_prec_dia = dia_model.obs_man.split_location_precision(obs_bias_dia)
    print("obs_loc:", obs_loc_dia.params)
    print("obs_prec:\n", dia_model.obs_man.cov_man.to_dense(obs_prec_dia))
    print("int_mat:\n", dia_model.int_man.to_dense(int_mat_dia))

    # Split parameters for positive definite
    print("\nPositive Definite Model:")
    pod_lkl_params = pod_model.likelihood_function(pod_params)
    obs_bias_pd, int_mat_pd = pod_model.lkl_man.split_params(pod_lkl_params)
    obs_loc_pd, obs_prec_pd = pod_model.obs_man.split_location_precision(obs_bias_pd)
    print("obs_loc:", obs_loc_pd.params)
    print("obs_prec:\n", pod_model.obs_man.cov_man.to_dense(obs_prec_pd))
    print("int_mat:\n", pod_model.int_man.to_dense(int_mat_pd))

    print("\n=== Step 2: Intermediate Computations ===")

    # Isotropic computations
    obs_sigma_iso = iso_model.obs_man.cov_man.inverse(obs_prec_iso)
    log_det_iso = iso_model.obs_man.cov_man.logdet(obs_sigma_iso)
    obs_mean_iso = iso_model.obs_man.cov_man(obs_sigma_iso, expand_dual(obs_loc_iso))

    print("\nIsotropic Model:")
    print("obs_sigma:\n", iso_model.obs_man.cov_man.to_dense(obs_sigma_iso))
    print("log_det:", log_det_iso)
    print("obs_mean:", obs_mean_iso.params)

    # Diagonal computations
    obs_sigma_dia = dia_model.obs_man.cov_man.inverse(obs_prec_dia)
    log_det_dia = dia_model.obs_man.cov_man.logdet(obs_sigma_dia)
    obs_mean_dia = dia_model.obs_man.cov_man(obs_sigma_dia, expand_dual(obs_loc_dia))

    print("\nDiagonal Model:")
    print("obs_sigma:\n", dia_model.obs_man.cov_man.to_dense(obs_sigma_dia))
    print("log_det:", log_det_dia)
    print("obs_mean:", obs_mean_dia.params)

    # Positive definite computations
    obs_sigma_pd = pod_model.obs_man.cov_man.inverse(obs_prec_pd)
    log_det_pd = pod_model.obs_man.cov_man.logdet(obs_sigma_pd)
    obs_mean_pd = pod_model.obs_man.cov_man(obs_sigma_pd, expand_dual(obs_loc_pd))

    print("\nPositive Definite Model:")
    print("obs_sigma:\n", pod_model.obs_man.cov_man.to_dense(obs_sigma_pd))
    print("log_det:", log_det_pd)
    print("obs_mean:", obs_mean_pd.params)

    print("\n=== Step 3: Computing Conjugation Parameters ===")

    # Isotropic chi computation
    print("\nIsotropic Model:")
    quad_term_iso = 0.5 * iso_model.obs_man.loc_man.dot(
        obs_mean_iso, iso_model.obs_man.cov_man(obs_prec_iso, obs_mean_iso)
    )
    chi_iso = quad_term_iso + 0.5 * log_det_iso
    print("quadratic term:", quad_term_iso)
    print("chi:", chi_iso)

    # Get rho parameters
    rho_mean_iso = iso_model.int_man.transpose_apply(int_mat_iso, obs_mean_iso)
    _, rho_shape_iso = _change_of_basis(
        iso_model.int_man, int_mat_iso, iso_model.obs_man.cov_man, obs_sigma_iso
    )
    rho_shape_iso *= -1
    print("rho_mean:", rho_mean_iso.params)
    print("rho_shape:\n", iso_model.lat_man.cov_man.to_dense(rho_shape_iso))

    # Diagonal chi computation
    print("\nDiagonal Model:")
    quad_term_dia = 0.5 * dia_model.obs_man.loc_man.dot(
        obs_mean_dia, dia_model.obs_man.cov_man(obs_prec_dia, obs_mean_dia)
    )
    chi_dia = quad_term_dia + 0.5 * log_det_dia
    print("quadratic term:", quad_term_dia)
    print("chi:", chi_dia)

    # Get rho parameters
    rho_mean_dia = dia_model.int_man.transpose_apply(int_mat_dia, obs_mean_dia)
    _, rho_shape_dia = _change_of_basis(
        dia_model.int_man, int_mat_dia, dia_model.obs_man.cov_man, obs_sigma_dia
    )
    rho_shape_dia *= -1
    print("rho_mean:", rho_mean_dia.params)
    print("rho_shape:\n", dia_model.lat_man.cov_man.to_dense(rho_shape_dia))

    # Positive definite chi computation
    print("\nPositive Definite Model:")
    quad_term_pd = 0.5 * pod_model.obs_man.loc_man.dot(
        obs_mean_pd, pod_model.obs_man.cov_man(obs_prec_pd, obs_mean_pd)
    )
    chi_pd = quad_term_pd + 0.5 * log_det_pd
    print("quadratic term:", quad_term_pd)
    print("chi:", chi_pd)

    # Get rho parameters
    rho_mean_pd = pod_model.int_man.transpose_apply(int_mat_pd, obs_mean_pd)
    _, rho_shape_pd = _change_of_basis(
        pod_model.int_man, int_mat_pd, pod_model.obs_man.cov_man, obs_sigma_pd
    )
    rho_shape_pd *= -1
    print("rho_mean:", rho_mean_pd.params)
    print("rho_shape:\n", pod_model.lat_man.cov_man.to_dense(rho_shape_pd))

    print("\n=== Step 4: Comparing Natural-Mean Dot Products ===")


def main():
    # Set dimensions
    obs_dim = 3
    lat_dim = 2

    # Generate sample data
    sample = generate_sample_data(obs_dim, lat_dim)

    # Create models with different representations
    iso_model = LinearGaussianModel(obs_dim, Scale, lat_dim)
    dia_model = LinearGaussianModel(obs_dim, Diagonal, lat_dim)
    pod_model = LinearGaussianModel(obs_dim, PositiveDefinite, lat_dim)

    # Get isotropic mean parameters
    iso_means = iso_model.average_sufficient_statistic(sample)
    iso_params = iso_model.to_natural(iso_means)
    dia_params = iso_model.transform_observable_rep(dia_model, iso_params)
    pod_params = iso_model.transform_observable_rep(pod_model, iso_params)

    # Compare full density computation pieces
    print("\nScale:")
    print_dot_components(iso_model, iso_params, sample)

    print("\nDiagonal:")
    print_dot_components(dia_model, dia_params, sample)

    print("\nPositiveDefinite:")
    print_dot_components(pod_model, pod_params, sample)


if __name__ == "__main__":
    main()
