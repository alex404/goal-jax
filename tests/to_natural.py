"""Test script for analyzing Linear Gaussian Model natural parameter conversion."""

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Diagonal, Mean, Point, PositiveDefinite, Scale, expand_dual
from goal.models import LinearGaussianModel
from goal.models.linear_gaussian_model import _change_of_basis, _dual_composition

jax.config.update("jax_platform_name", "cpu")


"""Compare steps of to_natural_likelihood conversion across different representations."""


def compare_likelihood_steps(
    iso_model: LinearGaussianModel[Scale],
    dia_model: LinearGaussianModel[Diagonal],
    pd_model: LinearGaussianModel[PositiveDefinite],
    iso_means: Point[Mean, LinearGaussianModel[Scale]],
    dia_means: Point[Mean, LinearGaussianModel[Diagonal]],
    pd_means: Point[Mean, LinearGaussianModel[PositiveDefinite]],
) -> None:
    """Compare each step of natural likelihood conversion."""
    print("\n=== Step 0: Deconstruct Parameters ===")
    pd_params = pd_model.to_natural(pd_means)
    (obs_params, _, _) = pd_model.split_params(pd_params)
    (_, obs_prs) = pd_model.obs_man.split_location_precision(obs_params)
    print(
        "PD Observable Covariance:",
        pd_model.obs_man.cov_man.to_dense(obs_prs),
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
    pd_obs_means, pd_lat_means, pd_int_means = pd_model.split_params(pd_means)
    pd_obs_mean, pd_obs_cov = pd_model.obs_man.split_mean_covariance(pd_obs_means)
    pd_lat_mean, pd_lat_cov = pd_model.lat_man.split_mean_covariance(pd_lat_means)
    pd_int_cov = pd_int_means - pd_model.int_man.outer_product(pd_obs_mean, pd_lat_mean)
    print("obs_mean:", pd_obs_mean.params)
    print("obs_cov:\n", pd_model.obs_man.cov_man.to_dense(pd_obs_cov))
    print("lat_mean:", pd_lat_mean.params)
    print("lat_cov:\n", pd_model.lat_man.cov_man.to_dense(pd_lat_cov))
    print("int_cov:\n", pd_model.int_man.to_dense(pd_int_cov))

    print("\n=== Step 2: Construct Precisions ===")

    # Get latent precisions
    print("\nLatent Precisions:")
    iso_lat_prs = iso_model.lat_man.cov_man.inverse(iso_lat_cov)
    dia_lat_prs = dia_model.lat_man.cov_man.inverse(dia_lat_cov)
    pd_lat_prs = pd_model.lat_man.cov_man.inverse(pd_lat_cov)
    print("Isotropic:\n", iso_model.lat_man.cov_man.to_dense(iso_lat_prs))
    print("Diagonal:\n", dia_model.lat_man.cov_man.to_dense(dia_lat_prs))
    print("PositiveDefinite:\n", pd_model.lat_man.cov_man.to_dense(pd_lat_prs))

    # Get interaction transpose
    print("\nInteraction Transpose:")
    iso_int_man_t = iso_model.int_man.transpose_manifold()
    dia_int_man_t = dia_model.int_man.transpose_manifold()
    pd_int_man_t = pd_model.int_man.transpose_manifold()

    iso_int_cov_t = iso_model.int_man.transpose(iso_int_cov)
    dia_int_cov_t = dia_model.int_man.transpose(dia_int_cov)
    pd_int_cov_t = pd_model.int_man.transpose(pd_int_cov)
    print("Isotropic:\n", iso_model.int_man.to_dense(iso_int_cov_t))
    print("Diagonal:\n", dia_model.int_man.to_dense(dia_int_cov_t))
    print("PositiveDefinite:\n", pd_model.int_man.to_dense(pd_int_cov_t))

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
        pd_int_man_t, pd_int_cov_t, pd_model.lat_man.cov_man, pd_lat_prs
    )
    print("Isotropic:\n", iso_cob_man.to_dense(iso_cob))
    print("Diagonal:\n", dia_cob_man.to_dense(dia_cob))
    print("PositiveDefinite:\n", pd_cob_man.to_dense(pd_cob))

    # Get observable covariance adjusted by change of basis
    print("\nObservable Covariance minus Change of Basis:")
    iso_shaped_cob = iso_model.obs_man.cov_man.from_dense(iso_cob_man.to_dense(iso_cob))
    dia_shaped_cob = dia_model.obs_man.cov_man.from_dense(dia_cob_man.to_dense(dia_cob))
    pd_shaped_cob = pd_model.obs_man.cov_man.from_dense(pd_cob_man.to_dense(pd_cob))

    print("Isotropic shaped COB:\n", iso_model.obs_man.cov_man.to_dense(iso_shaped_cob))
    print("Diagonal shaped COB:\n", dia_model.obs_man.cov_man.to_dense(dia_shaped_cob))
    print(
        "PositiveDefinite shaped COB:\n",
        pd_model.obs_man.cov_man.to_dense(pd_shaped_cob),
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
        pd_model.obs_man.cov_man.to_dense(pd_obs_cov - pd_shaped_cob),
    )

    # Get observable precisions
    print("\nObservable Precisions:")
    iso_obs_prs = iso_model.obs_man.cov_man.inverse(iso_obs_cov - iso_shaped_cob)
    dia_obs_prs = dia_model.obs_man.cov_man.inverse(dia_obs_cov - dia_shaped_cob)
    pd_obs_prs = pd_model.obs_man.cov_man.inverse(pd_obs_cov - pd_shaped_cob)
    print("Isotropic:\n", iso_model.obs_man.cov_man.to_dense(iso_obs_prs))
    print("Diagonal:\n", dia_model.obs_man.cov_man.to_dense(dia_obs_prs))
    print("PositiveDefinite:\n", pd_model.obs_man.cov_man.to_dense(pd_obs_prs))

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
        pd_model.obs_man.cov_man,
        pd_obs_prs,
        pd_model.int_man,
        expand_dual(pd_int_cov),
        pd_model.lat_man.cov_man,
        pd_lat_prs,
    )
    print("Isotropic:\n", iso_model.int_man.to_dense(iso_int_params))
    print("Diagonal:\n", dia_model.int_man.to_dense(dia_int_params))
    print("PositiveDefinite:\n", pd_model.int_man.to_dense(pd_int_params))

    # # Get observable location parameters
    # print("\nObservable Location Parameters:")
    # iso_obs_loc0 = iso_model.obs_man.cov_man(iso_obs_prs, iso_obs_mean)
    # dia_obs_loc0 = dia_model.obs_man.cov_man(dia_obs_prs, dia_obs_mean)
    # pd_obs_loc0 = pd_model.obs_man.cov_man(pd_obs_prs, pd_obs_mean)
    #
    # iso_obs_loc1 = iso_model.int_man(iso_int_params, iso_lat_mean)
    # dia_obs_loc1 = dia_model.int_man(dia_int_params, dia_lat_mean)
    # pd_obs_loc1 = pd_model.int_man(pd_int_params, pd_lat_mean)
    #
    # print("\nInitial Location Parameters:")
    # print("Isotropic:", iso_obs_loc0.params)
    # print("Diagonal:", dia_obs_loc0.params)
    # print("PositiveDefinite:", pd_obs_loc0.params)
    #
    # print("\nInteraction Terms:")
    # print("Isotropic:", iso_obs_loc1.params)
    # print("Diagonal:", dia_obs_loc1.params)
    # print("PositiveDefinite:", pd_obs_loc1.params)
    #
    # print("\nFinal Observable Location Parameters:")
    # print("Isotropic:", (iso_obs_loc0 - iso_obs_loc1).params)
    # print("Diagonal:", (dia_obs_loc0 - dia_obs_loc1).params)
    # print("PositiveDefinite:", (pd_obs_loc0 - pd_obs_loc1).params)


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


def print_model_params[Rep: PositiveDefinite](
    name: str,
    model: LinearGaussianModel[Rep],
    mean_params: Point[Mean, LinearGaussianModel[Rep]],
) -> None:
    """Print detailed model parameters for debugging."""
    print(f"\n=== {name} Model Parameters ===")

    # Split parameters
    obs_means, lat_means, int_means = model.split_params(mean_params)
    obs_mean, obs_cov = model.obs_man.split_mean_covariance(obs_means)
    lat_mean, lat_cov = model.lat_man.split_mean_covariance(lat_means)

    print("Observable mean:", obs_mean.params)
    print("Observable covariance:\n", model.obs_man.cov_man.to_dense(obs_cov))
    print("Latent mean:", lat_mean.params)
    print("Latent covariance:\n", model.lat_man.cov_man.to_dense(lat_cov))
    print("Interaction matrix:\n", model.int_man.to_dense(int_means))


def compare_natural_conversion(
    iso_model: LinearGaussianModel[Scale],
    dia_model: LinearGaussianModel[Diagonal],
    pd_model: LinearGaussianModel[PositiveDefinite],
    iso_means: Point[Mean, LinearGaussianModel[Scale]],
    dia_means: Point[Mean, LinearGaussianModel[Diagonal]],
    pd_means: Point[Mean, LinearGaussianModel[PositiveDefinite]],
) -> None:
    """Compare the steps of natural parameter conversion across models."""
    print("\n=== Natural Parameter Conversion Steps ===")

    # Print initial mean parameters
    print_model_params("Isotropic", iso_model, iso_means)
    print_model_params("Diagonal", dia_model, dia_means)
    print_model_params("Positive Definite", pd_model, pd_means)

    # Convert to natural parameters
    print("\n=== Converting to Natural Parameters ===")

    # First convert to natural likelihood parameters
    iso_lkl = iso_model.to_natural_likelihood(iso_means)
    dia_lkl = dia_model.to_natural_likelihood(dia_means)
    pd_lkl = pd_model.to_natural_likelihood(pd_means)

    print("\nIsotropic likelihood parameters:")
    iso_obs, iso_int = iso_model.lkl_man.split_params(iso_lkl)
    print("Observable:", iso_obs.params)
    print("Interaction:", iso_int.params)

    print("\nDiagonal likelihood parameters:")
    dia_obs, dia_int = dia_model.lkl_man.split_params(dia_lkl)
    print("Observable:", dia_obs.params)
    print("Interaction:", dia_int.params)

    print("\nPositive Definite likelihood parameters:")
    pd_obs, pd_int = pd_model.lkl_man.split_params(pd_lkl)
    print("Observable:", pd_obs.params)
    print("Interaction:", pd_int.params)

    compare_likelihood_steps(
        iso_model, dia_model, pd_model, iso_means, dia_means, pd_means
    )
    # Then get conjugation parameters
    print("\n=== Computing Conjugation Parameters ===")

    iso_chi, iso_rho = iso_model.conjugation_parameters(iso_lkl)
    dia_chi, dia_rho = dia_model.conjugation_parameters(dia_lkl)
    pd_chi, pd_rho = pd_model.conjugation_parameters(pd_lkl)

    print("\nIsotropic conjugation parameters:")
    print("chi:", iso_chi)
    print("rho:", iso_rho.params)

    print("\nDiagonal conjugation parameters:")
    print("chi:", dia_chi)
    print("rho:", dia_rho.params)

    print("\nPositive Definite conjugation parameters:")
    print("chi:", pd_chi)
    print("rho:", pd_rho.params)

    # Finally convert to full natural parameters
    print("\n=== Final Natural Parameters ===")

    iso_nat = iso_model.to_natural(iso_means)
    dia_nat = dia_model.to_natural(dia_means)
    pd_nat = pd_model.to_natural(pd_means)

    print("\nIsotropic natural parameters:", iso_nat.params)
    print("Diagonal natural parameters:", dia_nat.params)
    print("Positive Definite natural parameters:", pd_nat.params)

    iso_rem = iso_model.to_mean(iso_nat)
    dia_rem = dia_model.to_mean(dia_nat)
    pd_rem = pd_model.to_mean(pd_nat)
    print("\nIsotropic remean parameters:", iso_rem.params)
    print("Diagonal remean parameters:", dia_rem.params)
    print("Positive Definite remean parameters:", pd_rem.params)


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
    iso_means = iso_model.average_sufficient_statistic(sample_data)
    iso_params = iso_model.to_natural(iso_means)
    dia_params = iso_model.transform_observable_rep(dia_model, iso_params)
    dia_means = dia_model.to_mean(dia_params)
    pod_params = iso_model.transform_observable_rep(pod_model, iso_params)
    pod_means = pod_model.to_mean(pod_params)

    # Compare natural parameter conversion
    compare_natural_conversion(
        iso_model, dia_model, pod_model, iso_means, dia_means, pod_means
    )


if __name__ == "__main__":
    main()
