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
)
from goal.models import LinearGaussianModel

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
    obs_params, lat_params, int_params = model.split_params(params)

    means = model.to_mean(params)
    obs_means, lat_means, int_means = model.split_params(means)
    dot = model.dot(means, params)
    obs_dot = model.obs_man.dot(obs_means, obs_params)
    lat_dot = model.lat_man.dot(lat_means, lat_params)
    int_dot = model.int_man.dot(int_means, int_params)

    sample_stats = model.average_sufficient_statistic(sample)
    obs_stats, lat_stats, int_stats = model.split_params(sample_stats)
    stat_dot = model.dot(sample_stats, params)
    obs_stat_dot = model.obs_man.dot(obs_stats, obs_params)
    lat_stat_dot = model.lat_man.dot(lat_stats, lat_params)
    int_stat_dot = model.int_man.dot(int_stats, int_params)

    # print("Params:", params.params)
    # print("Means:", means.params)
    # print("Average statistics:", sample_stats)
    # print("Reparams:", model.to_natural(means).params)
    # print("dot product:", dot)
    # print("average data dot product:", stat_dot)
    # print("log likelihood:", model.average_log_density(params, sample))
    # print("log partition:", model.log_partition_function(params))
    print(f"Observable params: {obs_params.params}")
    print(f"\nObservable means: {obs_means.params}")
    print(f"\nObservable stats: {obs_stats.params}")
    print(f"Observable dot: {obs_dot}")
    print(f"Observable stat dot: {obs_stat_dot}")
    # print(f"Latent params: {lat_params.params}")
    # print(f"\nLatent means: {lat_means.params}")
    # print(f"\nLatent stats: {lat_stats.params}")
    # print(f"Latent dot: {lat_dot}")
    # print(f"Latent stat dot: {lat_stat_dot}")
    # print(f"Interaction params: {int_params.params}")
    # print(f"\nInteraction means: {int_means.params}")
    # print(f"\nInteraction stats: {int_stats.params}")
    # print(f"Interaction dot: {int_dot}")
    # print(f"Interaction stat dot: {int_stat_dot}")


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
    # print("Isotropic means:", iso_model.split_params(iso_means))
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
