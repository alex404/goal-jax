"""Debug script comparing Normal and LGM representations."""

import jax
import jax.numpy as jnp

from goal.geometry import PositiveDefinite
from goal.models import LinearGaussianModel, Normal

# Configure JAX
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Dimensions
obs_dim = 3
lat_dim = 2
total_dim = obs_dim + lat_dim

# Ground truth parameters
mean = jnp.array([2.0, -1.0, 0.0, 3.0, -2.0])  # Observable then latent
covariance = jnp.array(
    [
        [4.0, 1.5, 0.0, -1.0, 0.5],
        [1.5, 3.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.5, 0.0],
        [-1.0, 0.0, 0.5, 3.0, 1.0],
        [0.5, 0.0, 0.0, 1.0, 2.0],
    ]
)

# Create ground truth normal and generate sample
gt_normal = Normal(total_dim)
gt_mean = gt_normal.loc_man.mean_point(mean)
gt_cov = gt_normal.cov_man.from_dense(covariance)
gt_params = gt_normal.join_mean_covariance(gt_mean, gt_cov)
gt_natural = gt_normal.to_natural(gt_params)

# Generate sample
key = jax.random.PRNGKey(0)
sample = gt_normal.sample(key, gt_natural, 1000)

# Create models
pd_normal = Normal(total_dim, PositiveDefinite)
pd_lgm = LinearGaussianModel(obs_dim, PositiveDefinite, lat_dim)

# Fit models
print("\nFitting models...")
normal_means = pd_normal.average_sufficient_statistic(sample)
normal_natural = pd_normal.to_natural(normal_means)
print("\nNormal natural parameters:", normal_natural.params)

lgm_means = pd_lgm.average_sufficient_statistic(sample)
lgm_natural = pd_lgm.to_natural(lgm_means)
print("\nLGM natural parameters:", lgm_natural.params)

# Convert LGM to Normal
lgm_as_normal = pd_lgm.to_normal(lgm_natural)
print("\nLGM converted to normal parameters:", lgm_as_normal.params)

# Compare log partition functions
print("\nLog partition functions:")
print("Normal:", pd_normal.log_partition_function(normal_natural))
print("LGM:", pd_lgm.log_partition_function(lgm_natural))
print("LGM as Normal:", pd_normal.log_partition_function(lgm_as_normal))

# Compare log densities
print("\nAverage log densities:")
print("Normal:", pd_normal.average_log_density(normal_natural, sample))
print("LGM:", pd_lgm.average_log_density(lgm_natural, sample))
print("LGM as Normal:", pd_normal.average_log_density(lgm_as_normal, sample))

# Compare natural parameters structurally
print("\nComparing natural parameters...")

# Extract components from normal
normal_loc, normal_precision = pd_normal.split_location_precision(normal_natural)
normal_precision_dense = pd_normal.cov_man.to_dense(normal_precision)
print("\nNormal precision matrix:\n", normal_precision_dense)

# Extract components from LGM-as-normal
lgm_normal_loc, lgm_normal_precision = pd_normal.split_location_precision(lgm_as_normal)
lgm_normal_precision_dense = pd_normal.cov_man.to_dense(lgm_normal_precision)
print("\nLGM-as-normal precision matrix:\n", lgm_normal_precision_dense)

# Extract components from LGM
obs_params, lat_params, int_params = pd_lgm.split_params(lgm_natural)
obs_loc, obs_prs = pd_lgm.obs_man.split_location_precision(obs_params)
lat_loc, lat_prs = pd_lgm.lat_man.split_location_precision(lat_params)
print("\nLGM components:")
print("Observable location:", obs_loc.params)
print("Latent location:", lat_loc.params)
print("Observable precision:\n", pd_lgm.obs_man.cov_man.to_dense(obs_prs))
print("Latent precision:\n", pd_lgm.lat_man.cov_man.to_dense(lat_prs))
print("Interaction matrix:\n", pd_lgm.int_man.to_dense(int_params))

if __name__ == "__main__":
    print("\nStarting comparison...")
