"""Test script for hierarchical mixture of Gaussians."""

import json
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Diagonal, Mean, Natural, Point, PositiveDefinite
from goal.models import (
    Categorical,
    Covariance,
    Euclidean,
    FactorAnalysis,
    HierarchicalMixtureOfGaussians,
    Normal,
)

from .common import HMoGResults, analysis_path

### Constants ###

# Model parameters from Haskell code
WIDTH = 0.1  # Width parameter
HEIGHT = 8.0  # Height parameter
SEP = 5.0  # Separation parameter

### Ground Truth ###


def create_ground_truth_model() -> (
    tuple[
        HierarchicalMixtureOfGaussians[Diagonal],
        Point[Natural, HierarchicalMixtureOfGaussians[Diagonal]],
    ]
):
    """Create ground truth hierarchical mixture of Gaussians model."""
    hmog = HierarchicalMixtureOfGaussians(
        obs_dim=2,
        obs_rep=Diagonal,
        lat_dim=1,
        n_components=2,
    )

    # Create latent categorical prior
    z_nat = Point[Natural, Categorical](jnp.array([0.0]))

    # Create latent Gaussian components
    with hmog.upr_man as um:
        y0_nat = um.obs_man.join_mean_covariance(
            Point[Mean, Euclidean](jnp.array([-SEP / 2])),
            Point[Mean, Covariance[PositiveDefinite]](jnp.array([[1.0]])),
        )
        y1_nat = um.obs_man.join_mean_covariance(
            Point[Mean, Euclidean](jnp.array([SEP / 2])),
            Point[Mean, Covariance[PositiveDefinite]](jnp.array([[1.0]])),
        )

        y_mix_nat = um.join_natural_mixture([y0_nat, y1_nat], z_nat)

    # Create observable normal with diagonal covariance
    x = Normal(2, Diagonal)
    x_nat = x.join_mean_covariance(
        Point[Mean, Euclidean](jnp.array([0.0, 0.0])),
        Point[Mean, Covariance[Diagonal]](jnp.array([WIDTH, HEIGHT])),
    )

    # Create factor analysis model
    fa = FactorAnalysis(2, 1)
    # Need to handle the factor analysis parameters properly
    fa_nat = fa.join_params(
        x_nat,
        Point[Natural, LinearMap[Rectangular, Euclidean, Euclidean]](
            jnp.array([1.0, 0.0])
        ),
    )

    return hmog, hmog.join_conjugated(fa_nat, y_mix_nat)


### Training ###


def fit_hmog(
    key: Array,
    hmog: HierarchicalMixtureOfGaussians[Diagonal],
    n_steps: int,
    sample: Array,
) -> tuple[Array, Point[Natural, HierarchicalMixtureOfGaussians[Diagonal]]]:
    """Train HMoG model using expectation maximization."""
    init_params = hmog.shape_initialize(key)

    def em_step(
        carry: Point[Natural, HierarchicalMixtureOfGaussians[Diagonal]], _: Any
    ) -> tuple[Point[Natural, HierarchicalMixtureOfGaussians[Diagonal]], Array]:
        params = carry
        ll = hmog.average_log_observable_density(params, sample)
        next_params = hmog.expectation_maximization(params, sample)
        return next_params, ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), final_params


fit_hmog = jax.jit(fit_hmog, static_argnames=["hmog", "n_steps"])

### Analysis ###


def compute_hmog_results(
    key: Array,
    n_obs: int = 400,
    n_steps: int = 200,
    n_points: int = 100,
) -> HMoGResults:
    """Generate samples and train model following Haskell implementation."""

    # Create ground truth model
    true_hmog = create_ground_truth_model()

    # Generate samples
    key_sample, key_train = jax.random.split(key)
    sample = true_hmog.observable_sample(key_sample, true_hmog, n_obs)

    # Create HMoG model
    hmog = HierarchicalMixtureOfGaussians(2, Diagonal, 1, 2)

    # Train model
    lls, final_params = fit_hmog(key_train, hmog, n_steps, sample)

    # Create evaluation grids
    x1 = jnp.linspace(-10, 10, n_points)
    x2 = jnp.linspace(-10, 10, n_points)
    x1s, x2s = jnp.meshgrid(x1, x2)
    grid_points = jnp.stack([x1s.ravel(), x2s.ravel()], axis=1)

    y = jnp.linspace(-6, 6, n_points)

    # Compute densities
    def compute_observable_density(params):
        return jax.vmap(hmog.observable_density, in_axes=(None, 0))(
            params, grid_points
        ).reshape(x1s.shape)

    true_density = compute_observable_density(true_hmog)
    init_density = compute_observable_density(init_params)
    final_density = compute_observable_density(final_params)

    # Get mixture densities
    def compute_mixture_density(params):
        mix_model = hmog.split_conjugated(params)[1]
        return jax.vmap(mix_model.observable_density, in_axes=(None, 0))(
            mix_model, y[:, None]
        )

    true_mix = compute_mixture_density(true_hmog)
    init_mix = compute_mixture_density(init_params)
    final_mix = compute_mixture_density(final_params)

    return HMoGResults(
        plot_range_x1=x1.tolist(),
        plot_range_x2=x2.tolist(),
        observations=sample.tolist(),
        log_likelihoods=lls.tolist(),
        observable_densities=[
            true_density.tolist(),
            init_density.tolist(),
            final_density.tolist(),
        ],
        plot_range_y=y.tolist(),
        mixture_densities=[true_mix.tolist(), init_mix.tolist(), final_mix.tolist()],
    )


### Main ###

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_disable_jit", False)


def main():
    """Run hierarchical mixture of Gaussians tests."""
    key = jax.random.PRNGKey(0)

    # Run analysis
    results = compute_hmog_results(key)

    # Save results
    with open(analysis_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
