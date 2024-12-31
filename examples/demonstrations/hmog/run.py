"""Test script for hierarchical mixture of Gaussians."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Diagonal,
    Mean,
    Natural,
    Point,
    PositiveDefinite,
    Scale,
)
from goal.models import (
    AnalyticHMoG,
    Categorical,
    Covariance,
    Euclidean,
)

from ...shared import initialize_jax, initialize_paths, save_results
from .types import HMoGResults

### Constants ###

# Model parameters from Haskell code
WIDTH = 0.1  # Width parameter
HEIGHT = 8.0  # Height parameter
SEP = 3.0  # Separation parameter

### Ground Truth ###


def create_ground_truth_model() -> (
    tuple[
        AnalyticHMoG[Diagonal],
        Point[Natural, AnalyticHMoG[Diagonal]],
    ]
):
    """Create ground truth hierarchical mixture of Gaussians model."""
    hmog = AnalyticHMoG(
        obs_dim=2,
        obs_rep=Diagonal,
        lat_dim=1,
        n_components=2,
    )

    # Create latent categorical prior
    cat_nat = Point[Natural, Categorical](jnp.array([0.5]))
    cat_means = hmog.lat_man.lat_man.to_mean(cat_nat)

    # Create latent Gaussian components
    with hmog.lat_man as um:
        y0_means = um.obs_man.join_mean_covariance(
            Point[Mean, Euclidean](jnp.array([-SEP / 2])),
            Point[Mean, Covariance[PositiveDefinite]](jnp.array([1.0])),
        )
        y1_means = um.obs_man.join_mean_covariance(
            Point[Mean, Euclidean](jnp.array([SEP / 2])),
            Point[Mean, Covariance[PositiveDefinite]](jnp.array([1.0])),
        )

        mix_means = um.join_mean_mixture([y0_means, y1_means], cat_means)
        mix_params = um.to_natural(mix_means)

    # Create observable normal with diagonal covariance
    obs_means = hmog.obs_man.join_mean_covariance(
        Point[Mean, Euclidean](jnp.array([0.0, 0.0])),
        Point[Mean, Covariance[Diagonal]](jnp.array([WIDTH, HEIGHT])),
    )
    obs_params = hmog.obs_man.to_natural(obs_means)

    # NB: Multiplying the interaction parameters by the observable precision makes them scale more intuitively
    int_mat0 = jnp.array([1.0, 0.0])
    obs_prs = hmog.obs_man.split_location_precision(obs_params)[1]
    int_mat = hmog.int_man.from_dense(hmog.obs_man.cov_man.to_dense(obs_prs) @ int_mat0)

    lkl_params = hmog.lkl_man.join_params(obs_params, int_mat)

    hmog_params = hmog.join_conjugated(lkl_params, mix_params)

    return hmog, hmog_params


### Training ###


def initialize_hmog[Rep: PositiveDefinite](
    key: Array,
    hmog: AnalyticHMoG[Rep],
    sample: Array,
    n_components: int = 2,
) -> Point[Natural, AnalyticHMoG[Rep]]:
    keys = jax.random.split(key, 2)

    # Initialize observable normal with data statistics
    obs_means = hmog.obs_man.average_sufficient_statistic(sample)
    obs_params = hmog.obs_man.to_natural(obs_means)

    # Create latent categorical prior - slightly break symmetry
    cat_params = hmog.lat_man.lat_man.initialize(keys[0])
    cat_means = hmog.lat_man.lat_man.to_mean(cat_params)

    # Initialize separated components
    with hmog.lat_man as um:
        components = []
        sg = 1
        stp = 4 * sg
        strt = -stp * (n_components - 1) / 2
        rng = jnp.arange(strt, strt + n_components * stp, stp)
        for i in range(n_components):
            comp_means = um.obs_man.join_mean_covariance(
                Point[Mean, Euclidean](jnp.atleast_1d(rng[i])),
                Point[Mean, Covariance[PositiveDefinite]](jnp.array([sg**2])),
            )
            components.append(comp_means)

        mix_means = um.join_mean_mixture(components, cat_means)
        mix_params = um.to_natural(mix_means)

    # Initialize interaction matrix scaled by precision
    obs_prs = hmog.obs_man.split_location_precision(obs_params)[1]
    int_mat0 = 0.1 * jax.random.normal(keys[1], shape=(2,))
    int_mat = hmog.int_man.from_dense(hmog.obs_man.cov_man.to_dense(obs_prs) @ int_mat0)

    # Combine parameters
    lkl_params = hmog.lkl_man.join_params(obs_params, int_mat)
    return hmog.join_conjugated(lkl_params, mix_params)


def fit_hmog[R: PositiveDefinite](
    key: Array,
    hmog: AnalyticHMoG[R],
    n_steps: int,
    sample: Array,
) -> tuple[
    Array,
    Point[Natural, AnalyticHMoG[R]],
    Point[Natural, AnalyticHMoG[R]],
]:
    """Train HMoG model using expectation maximization."""
    init_params = initialize_hmog(key, hmog, sample)

    def em_step(
        carry: Point[Natural, AnalyticHMoG[R]], _: Any
    ) -> tuple[Point[Natural, AnalyticHMoG[R]], Array]:
        params = carry
        ll = hmog.average_log_observable_density(params, sample)
        next_params = hmog.expectation_maximization(params, sample)
        return next_params, ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_hmog = jax.jit(fit_hmog, static_argnames=["hmog", "n_steps"])

### Analysis ###


def compute_hmog_results(
    key: Array,
    sample_size: int = 400,
    n_steps: int = 200,
    n_points: int = 100,
) -> HMoGResults:
    """Generate samples and train model following Haskell implementation."""

    # Create ground truth model
    gt_hmog, gt_hmog_params = create_ground_truth_model()

    # Generate samples
    key_sample, key_train = jax.random.split(key)
    sample = gt_hmog.observable_sample(key_sample, gt_hmog_params, sample_size)

    # Create HMoG model
    iso_hmog = AnalyticHMoG(2, Scale, 1, 2)
    dia_hmog = AnalyticHMoG(2, Diagonal, 1, 2)

    # Train model
    iso_lls, init_iso_params, final_iso_params = fit_hmog(
        key_train, iso_hmog, n_steps, sample
    )
    dia_lls, init_dia_params, final_dia_params = fit_hmog(
        key_train, dia_hmog, n_steps, sample
    )

    # Create evaluation grids
    x1 = jnp.linspace(-10, 10, n_points)
    x2 = jnp.linspace(-10, 10, n_points)
    x1s, x2s = jnp.meshgrid(x1, x2)
    grid_points = jnp.stack([x1s.ravel(), x2s.ravel()], axis=1)

    y = jnp.linspace(-6, 6, n_points)

    # Compute densities
    def compute_observable_density[R: PositiveDefinite](
        model: AnalyticHMoG[R],
        params: Point[Natural, AnalyticHMoG[R]],
    ):
        return jax.vmap(model.observable_density, in_axes=(None, 0))(
            params, grid_points
        ).reshape(x1s.shape)

    true_density = compute_observable_density(gt_hmog, gt_hmog_params)
    init_iso_density = compute_observable_density(iso_hmog, init_iso_params)
    final_iso_density = compute_observable_density(iso_hmog, final_iso_params)
    init_dia_density = compute_observable_density(dia_hmog, init_dia_params)
    final_dia_density = compute_observable_density(dia_hmog, final_dia_params)

    # Get mixture densities
    def compute_mixture_density[R: PositiveDefinite](
        model: AnalyticHMoG[R],
        params: Point[Natural, AnalyticHMoG[R]],
    ):
        mix_model = model.split_conjugated(params)[1]
        return jax.vmap(model.lat_man.observable_density, in_axes=(None, 0))(
            mix_model, y[:, None]
        )

    true_mix = compute_mixture_density(gt_hmog, gt_hmog_params)
    init_iso_mix = compute_mixture_density(iso_hmog, init_iso_params)
    final_iso_mix = compute_mixture_density(iso_hmog, final_iso_params)
    init_dia_mix = compute_mixture_density(dia_hmog, init_dia_params)
    final_dia_mix = compute_mixture_density(dia_hmog, final_dia_params)

    return HMoGResults(
        plot_range_x1=x1.tolist(),
        plot_range_x2=x2.tolist(),
        observations=sample.tolist(),
        log_likelihoods={"Isotropic": iso_lls.tolist(), "Diagonal": dia_lls.tolist()},
        observable_densities={
            "Ground Truth": true_density.tolist(),
            "Isotropic Initial": init_iso_density.tolist(),
            "Isotropic Final": final_iso_density.tolist(),
            "Diagonal Initial": init_dia_density.tolist(),
            "Diagonal Final": final_dia_density.tolist(),
        },
        plot_range_y=y.tolist(),
        mixture_densities={
            "Ground Truth": true_mix.tolist(),
            "Isotropic Initial": init_iso_mix.tolist(),
            "Isotropic Final": final_iso_mix.tolist(),
            "Diagonal Initial": init_dia_mix.tolist(),
            "Diagonal Final": final_dia_mix.tolist(),
        },
    )


### Main ###


def main():
    """Run hierarchical mixture of Gaussians tests."""
    initialize_jax()
    paths = initialize_paths(__file__)
    key = jax.random.PRNGKey(0)

    # Run analysis
    results = compute_hmog_results(key)

    # Save results
    save_results(results, paths)


if __name__ == "__main__":
    main()
