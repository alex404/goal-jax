"""Test script for hierarchical mixture of Gaussians."""

from typing import Any

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
from goal.models import (
    AnalyticHMoG,
    analytic_hmog,
)

from ..shared import example_paths, initialize_jax
from .types import HMoGResults

### Constants ###

# Model parameters from Haskell code
WIDTH = 0.1  # Width parameter
HEIGHT = 8.0  # Height parameter
SEP = 3.0  # Separation parameter

### Ground Truth ###


def create_ground_truth_model() -> tuple[
    AnalyticHMoG[Diagonal],
    Point[Natural, AnalyticHMoG[Diagonal]],
]:
    """Create ground truth hierarchical mixture of Gaussians model."""
    hmog = analytic_hmog(
        obs_dim=2,
        obs_rep=Diagonal,
        lat_dim=1,
        n_components=2,
    )
    with hmog.lat_man.con_lat_man as cm:
        # Create latent categorical prior
        cat_params = cm.natural_point(jnp.array([0.5]))
        cat_means = cm.to_mean(cat_params)

    # Create latent Gaussian components
    with hmog.lat_man as um, um.obs_man as lm:
        y0_means = lm.join_mean_covariance(
            lm.loc_man.mean_point(jnp.array([-SEP / 2])),
            lm.cov_man.mean_point(jnp.array([1.0])),
        )
        y1_means = lm.join_mean_covariance(
            lm.loc_man.mean_point(jnp.array([SEP / 2])),
            lm.cov_man.mean_point(jnp.array([1.0])),
        )

        # components = mix_man.cmp_man.mean_point(jnp.stack(component_list))
        components = um.cmp_man.mean_point(jnp.stack([y0_means.array, y1_means.array]))
        mix_means = um.join_mean_mixture(components, cat_means)
        mix_params = um.to_natural(mix_means)

    with hmog.obs_man as om:
        # Create observable normal with diagonal covariance
        obs_means = om.join_mean_covariance(
            om.loc_man.mean_point(jnp.array([0.0, 0.0])),
            om.cov_man.mean_point(jnp.array([WIDTH, HEIGHT])),
        )
        obs_params = om.to_natural(obs_means)

        # NB: Multiplying the interaction parameters by the observable precision makes them scale more intuitively
        int_mat0 = jnp.array([1.0, 0.0])
        obs_prs = om.split_location_precision(obs_params)[1]
        int_mat = hmog.int_man.from_dense(om.cov_man.to_dense(obs_prs) @ int_mat0)

    lkl_params = hmog.lkl_man.join_params(obs_params, int_mat)

    hmog_params = hmog.join_conjugated(lkl_params, mix_params)

    return hmog, hmog_params


### Training ###


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
    init_params = hmog.initialize_from_sample(key, sample, shape=2)

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
    n_steps: int = 1000,
    n_points: int = 100,
) -> HMoGResults:
    """Generate samples and train model following Haskell implementation."""

    # Create ground truth model
    gt_hmog, gt_hmog_params = create_ground_truth_model()

    # Generate samples
    key_sample, key_train = jax.random.split(key)
    sample = gt_hmog.observable_sample(key_sample, gt_hmog_params, sample_size)

    # Create HMoG model
    iso_hmog = analytic_hmog(2, Scale, 1, 2)
    dia_hmog = analytic_hmog(2, Diagonal, 1, 2)

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
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)

    # Run analysis
    results = compute_hmog_results(key)

    # Save results
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
