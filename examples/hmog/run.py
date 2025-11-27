"""Test script for hierarchical mixture of Gaussians."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import (
    Diagonal,
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
    AnalyticHMoG,
    Array,
]:
    """Create ground truth hierarchical mixture of Gaussians model."""
    hmog = analytic_hmog(
        obs_dim=2,
        obs_rep=Diagonal(),
        lat_dim=1,
        n_components=2,
    )
    with hmog.pst_man.prr_man as cm:
        # Create latent categorical prior
        cat_natural_params = jnp.array([0.5])
        cat_mean_params = cm.to_mean(cat_natural_params)

    # Create latent Gaussian components
    with hmog.pst_man as um, um.obs_man as lm:
        y0_means = lm.join_mean_covariance(
            jnp.array([-SEP / 2]),
            jnp.array([1.0]),
        )
        y1_means = lm.join_mean_covariance(
            jnp.array([SEP / 2]),
            jnp.array([1.0]),
        )

        # components = mix_man.cmp_man.mean_point(jnp.stack(component_list))
        components = jnp.stack([y0_means, y1_means])
        mix_mean_params = um.join_mean_mixture(components, cat_mean_params)
        mix_natural_params = um.to_natural(mix_mean_params)

    with hmog.obs_man as om:
        # Create observable normal with diagonal covariance
        obs_mean_params = om.join_mean_covariance(
            jnp.array([0.0, 0.0]),
            jnp.array([WIDTH, HEIGHT]),
        )
        obs_natural_params = om.to_natural(obs_mean_params)

        # NB: Multiplying the interaction parameters by the observable precision makes them scale more intuitively
        int_mat0 = jnp.array([1.0, 0.0])
        obs_prs = om.split_location_precision(obs_natural_params)[1]
        int_mat = hmog.int_man.from_dense(om.cov_man.to_dense(obs_prs) @ int_mat0)

    lkl_params = hmog.lkl_fun_man.join_coords(obs_natural_params, int_mat)

    hmog_natural_params = hmog.join_conjugated(lkl_params, mix_natural_params)

    return hmog, hmog_natural_params


### Training ###


def fit_hmog(
    key: Array,
    hmog: AnalyticHMoG,
    n_steps: int,
    sample: Array,
) -> tuple[
    Array,
    Array,
    Array,
]:
    """Train HMoG model using expectation maximization."""
    init_params = hmog.initialize_from_sample(key, sample, shape=2)

    def em_step(carry: Array, _: Any) -> tuple[Array, Array]:
        natural_params = carry
        ll = hmog.average_log_observable_density(natural_params, sample)
        next_natural_params = hmog.expectation_maximization(natural_params, sample)
        return next_natural_params, ll

    final_natural_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls.ravel(), init_params, final_natural_params


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
    gt_hmog, gt_hmog_natural_params = create_ground_truth_model()

    # Generate samples
    key_sample, key_train = jax.random.split(key)
    sample = gt_hmog.observable_sample(key_sample, gt_hmog_natural_params, sample_size)

    # Create HMoG model
    iso_hmog = analytic_hmog(2, Scale(), 1, 2)
    dia_hmog = analytic_hmog(2, Diagonal(), 1, 2)

    # Train model
    iso_lls, init_iso_params, final_iso_natural_params = fit_hmog(
        key_train, iso_hmog, n_steps, sample
    )
    dia_lls, init_dia_params, final_dia_natural_params = fit_hmog(
        key_train, dia_hmog, n_steps, sample
    )

    # Create evaluation grids
    x1 = jnp.linspace(-10, 10, n_points)
    x2 = jnp.linspace(-10, 10, n_points)
    x1s, x2s = jnp.meshgrid(x1, x2)
    grid_points = jnp.stack([x1s.ravel(), x2s.ravel()], axis=1)

    y = jnp.linspace(-6, 6, n_points)

    # Compute densities
    def compute_observable_density(
        model: AnalyticHMoG,
        natural_params: Array,
    ):
        return jax.vmap(model.observable_density, in_axes=(None, 0))(
            natural_params, grid_points
        ).reshape(x1s.shape)

    true_density = compute_observable_density(gt_hmog, gt_hmog_natural_params)
    init_iso_density = compute_observable_density(iso_hmog, init_iso_params)
    final_iso_density = compute_observable_density(iso_hmog, final_iso_natural_params)
    init_dia_density = compute_observable_density(dia_hmog, init_dia_params)
    final_dia_density = compute_observable_density(dia_hmog, final_dia_natural_params)

    # Get mixture densities
    def compute_mixture_density(
        model: AnalyticHMoG,
        natural_params: Array,
    ):
        mix_model = model.split_conjugated(natural_params)[1]
        return jax.vmap(model.pst_man.observable_density, in_axes=(None, 0))(
            mix_model, y[:, None]
        )

    true_mix = compute_mixture_density(gt_hmog, gt_hmog_natural_params)
    init_iso_mix = compute_mixture_density(iso_hmog, init_iso_params)
    final_iso_mix = compute_mixture_density(iso_hmog, final_iso_natural_params)
    init_dia_mix = compute_mixture_density(dia_hmog, init_dia_params)
    final_dia_mix = compute_mixture_density(dia_hmog, final_dia_natural_params)

    return HMoGResults(
        plot_range_x1=x1.tolist(),
        plot_range_x2=x2.tolist(),
        observations=sample.tolist(),
        log_likelihoods={"Isotropic": iso_lls.tolist(), "Diagonal()": dia_lls.tolist()},
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
