"""Test script for numerically optimized univariate distributions."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Natural, Optimizer, OptState, Point, reduce_dual
from goal.models import VonMises

from ...shared import initialize_jax, initialize_paths, save_results
from .types import (
    DifferentiableUnivariateResults,
    VonMisesResults,
)


def fit_von_mises(
    key: Array,
    von_man: VonMises,
    true_mu: float,
    true_kappa: float,
    n_samples: int = 100,
    n_steps: int = 100,
    learning_rate: float = 0.1,
    eval_points: Array | None = None,
) -> VonMisesResults:
    # Generate synthetic data
    true_params = von_man.from_mean_concentration(true_mu, true_kappa)
    sample = von_man.sample(key, true_params, n_samples)

    # Initialize optimization at reasonable starting point
    init_params = von_man.from_mean_concentration(0.0, 1.0)

    # Setup optimizer
    optimizer: Optimizer[VonMises] = Optimizer.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    def cross_entropy_loss(params: Point[Natural, VonMises]) -> Array:
        return -von_man.average_log_density(params, sample)

    def adam_step(
        opt_state_and_params: tuple[OptState, Point[Natural, VonMises]], _: Any
    ) -> tuple[tuple[OptState, Point[Natural, VonMises]], Array]:
        # Compute loss and gradients
        opt_state, params = opt_state_and_params
        loss_val, grads = von_man.value_and_grad(cross_entropy_loss, params)

        # Get updates from optimizer
        opt_state, params = optimizer.update(opt_state, reduce_dual(grads), params)

        return (opt_state, params), loss_val

    (_, final_params), ces = jax.lax.scan(
        adam_step, (opt_state, init_params), None, length=n_steps
    )
    # Evaluate densities
    if eval_points is None:
        eval_points = jnp.linspace(-jnp.pi, jnp.pi, 200)

    true_density = jax.vmap(von_man.density, in_axes=(None, 0))(
        true_params, eval_points
    )
    fitted_density = jax.vmap(von_man.density, in_axes=(None, 0))(
        final_params, eval_points
    )
    return VonMisesResults(
        samples=sample.ravel().tolist(),
        training_history=ces.tolist(),
        eval_points=eval_points.tolist(),
        true_density=true_density.tolist(),
        fitted_density=fitted_density.tolist(),
    )


def main():
    """Run numerical optimization tests."""
    initialize_jax()
    paths = initialize_paths(__file__)

    key = jax.random.PRNGKey(0)
    manifold = VonMises()

    # Test with synthetic data
    results = fit_von_mises(
        key=key,
        von_man=manifold,
        true_mu=jnp.pi / 4,  # 45 degrees
        true_kappa=2.0,  # moderate concentration
        n_samples=1000,
        n_steps=100,
        learning_rate=0.1,
    )

    save_results(DifferentiableUnivariateResults(von_mises=results), paths)


if __name__ == "__main__":
    main()
