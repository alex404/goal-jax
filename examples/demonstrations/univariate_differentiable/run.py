"""Test script for numerically optimized univariate distributions."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Natural, Optimizer, OptState, Point
from goal.models import CoMPoisson, VonMises

from ...shared import initialize_jax, initialize_paths, save_results
from .types import CoMPoissonResults, DifferentiableUnivariateResults, VonMisesResults


def fit_von_mises(
    key: Array,
    von_man: VonMises,
    true_mu: float,
    true_kappa: float,
    n_samples: int = 100,
    n_steps: int = 100,
    learning_rate: float = 0.1,
) -> VonMisesResults:
    # Generate synthetic data
    true_params = von_man.join_mean_concentration(true_mu, true_kappa)
    sample = von_man.sample(key, true_params, n_samples)

    # Initialize optimization at reasonable starting point
    init_params = von_man.join_mean_concentration(0.0, 1.0)

    # Setup optimizer
    optimizer: Optimizer[Natural, VonMises] = Optimizer.adam(
        learning_rate=learning_rate
    )
    opt_state = optimizer.init(init_params)

    def cross_entropy_loss(params: Point[Natural, VonMises]) -> Array:
        return -von_man.average_log_density(params, sample)

    def grad_step(
        opt_state_and_params: tuple[OptState, Point[Natural, VonMises]], _: Any
    ) -> tuple[tuple[OptState, Point[Natural, VonMises]], Array]:
        # Compute loss and gradients
        opt_state, params = opt_state_and_params
        loss_val, grads = von_man.value_and_grad(cross_entropy_loss, params)

        # Get updates from optimizer
        opt_state, params = optimizer.update(opt_state, grads, params)

        return (opt_state, params), loss_val

    (_, final_params), ces = jax.lax.scan(
        grad_step, (opt_state, init_params), None, length=n_steps
    )
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


def fit_com_poisson(
    key: Array,
    com_man: CoMPoisson,
    true_mu: float,
    true_nu: float,
    n_samples: int = 100,
    n_steps: int = 100,
    learning_rate: float = 0.1,
) -> CoMPoissonResults:
    """Fit COM-Poisson distribution to synthetic data.

    Args:
        key: PRNG key
        com_man: COM-Poisson manifold
        true_mu: True mode parameter
        true_nu: True dispersion parameter
        n_samples: Number of synthetic samples
        n_steps: Number of optimization steps
        learning_rate: Learning rate for optimizer
        eval_points: Points at which to evaluate pmf

    Returns:
        Results dictionary containing samples, training history and pmf evaluations
    """
    # Generate synthetic data
    true_params = com_man.join_mode_dispersion(
        jnp.atleast_1d(true_mu), jnp.atleast_1d(true_nu)
    )
    sample = com_man.sample(key, true_params, n_samples)

    # Initialize optimization at reasonable starting point
    init_params = com_man.join_mode_dispersion(
        jnp.atleast_1d(1.0), jnp.atleast_1d(1.0)
    )  # Start at Poisson(1)

    # Setup optimizer
    optimizer: Optimizer[Natural, CoMPoisson] = Optimizer.adam(
        learning_rate=learning_rate
    )
    opt_state = optimizer.init(init_params)

    def cross_entropy_loss(params: Point[Natural, CoMPoisson]) -> Array:
        return -com_man.average_log_density(params, sample)

    def grad_step(
        opt_state_and_params: tuple[OptState, Point[Natural, CoMPoisson]], _: Any
    ) -> tuple[tuple[OptState, Point[Natural, CoMPoisson]], Array]:
        opt_state, params = opt_state_and_params
        loss_val, grads = com_man.value_and_grad(cross_entropy_loss, params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), loss_val

    (_, final_params), ces = jax.lax.scan(
        grad_step, (opt_state, init_params), None, length=n_steps
    )

    # Evaluate pmf
    max_x = int(jnp.max(sample) + 5)  # Add buffer
    eval_points = jnp.arange(0, max_x)

    true_pmf = jax.vmap(com_man.density, in_axes=(None, 0))(true_params, eval_points)
    fitted_pmf = jax.vmap(com_man.density, in_axes=(None, 0))(final_params, eval_points)

    return CoMPoissonResults(
        samples=sample.ravel().tolist(),
        training_history=ces.tolist(),
        eval_points=eval_points.tolist(),
        true_pmf=true_pmf.tolist(),
        fitted_pmf=fitted_pmf.tolist(),
    )


def main():
    """Run numerical optimization tests."""
    initialize_jax()
    paths = initialize_paths(__file__)

    key = jax.random.PRNGKey(0)
    von_man = VonMises()
    von_key, com_key = jax.random.split(key)

    # Test with synthetic data
    von_results = fit_von_mises(
        key=von_key,
        von_man=von_man,
        true_mu=jnp.pi / 4,  # 45 degrees
        true_kappa=2.0,  # moderate concentration
        n_samples=20,
        n_steps=100,
        learning_rate=0.1,
    )

    com_man = CoMPoisson()

    con_results = fit_com_poisson(
        key=com_key,
        com_man=com_man,
        true_mu=10.0,
        true_nu=0.5,
        n_samples=100,
        n_steps=200,
        learning_rate=0.1,
    )

    save_results(
        DifferentiableUnivariateResults(von_mises=von_results, com_poisson=con_results),
        paths,
    )


if __name__ == "__main__":
    main()
