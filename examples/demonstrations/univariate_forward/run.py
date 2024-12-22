"""Test script for numerically optimized univariate distributions."""

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import Natural, Point
from goal.models import VonMises

from ...shared import initialize_jax, initialize_paths, save_results
from .types import ForwardUnivariateResults, VonMisesOptimizationStep, VonMisesResults


@jax.jit
def negative_log_likelihood(
    params: Array,
    manifold: VonMises,
    data: Array,
) -> Array:
    """Compute negative log likelihood of data under von Mises distribution."""
    p = Point[Natural, VonMises](params)
    return -manifold.average_log_density(p, data)


def fit_von_mises(
    key: Array,
    manifold: VonMises,
    true_mu: float,
    true_kappa: float,
    n_samples: int = 1000,
    n_steps: int = 100,
    learning_rate: float = 0.1,
    eval_points: Array | None = None,
) -> VonMisesResults:
    # Generate synthetic data
    true_params = manifold.from_mean_concentration(
        jnp.atleast_1d(true_mu), jnp.atleast_1d(true_kappa)
    )
    samples = manifold.sample(key, true_params, n_samples)

    # Initialize optimization at reasonable starting point
    init_params = manifold.from_mean_concentration(
        jnp.atleast_1d(0.0), jnp.atleast_1d(1.0)
    ).params

    # Setup optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    # Loss function (closed over data)
    loss_fn = lambda p: negative_log_likelihood(p, manifold, samples)

    # Run optimization
    optimization_path = []
    params = init_params

    for _ in range(n_steps):
        # Compute loss and gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(params)

        # Get updates from optimizer
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Record step information
        current_point = Point[Natural, VonMises](params)
        mean_params = manifold.to_mean(current_point).params

        step_info = VonMisesOptimizationStep(
            natural_params=params.tolist(),
            mean_params=mean_params.tolist(),
            target_means=manifold.average_sufficient_statistic(samples).params.tolist(),
            gradient=grads.tolist(),
            loss=float(loss_val),
        )
        optimization_path.append(step_info)

    # Evaluate densities
    if eval_points is None:
        eval_points = jnp.linspace(-jnp.pi, jnp.pi, 200)

    true_density = jax.vmap(manifold.density, in_axes=(None, 0))(
        true_params, eval_points
    )
    final_point = Point[Natural, VonMises](params)
    fitted_density = jax.vmap(manifold.density, in_axes=(None, 0))(
        final_point, eval_points
    )

    return VonMisesResults(
        true_mu=float(true_mu),
        true_kappa=float(true_kappa),
        samples=samples.ravel().tolist(),
        optimization_path=optimization_path,
        final_natural_params=params.tolist(),
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
        manifold=manifold,
        true_mu=jnp.pi / 4,  # 45 degrees
        true_kappa=2.0,  # moderate concentration
        n_samples=1000,
        n_steps=100,
        learning_rate=0.1,
    )

    save_results(ForwardUnivariateResults(von_mises=results), paths)


if __name__ == "__main__":
    main()
