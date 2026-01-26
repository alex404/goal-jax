"""Numerically optimized univariate distributions (von Mises, COM-Poisson)."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.models import CoMPoisson, VonMises

from ..shared import example_paths, initialize_jax
from .types import CoMPoissonResults, DifferentiableUnivariateResults, VonMisesResults


def fit_von_mises(
    key: Array,
    model: VonMises,
    true_mu: float,
    true_kappa: float,
    n_samples: int = 100,
    n_steps: int = 100,
    learning_rate: float = 0.1,
) -> VonMisesResults:
    """Fit von Mises distribution via gradient descent."""
    true_params = model.join_mean_concentration(true_mu, true_kappa)
    sample = model.sample(key, true_params, n_samples)
    init_params = model.join_mean_concentration(0.0, 1.0)

    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    def loss_fn(params: Array) -> Array:
        return -model.average_log_density(params, sample)

    def step(
        state: tuple[Any, Any], _: Any
    ) -> tuple[tuple[Any, Any], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), loss

    (_, final_params), losses = jax.lax.scan(
        step, (opt_state, init_params), None, length=n_steps
    )

    eval_points = jnp.linspace(-jnp.pi, jnp.pi, 200)
    true_density = jax.vmap(model.density, in_axes=(None, 0))(true_params, eval_points)
    fitted_density = jax.vmap(model.density, in_axes=(None, 0))(final_params, eval_points)

    return VonMisesResults(
        samples=sample.ravel().tolist(),
        training_history=losses.tolist(),
        eval_points=eval_points.tolist(),
        true_density=true_density.tolist(),
        fitted_density=fitted_density.tolist(),
    )


def fit_com_poisson(
    key: Array,
    model: CoMPoisson,
    true_mu: float,
    true_nu: float,
    n_samples: int = 100,
    n_steps: int = 100,
    learning_rate: float = 0.1,
) -> CoMPoissonResults:
    """Fit COM-Poisson distribution via gradient descent."""
    true_params = model.join_mode_dispersion(
        jnp.atleast_1d(true_mu), jnp.atleast_1d(true_nu)
    )
    sample = model.sample(key, true_params, n_samples)
    init_params = model.join_mode_dispersion(jnp.atleast_1d(1.0), jnp.atleast_1d(1.0))

    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    def loss_fn(params: Array) -> Array:
        return -model.average_log_density(params, sample)

    def step(
        state: tuple[Any, Any], _: Any
    ) -> tuple[tuple[Any, Any], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), loss

    (_, final_params), losses = jax.lax.scan(
        step, (opt_state, init_params), None, length=n_steps
    )

    max_x = int(jnp.max(sample) + 5)
    eval_points = jnp.arange(0, max_x)
    true_pmf = jax.vmap(model.density, in_axes=(None, 0))(true_params, eval_points)
    fitted_pmf = jax.vmap(model.density, in_axes=(None, 0))(final_params, eval_points)

    return CoMPoissonResults(
        samples=sample.ravel().tolist(),
        training_history=losses.tolist(),
        eval_points=eval_points.tolist(),
        true_pmf=true_pmf.tolist(),
        fitted_pmf=fitted_pmf.tolist(),
    )


def main():
    initialize_jax()
    paths = example_paths(__file__)

    key = jax.random.PRNGKey(0)
    von_key, com_key = jax.random.split(key)

    von_results = fit_von_mises(
        key=von_key,
        model=VonMises(),
        true_mu=jnp.pi / 4,
        true_kappa=2.0,
        n_samples=20,
        n_steps=100,
        learning_rate=0.1,
    )

    com_results = fit_com_poisson(
        key=com_key,
        model=CoMPoisson(),
        true_mu=10.0,
        true_nu=2.0,
        n_samples=100,
        n_steps=1000,
        learning_rate=0.1,
    )

    paths.save_analysis(
        DifferentiableUnivariateResults(von_mises=von_results, com_poisson=com_results)
    )


if __name__ == "__main__":
    main()
