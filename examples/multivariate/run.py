"""Multivariate benchmark: analytic MVN fit + Adam-fitted Dirichlet on 20 samples each.

Ports ``examples/multivariate.hs`` from the Haskell reference implementation.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.scipy.stats import multivariate_normal

from goal.geometry import PositiveDefinite
from goal.models import Dirichlet, Normal

from ..shared import example_paths, jax_cli
from .types import DirichletPanel, MultivariateResults, NormalPanel

PLOT_POINTS = 100
N_SAMPLES = 20
N_EPOCHS = 500
LEARNING_RATE = 0.05


def fit_dirichlet_adam(
    model: Dirichlet,
    samples: Array,
    init_params: Array,
    n_epochs: int,
    learning_rate: float,
) -> tuple[Array, Array]:
    """Fit a Dirichlet via Adam on the average log density. Returns ``(final_params, log_likelihood_trace)``."""
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    def neg_log_lik(params: Array) -> Array:
        return -model.average_log_density(params, samples)

    def step(state: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(neg_log_lik)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), -loss

    (_, final_params), lls = jax.lax.scan(
        step, (opt_state, init_params), None, length=n_epochs
    )
    return final_params, lls


def dirichlet_density_grid(
    model: Dirichlet, params: Array, xrange: Array, yrange: Array
) -> Array:
    """Evaluate a 3-category Dirichlet density on the $(x, y)$ simplex projection with $z = 1 - x - y$."""
    xs, ys = jnp.meshgrid(xrange, yrange)
    zs = 1.0 - xs - ys
    points = jnp.stack([xs, ys, zs], axis=-1)  # (ny, nx, 3)
    flat = points.reshape(-1, 3)

    def safe_density(p: Array) -> Array:
        in_simplex = jnp.all(p > 1e-6)
        # Replace out-of-simplex points with a dummy valid point to avoid NaNs; mask below.
        safe_p = jnp.where(in_simplex, p, jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))
        d = model.density(params, safe_p)
        return jnp.where(in_simplex, d, 0.0)

    densities = jax.vmap(safe_density)(flat)
    return densities.reshape(xs.shape)


def compute_dirichlet_panel(key: Array) -> DirichletPanel:
    model = Dirichlet(n_categories=3)
    true_alpha = jnp.array([3.0, 7.0, 5.0])
    init_alpha = jnp.ones(3)

    samples = model.sample(key, true_alpha, N_SAMPLES)

    fitted_alpha, log_likelihood = fit_dirichlet_adam(
        model, samples, init_alpha, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE
    )

    xrange = jnp.linspace(0.0, 1.0, PLOT_POINTS)
    yrange = jnp.linspace(0.0, 1.0, PLOT_POINTS)
    true_density = dirichlet_density_grid(model, true_alpha, xrange, yrange)
    learned_density = dirichlet_density_grid(model, fitted_alpha, xrange, yrange)

    return DirichletPanel(
        samples=samples.tolist(),
        xrange=xrange.tolist(),
        yrange=yrange.tolist(),
        true_density=true_density.tolist(),
        learned_density=learned_density.tolist(),
        log_likelihood=log_likelihood.tolist(),
    )


def compute_normal_panel(key: Array) -> NormalPanel:
    model = Normal(2, PositiveDefinite())

    true_mean = jnp.array([1.0, -1.0])
    true_cov = jnp.array([[3.0, 1.0], [1.0, 2.0]])

    true_cov_coords = model.cov_man.from_matrix(true_cov)
    true_mean_params = model.join_mean_covariance(true_mean, true_cov_coords)
    true_natural = model.to_natural(true_mean_params)

    samples = model.sample(key, true_natural, N_SAMPLES)

    fit_mean_params = model.average_sufficient_statistic(samples)
    fit_natural = model.to_natural(fit_mean_params)
    source_mean, source_cov_coords = model.split_mean_covariance(fit_mean_params)
    source_cov = model.cov_man.to_matrix(source_cov_coords)

    xrange = jnp.linspace(-6.0, 10.0, PLOT_POINTS)
    yrange = jnp.linspace(-6.0, 8.0, PLOT_POINTS)
    xs, ys = jnp.meshgrid(xrange, yrange)
    flat = jnp.stack([xs.ravel(), ys.ravel()], axis=1)

    true_density = jax.vmap(multivariate_normal.pdf, in_axes=(0, None, None))(
        flat, true_mean, true_cov
    ).reshape(xs.shape)
    source_density = jax.vmap(multivariate_normal.pdf, in_axes=(0, None, None))(
        flat, source_mean, source_cov
    ).reshape(xs.shape)
    natural_density = jax.vmap(model.density, in_axes=(None, 0))(fit_natural, flat).reshape(
        xs.shape
    )

    return NormalPanel(
        samples=samples.tolist(),
        xrange=xrange.tolist(),
        yrange=yrange.tolist(),
        true_density=true_density.tolist(),
        source_fit_density=source_density.tolist(),
        natural_fit_density=natural_density.tolist(),
    )


def main() -> None:
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)
    key_d, key_n = jax.random.split(key)

    results = MultivariateResults(
        dirichlet=compute_dirichlet_panel(key_d),
        normal=compute_normal_panel(key_n),
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
