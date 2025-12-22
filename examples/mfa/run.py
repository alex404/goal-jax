"""Mixture of Factor Analyzers (MFA) example."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from goal.geometry import Optimizer, OptState
from goal.models import FactorAnalysis, Normal
from goal.models.graphical.mixture import MixtureOfConjugated

from ..shared import example_paths, initialize_jax
from .types import MFAResults


def create_ground_truth(key: Array, sample_size: int) -> tuple[Array, Array, Array]:
    """Create ground truth samples from a known 2-component MFA."""
    obs_dim, lat_dim = 3, 2

    loadings_1 = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    means_1 = jnp.array([2.0, 1.0, 2.0])
    diags_1 = jnp.array([0.3, 0.3, 0.3])

    loadings_2 = jnp.array([[-1.0, 0.0], [0.5, -0.5], [0.0, -1.0]])
    means_2 = jnp.array([-2.0, -1.0, -2.0])
    diags_2 = jnp.array([0.3, 0.3, 0.3])

    mixing_probs = jnp.array([0.6, 0.4])

    fa = FactorAnalysis(obs_dim, lat_dim)
    fa1_params = fa.from_loadings(loadings_1, means_1, diags_1)
    fa2_params = fa.from_loadings(loadings_2, means_2, diags_2)

    key_assign, key_sample = jax.random.split(key)
    components = jax.random.categorical(key_assign, jnp.log(mixing_probs), shape=(sample_size,))
    keys = jax.random.split(key_sample, sample_size)

    def sample_one(k: Array, c: Array) -> Array:
        s0 = fa.observable_sample(k, fa1_params, 1)[0]
        s1 = fa.observable_sample(k, fa2_params, 1)[0]
        return jnp.where(c == 0, s0, s1)

    samples = jax.vmap(sample_one)(keys, components)
    return samples, components, mixing_probs


def compute_marginal_2d(
    loadings_1: Array, means_1: Array, diags_1: Array,
    loadings_2: Array, means_2: Array, diags_2: Array,
    mixing: Array, x_range: Array, dims: tuple[int, int]
) -> Array:
    """Compute 2D marginal density analytically."""
    grid_2d = jnp.stack(jnp.meshgrid(x_range, x_range), axis=-1).reshape(-1, 2)

    def point_density(pt: Array) -> Array:
        densities = []
        for loadings, means, diags in [(loadings_1, means_1, diags_1), (loadings_2, means_2, diags_2)]:
            mu = means[jnp.array(dims)]
            cov = (loadings @ loadings.T + jnp.diag(diags))[jnp.array(dims)][:, jnp.array(dims)]
            diff = pt - mu
            inv_cov = jnp.linalg.inv(cov)
            log_det = jnp.linalg.slogdet(cov)[1]
            log_dens = -0.5 * jnp.dot(diff, inv_cov @ diff) - 0.5 * log_det - jnp.log(2 * jnp.pi)
            densities.append(jnp.exp(log_dens))
        return mixing[0] * densities[0] + mixing[1] * densities[1]

    return jax.vmap(point_density)(grid_2d).reshape(len(x_range), len(x_range))


def compute_mfa_marginal_2d(
    mfa: MixtureOfConjugated[Normal, Normal], params: Array, x_range: Array, dims: tuple[int, int], marg_dim: int
) -> Array:
    """Compute 2D marginal by numerical integration."""
    grid_2d = jnp.stack(jnp.meshgrid(x_range, x_range), axis=-1).reshape(-1, 2)
    marg_range = jnp.linspace(-10, 10, 50)
    dmarg = marg_range[1] - marg_range[0]

    def point_marginal(pt: Array) -> Array:
        def density_at_marg(x_marg: Array) -> Array:
            pt_3d = jnp.zeros(3).at[dims[0]].set(pt[0]).at[dims[1]].set(pt[1]).at[marg_dim].set(x_marg)
            return mfa.observable_density(params, pt_3d)
        return jnp.sum(jax.vmap(density_at_marg)(marg_range)) * dmarg

    return jax.vmap(point_marginal)(grid_2d).reshape(len(x_range), len(x_range))


def fit_mfa(
    key: Array, mfa: MixtureOfConjugated[Normal, Normal], sample: Array, n_steps: int, learning_rate: float = 1e-2
) -> tuple[Array, Array, Array]:
    """Fit MFA via gradient descent."""
    init_params = mfa.initialize_from_sample(key, sample)
    optimizer: Optimizer[MixtureOfConjugated[Normal, Normal]] = Optimizer.adamw(man=mfa, learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    def loss_fn(params: Array) -> Array:
        return -mfa.average_log_observable_density(params, sample)

    def step(state: tuple[OptState, Array], _: Any) -> tuple[tuple[OptState, Array], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        opt_state, params = optimizer.update(opt_state, grads, params)
        return (opt_state, params), -loss

    (_, final_params), lls = jax.lax.scan(step, (opt_state, init_params), None, length=n_steps)
    return lls.ravel(), init_params, final_params


fit_mfa = jax.jit(fit_mfa, static_argnames=["mfa", "n_steps"])


def main():
    initialize_jax()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)
    key_data, key_train = jax.random.split(key)

    # Data
    samples, gt_components, mixing = create_ground_truth(key_data, 1000)

    # Model
    mfa = MixtureOfConjugated(n_categories=2, hrm=FactorAnalysis(obs_dim=3, lat_dim=2))
    lls, init_params, final_params = fit_mfa(key_train, mfa, samples, 20000)

    # Ground truth marginals (analytic)
    x_range = jnp.linspace(-5.0, 5.0, 50)
    loadings_1 = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    loadings_2 = jnp.array([[-1.0, 0.0], [0.5, -0.5], [0.0, -1.0]])
    means_1, means_2 = jnp.array([2.0, 1.0, 2.0]), jnp.array([-2.0, -1.0, -2.0])
    diags_1, diags_2 = jnp.array([0.3, 0.3, 0.3]), jnp.array([0.3, 0.3, 0.3])

    gt_x1x2 = compute_marginal_2d(loadings_1, means_1, diags_1, loadings_2, means_2, diags_2, mixing, x_range, (0, 1))
    gt_x1x3 = compute_marginal_2d(loadings_1, means_1, diags_1, loadings_2, means_2, diags_2, mixing, x_range, (0, 2))
    gt_x2x3 = compute_marginal_2d(loadings_1, means_1, diags_1, loadings_2, means_2, diags_2, mixing, x_range, (1, 2))

    # Fitted marginals (numerical)
    init_x1x2 = compute_mfa_marginal_2d(mfa, init_params, x_range, (0, 1), 2)
    init_x1x3 = compute_mfa_marginal_2d(mfa, init_params, x_range, (0, 2), 1)
    init_x2x3 = compute_mfa_marginal_2d(mfa, init_params, x_range, (1, 2), 0)

    final_x1x2 = compute_mfa_marginal_2d(mfa, final_params, x_range, (0, 1), 2)
    final_x1x3 = compute_mfa_marginal_2d(mfa, final_params, x_range, (0, 2), 1)
    final_x2x3 = compute_mfa_marginal_2d(mfa, final_params, x_range, (1, 2), 0)

    # Assignments
    gt_assign = jnp.eye(2)[gt_components]
    final_assign = jax.vmap(mfa.posterior_soft_assignments, in_axes=(None, 0))(final_params, samples)

    results = MFAResults(
        observations=samples.tolist(),
        ground_truth_components=gt_components.tolist(),
        plot_range=x_range.tolist(),
        log_likelihoods=lls.tolist(),
        ground_truth_ll=float(mfa.average_log_observable_density(final_params, samples)),
        ground_truth_density_x1x2=gt_x1x2.tolist(),
        ground_truth_density_x1x3=gt_x1x3.tolist(),
        ground_truth_density_x2x3=gt_x2x3.tolist(),
        initial_density_x1x2=init_x1x2.tolist(),
        initial_density_x1x3=init_x1x3.tolist(),
        initial_density_x2x3=init_x2x3.tolist(),
        final_density_x1x2=final_x1x2.tolist(),
        final_density_x1x3=final_x1x3.tolist(),
        final_density_x2x3=final_x2x3.tolist(),
        ground_truth_assignments=gt_assign.tolist(),
        final_assignments=final_assign.tolist(),
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
