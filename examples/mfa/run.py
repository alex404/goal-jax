"""Mixture of Factor Analyzers (MFA) example."""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import Diagonal, Differentiable
from goal.models import DiagonalNormal, MixtureOfFactorAnalyzers, factor_analysis
from goal.models.graphical.mixture import CompleteMixtureOfConjugated
from goal.models.harmonium.lgm import NormalLGM

from ..shared import example_paths, jax_cli
from .types import MFAResults


def create_ground_truth(
    key: Array, sample_size: int
) -> tuple[
    Array,
    Array,
    Array,
    list[tuple[Array, Array, Array]],
]:
    """Create ground truth samples from a known 3-component MFA.

    Returns samples, component assignments, mixing probabilities, and
    the GT component parameters (loadings, means, diags) for each component.
    """
    obs_dim, lat_dim = 3, 2

    gt_components = [
        (
            jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            jnp.array([2.0, 1.0, 2.0]),
            jnp.array([0.3, 0.3, 0.3]),
        ),
        (
            jnp.array([[-1.0, 0.0], [0.5, -0.5], [0.0, -1.0]]),
            jnp.array([-2.0, -1.0, -2.0]),
            jnp.array([0.3, 0.3, 0.3]),
        ),
        (
            jnp.array([[0.0, 1.0], [-0.5, 0.5], [1.0, 0.0]]),
            jnp.array([0.0, 3.0, -3.0]),
            jnp.array([0.3, 0.3, 0.3]),
        ),
    ]
    mixing_probs = jnp.array([0.4, 0.35, 0.25])

    fa = factor_analysis(obs_dim, lat_dim)
    fa_params = [
        fa.initialize_from_loadings(loadings, means, diags)
        for loadings, means, diags in gt_components
    ]

    key_assign, key_sample = jax.random.split(key)
    components = jax.random.categorical(
        key_assign, jnp.log(mixing_probs), shape=(sample_size,)
    )
    keys = jax.random.split(key_sample, sample_size)

    def sample_one(k: Array, c: Array) -> Array:
        s0 = fa.observable_sample(k, fa_params[0], 1)[0]
        s1 = fa.observable_sample(k, fa_params[1], 1)[0]
        s2 = fa.observable_sample(k, fa_params[2], 1)[0]
        return jnp.where(c == 0, s0, jnp.where(c == 1, s1, s2))

    samples = jax.vmap(sample_one)(keys, components)
    return samples, components, mixing_probs, gt_components


def compute_marginal_2d(
    components: list[tuple[Array, Array, Array]],
    mixing: Array,
    x_range: Array,
    dims: tuple[int, int],
) -> Array:
    """Compute 2D marginal density analytically."""
    grid_2d = jnp.stack(jnp.meshgrid(x_range, x_range), axis=-1).reshape(-1, 2)

    def point_density(pt: Array) -> Array:
        total = 0.0
        for i, (loadings, means, diags) in enumerate(components):
            mu = means[jnp.array(dims)]
            cov = (loadings @ loadings.T + jnp.diag(diags))[jnp.array(dims)][
                :, jnp.array(dims)
            ]
            diff = pt - mu
            inv_cov = jnp.linalg.inv(cov)
            log_det = jnp.linalg.slogdet(cov)[1]
            log_dens = (
                -0.5 * jnp.dot(diff, inv_cov @ diff)
                - 0.5 * log_det
                - jnp.log(2 * jnp.pi)
            )
            total = total + mixing[i] * jnp.exp(log_dens)
        return total  # pyright: ignore[reportReturnType]

    return jax.vmap(point_density)(grid_2d).reshape(len(x_range), len(x_range))


def compute_mfa_marginal_2d[Pst: Differentiable, Prr: Differentiable](
    mfa: CompleteMixtureOfConjugated[DiagonalNormal, Pst, Prr],
    params: Array,
    x_range: Array,
    dims: tuple[int, int],
    marg_dim: int,
) -> Array:
    """Compute 2D marginal by numerical integration."""
    grid_2d = jnp.stack(jnp.meshgrid(x_range, x_range), axis=-1).reshape(-1, 2)
    marg_range = jnp.linspace(-10, 10, 30)
    dmarg = marg_range[1] - marg_range[0]

    def point_marginal(pt: Array) -> Array:
        def density_at_marg(x_marg: Array) -> Array:
            pt_3d = (
                jnp.zeros(3)
                .at[dims[0]]
                .set(pt[0])
                .at[dims[1]]
                .set(pt[1])
                .at[marg_dim]
                .set(x_marg)
            )
            return mfa.observable_density(params, pt_3d)

        return jnp.sum(jax.vmap(density_at_marg)(marg_range)) * dmarg

    return jax.vmap(point_marginal)(grid_2d).reshape(len(x_range), len(x_range))


def fit_mfa_em(
    key: Array,
    mfa: MixtureOfFactorAnalyzers,
    sample: Array,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    """Fit MFA via closed-form expectation maximization with latent whitening."""
    init_params = mfa.initialize_from_sample(key, sample)

    def em_step(params: Array, _: Any) -> tuple[Array, Array]:
        ll = mfa.average_log_observable_density(params, sample)
        return mfa.expectation_maximization(params, sample), ll

    final_params, lls = jax.lax.scan(em_step, init_params, None, length=n_steps)
    return lls, init_params, final_params


def fit_mfa_gd[Pst: Differentiable, Prr: Differentiable](
    key: Array,
    mfa: CompleteMixtureOfConjugated[DiagonalNormal, Pst, Prr],
    sample: Array,
    n_steps: int,
    learning_rate: float = 1e-3,
) -> tuple[Array, Array, Array]:
    """Fit MFA via gradient descent."""
    init_params = mfa.initialize_from_sample(key, sample)
    optimizer = optax.adamw(learning_rate=learning_rate)
    opt_state = optimizer.init(init_params)

    def loss_fn(params: Array) -> Array:
        return -mfa.average_log_observable_density(params, sample)

    def train_step(
        state: tuple[Any, Any], _: Any
    ) -> tuple[tuple[Any, Any], Array]:
        opt_state, params = state
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (opt_state, params), -loss

    (_, final_params), lls = jax.lax.scan(
        train_step, (opt_state, init_params), None, length=n_steps
    )
    return lls, init_params, final_params


def main():
    jax_cli()
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)
    key_data, key_fa, key_diag = jax.random.split(key, 3)

    # Data
    samples, gt_assignments, mixing, gt_components = create_ground_truth(key_data, 1000)

    # Model 1: Mixture of Factor Analyzers — closed-form EM with latent whitening
    mfa_fa = MixtureOfFactorAnalyzers(
        n_categories=3, bas_hrm=factor_analysis(obs_dim=3, lat_dim=2)
    )
    print("Training FA model (EM)...")
    fa_lls, fa_init, fa_final = fit_mfa_em(key_fa, mfa_fa, samples, 500)

    # Model 2: NormalLGM with diagonal latent covariance — gradient descent
    diag_lgm = NormalLGM(obs_dim=3, obs_rep=Diagonal(), lat_dim=2, pst_rep=Diagonal())
    mfa_diag = CompleteMixtureOfConjugated(n_categories=3, bas_hrm=diag_lgm)
    print("\nTraining Diag model (GD)...")
    diag_lls, diag_init, diag_final = fit_mfa_gd(
        key_diag, mfa_diag, samples, 2000
    )

    # Ground truth marginals (analytic)
    x_range = jnp.linspace(-5.0, 5.0, 50)

    # Compute all marginals
    def compute_all_marginals(mfa: Any, params: Array) -> tuple[Array, Array, Array]:
        return (
            compute_mfa_marginal_2d(mfa, params, x_range, (0, 1), 2),
            compute_mfa_marginal_2d(mfa, params, x_range, (0, 2), 1),
            compute_mfa_marginal_2d(mfa, params, x_range, (1, 2), 0),
        )

    # Ground truth
    gt_x1x2 = compute_marginal_2d(gt_components, mixing, x_range, (0, 1))
    gt_x1x3 = compute_marginal_2d(gt_components, mixing, x_range, (0, 2))
    gt_x2x3 = compute_marginal_2d(gt_components, mixing, x_range, (1, 2))

    # FA model marginals
    fa_init_x1x2, fa_init_x1x3, fa_init_x2x3 = compute_all_marginals(mfa_fa, fa_init)
    fa_final_x1x2, fa_final_x1x3, fa_final_x2x3 = compute_all_marginals(
        mfa_fa, fa_final
    )

    # Diag model marginals
    diag_init_x1x2, diag_init_x1x3, diag_init_x2x3 = compute_all_marginals(
        mfa_diag, diag_init
    )
    diag_final_x1x2, diag_final_x1x3, diag_final_x2x3 = compute_all_marginals(
        mfa_diag, diag_final
    )

    # Ground truth log-likelihood (construct GT model and evaluate)
    fa = factor_analysis(obs_dim=3, lat_dim=2)
    fa_params_list = [
        fa.initialize_from_loadings(loadings, means, diags)
        for loadings, means, diags in gt_components
    ]

    # Build ground truth mixture model
    gt_mfa = MixtureOfFactorAnalyzers(n_categories=3, bas_hrm=fa)
    gt_comp_params = jnp.concatenate(fa_params_list)
    gt_cat_params = gt_mfa.pst_man.lat_man.to_natural(
        gt_mfa.pst_man.lat_man.from_probs(mixing)
    )
    gt_params = gt_mfa.mix_man.join_natural_mixture(gt_comp_params, gt_cat_params)
    gt_params = gt_mfa.from_mixture_coords(gt_params)
    gt_ll = float(gt_mfa.average_log_observable_density(gt_params, samples))
    print(f"\nGround truth LL: {gt_ll:.4f}")

    # Assignments
    gt_assign = jnp.eye(3)[gt_assignments]
    fa_assign = jax.vmap(mfa_fa.posterior_soft_assignments, in_axes=(None, 0))(
        fa_final, samples
    )
    diag_assign = jax.vmap(mfa_diag.posterior_soft_assignments, in_axes=(None, 0))(
        diag_final, samples
    )

    results = MFAResults(
        observations=samples.tolist(),
        ground_truth_components=gt_assignments.tolist(),
        plot_range=x_range.tolist(),
        log_likelihoods={
            "FA": fa_lls.tolist(),
            "Diag": diag_lls.tolist(),
        },
        ground_truth_ll=gt_ll,
        density_x1x2={
            "Ground Truth": gt_x1x2.tolist(),
            "FA Initial": fa_init_x1x2.tolist(),
            "FA Final": fa_final_x1x2.tolist(),
            "Diag Initial": diag_init_x1x2.tolist(),
            "Diag Final": diag_final_x1x2.tolist(),
        },
        density_x1x3={
            "Ground Truth": gt_x1x3.tolist(),
            "FA Initial": fa_init_x1x3.tolist(),
            "FA Final": fa_final_x1x3.tolist(),
            "Diag Initial": diag_init_x1x3.tolist(),
            "Diag Final": diag_final_x1x3.tolist(),
        },
        density_x2x3={
            "Ground Truth": gt_x2x3.tolist(),
            "FA Initial": fa_init_x2x3.tolist(),
            "FA Final": fa_final_x2x3.tolist(),
            "Diag Initial": diag_init_x2x3.tolist(),
            "Diag Final": diag_final_x2x3.tolist(),
        },
        ground_truth_assignments=gt_assign.tolist(),
        final_assignments={
            "FA": fa_assign.tolist(),
            "Diag": diag_assign.tolist(),
        },
    )
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
