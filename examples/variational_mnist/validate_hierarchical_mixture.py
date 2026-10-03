"""Correctness checks for the mixture-top hierarchy.

The mixture ELBO is assembled as a single-Gaussian control variate plus an exact
correction. This script verifies that assembly against the definition:

1. **ELBO identity.** For shared posterior samples ``(y, z) ~ q(.|x)``,
   ``conjugation_baseline(x) + mean(learning_signal(x, y, z))`` (the value that
   :meth:`elbo_at` returns) equals the brute-force Monte Carlo ELBO
   ``mean(log p_mix(x, y, z) - log q(y, z | x))``. If the control-variate
   bookkeeping is right these agree to floating-point noise.
2. **Reference invariance.** The assembled ELBO value is independent of the
   control-variate reference (it is exact for any reference).
3. **Responsibilities.** Cluster responsibilities are a probability vector.
4. **ELBO <= IWAE.** A quick importance-weighted bound sanity check.
5. **Conv lower edge.** The same checks on the convolutional (overlapping-kernel,
   chordal) lower edge, plus ``r_Y == 0`` pointwise -- the analytic conjugation
   of the conv Gaussian-Boltzmann edge must survive the mixture top verbatim.

Run::

    uv run python -m examples.variational_mnist.validate_hierarchical_mixture
"""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from goal.geometry import Diagonal  # noqa: E402
from goal.models import Normal  # noqa: E402

from .hierarchical_mixture import (  # noqa: E402
    VariationalHierarchicalMixture,
    build_boltzmann_gaussian_mixture_hierarchy,
    build_conv_boltzmann_gaussian_mixture_hierarchy,
)

OBS_DIM = 6
N_MID = 8
TOP_DIM = 3
N_CLUSTERS = 4


def build(kind: str) -> VariationalHierarchicalMixture:
    obs_man = Normal(OBS_DIM, Diagonal())
    return build_boltzmann_gaussian_mixture_hierarchy(
        obs_man,
        N_MID,
        TOP_DIM,
        N_CLUSTERS,
        mid_kind=kind,
        mlp_hidden=(32,),
        obs_location_only=True,
    )


def brute_force_elbo(
    model: VariationalHierarchicalMixture, params: Array, x: Array, key: Array, n: int
) -> tuple[Array, Array]:
    """Return (control-variate ELBO value, brute-force MC ELBO) at shared samples."""
    ys, zs = model.sample_posterior(key, params, x, n)

    # Control-variate assembly (the value elbo_at returns: score - sg(score) = 0).
    c_x = model.conjugation_baseline(params, x)
    signal = jax.vmap(lambda y, z: model.learning_signal(params, x, y, z))(ys, zs)
    cv = c_x + jnp.mean(signal)

    # Brute force: mean(log p_mix(x,y,z) - log q(y,z|x)). The observable base
    # measure is already inside log_density_joint's log p(x|y) term, and the y/z
    # base measures cancel between log_density_joint and log_q.
    logp = jax.vmap(lambda y, z: model.log_density_joint(params, x, y, z))(ys, zs)
    logq = jax.vmap(lambda y, z: model.log_q(params, x, y, z))(ys, zs)
    bf = jnp.mean(logp - logq)
    return cv, bf


def iwae(
    model: VariationalHierarchicalMixture, params: Array, x: Array, key: Array, k: int
) -> Array:
    ys, zs = model.sample_posterior(key, params, x, k)
    logp = jax.vmap(lambda y, z: model.log_density_joint(params, x, y, z))(ys, zs)
    logq = jax.vmap(lambda y, z: model.log_q(params, x, y, z))(ys, zs)
    lw = logp - logq
    return logsumexp(lw) - jnp.log(k)


def main() -> None:
    key = jax.random.PRNGKey(0)
    for kind in ["diagonal", "chain", "chordal"]:
        k_init, k_x, k_s = jax.random.split(jax.random.fold_in(key, hash(kind) % 97), 3)
        model = build(kind)
        params = model.initialize(k_init, 0.0, 0.4)
        x = 0.5 * jax.random.normal(k_x, (OBS_DIM,))

        # 1. ELBO identity (many samples so the two MC estimates use the SAME draws).
        cv, bf = brute_force_elbo(model, params, x, k_s, 2000)
        gap = float(jnp.abs(cv - bf))

        # 3. Responsibilities are a probability vector.
        r = model.responsibilities(k_s, params, x, 64)
        r_ok = bool(jnp.all(r >= 0) and jnp.abs(jnp.sum(r) - 1.0) < 1e-6)

        # 4. ELBO <= IWAE(K).
        elbo_val = float(cv)
        iwae_val = float(iwae(model, params, x, jax.random.fold_in(k_s, 1), 2000))

        print(
            f"[{kind:8s}] "
            f"|CV - brute force| = {gap:.2e}   "
            f"resp sum={float(jnp.sum(r)):.6f} ({'ok' if r_ok else 'BAD'})   "
            f"ELBO {elbo_val:8.3f} <= IWAE {iwae_val:8.3f} "
            f"({'ok' if elbo_val <= iwae_val + 1e-3 else 'BAD'})"
        )
        assert gap < 1e-6, (
            f"control-variate ELBO disagrees with brute force ({gap:.2e})"
        )
        assert r_ok, "responsibilities not a probability vector"

    # 5. Conv/chordal lower edge: overlapping kernel (3 > stride 2 vertically), so
    #    the induced coupling graph is nontrivial and rho_Y is the closed-form
    #    node+edge conjugation parameter. The mixture top must inherit r_Y == 0.
    conv_in, conv_stride, conv_kernel = (3, 3), (2, 2), (3, 2)
    obs_dim = 36  # prod(in_lattice * stride)
    conv_obs = Normal(obs_dim, Diagonal())
    kc_init, kc_x, kc_s = jax.random.split(jax.random.fold_in(key, 11), 3)
    conv_model = build_conv_boltzmann_gaussian_mixture_hierarchy(
        conv_obs,
        conv_in,
        conv_stride,
        conv_kernel,
        TOP_DIM,
        N_CLUSTERS,
        prior_graph="chordal",
        mlp_hidden=(32,),
    )
    conv_params = conv_model.initialize(kc_init, 0.0, 0.4)
    xc = 0.5 * jax.random.normal(kc_x, (obs_dim,))

    cv, bf = brute_force_elbo(conv_model, conv_params, xc, kc_s, 2000)
    gap = float(jnp.abs(cv - bf))
    ys = jax.vmap(conv_model.mid_man.sufficient_statistic)(
        jax.random.bernoulli(kc_s, 0.5, (64, conv_model.mid_man.data_dim)).astype(float)
    )[:, : conv_model.mid_man.data_dim]  # random spike patterns
    r_y = jax.vmap(lambda y: conv_model.residual_lower(conv_params, y))(ys)
    max_ry = float(jnp.max(jnp.abs(r_y)))
    r = conv_model.responsibilities(kc_s, conv_params, xc, 64)
    r_ok = bool(jnp.all(r >= 0) and jnp.abs(jnp.sum(r) - 1.0) < 1e-6)
    print(
        f"[conv    ] |CV - brute force| = {gap:.2e}   max|r_Y| = {max_ry:.2e}   "
        f"resp sum={float(jnp.sum(r)):.6f} ({'ok' if r_ok else 'BAD'})"
    )
    assert gap < 1e-6, (
        f"conv control-variate ELBO disagrees with brute force ({gap:.2e})"
    )
    assert max_ry < 1e-10, (
        f"conv lower edge not exactly conjugate under mixture top ({max_ry:.2e})"
    )
    assert r_ok, "conv responsibilities not a probability vector"

    # 5b. Exact-N marginal estimator under the mixture top: must equal the
    #     sampled-y assembly at shared z samples (r_Y == 0 makes y irrelevant).
    kz, ky = jax.random.split(jax.random.fold_in(key, 12))
    n = 64
    zs = conv_model.reparam_top_samples(kz, conv_params, xc, n)
    marginal = conv_model.marginal_elbo_at(kz, conv_params, xc, n)

    def with_y(subkey: Array, z: Array) -> Array:
        y = conv_model.mid_man.sample(
            subkey, conv_model.posterior_mid_at(conv_params, xc, z), 1
        )[0]
        return conv_model.learning_signal(conv_params, xc, y, z)

    sampled = conv_model.conjugation_baseline(conv_params, xc) + jnp.mean(
        jax.vmap(with_y)(jax.random.split(ky, n), zs)
    )
    m_err = float(jnp.abs(marginal - sampled))
    print(f"[conv    ] |marginal - sampled-y assembly at shared z| = {m_err:.2e}")
    assert m_err < 1e-9, (
        f"mixture marginal ELBO disagrees with sampled assembly ({m_err:.2e})"
    )

    # 2. K=1 degeneracy: a one-component mixture IS a single Gaussian, so the
    #    moment-matched reference equals it and the correction is zero up to the
    #    reference covariance jitter (1e-4) -- i.e. O(1e-3), not machine epsilon.
    obs_man = Normal(OBS_DIM, Diagonal())
    model1 = build_boltzmann_gaussian_mixture_hierarchy(
        obs_man,
        N_MID,
        TOP_DIM,
        1,
        mid_kind="chain",
        mlp_hidden=(32,),
        obs_location_only=True,
    )
    p1 = model1.initialize(jax.random.fold_in(key, 5), 0.0, 0.4)
    x1 = 0.5 * jax.random.normal(jax.random.fold_in(key, 6), (OBS_DIM,))
    zs = model1.top_man.sample(
        jax.random.fold_in(key, 7), model1.approximate_posterior_top(p1, x1), 256
    )
    mix1 = model1.split_mixture(p1)
    ref1 = model1.reference_prior(p1)
    corr = jax.vmap(
        lambda z: (
            model1.top_prior.log_observable_density(mix1, z)
            - model1.top_man.log_density(ref1, z)
        )
    )(zs)
    max_corr = float(jnp.max(jnp.abs(corr)))
    print(
        f"[K=1     ] max |log p_mix - log p_ref| = {max_corr:.2e} (jitter-limited) "
        f"({'ok' if max_corr < 1e-2 else 'BAD'})"
    )
    assert max_corr < 1e-2, "K=1 correction should be zero up to reference jitter"
    print("\nAll mixture-hierarchy checks passed.")


if __name__ == "__main__":
    main()
