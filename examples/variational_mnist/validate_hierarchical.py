"""Correctness checks for the hierarchical variational-conjugation model.

Run with::

    uv run python -m examples.variational_mnist.validate_hierarchical

Checks, on tiny models where everything is finite and cheap:

1. **Decomposition** -- pointwise
   ``log p(x,y,z) - log q(y,z|x) == r_Y + r*_Z - r_inner_Z + c(x)``.
   This is the load-bearing identity: it guarantees ``elbo_at`` estimates a real
   ELBO and pins down every residual sign.
2. **ELBO gradient** -- the score-function surrogate's autodiff gradient matches
   a finite-difference gradient of ``E[c(x) + signal]`` w.r.t. a few parameters.
3. **Lower-bound sanity** -- ELBO <= IWAE(K) (the MLP-agnostic bound claim).
4. **Exact-N marginal estimator** (conv lower edge) -- ``marginal_elbo_at``
   equals the sampled-y assembly at shared z samples to machine precision
   (``r_Y == 0`` makes y irrelevant pointwise), and its pathwise autodiff
   gradient matches a fixed-key finite difference (the estimator is
   deterministic given the key, so no MC averaging is needed).
"""

import jax
import jax.numpy as jnp
from jax import Array

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from goal.geometry import Diagonal  # noqa: E402
from goal.models import Normal  # noqa: E402

from .hierarchical import (  # noqa: E402
    build_boltzmann_gaussian_hierarchy,
    build_conv_boltzmann_gaussian_hierarchy,
)


def _tiny_model(obs_kind: str = "normal"):
    """A small p(x,y,z): 3-unit chain-Boltzmann middle, 2-D Gaussian top."""
    if obs_kind == "normal":
        obs_man = Normal(2, Diagonal())
    else:
        from goal.models import Binomials

        obs_man = Binomials(2, 4)
    return build_boltzmann_gaussian_hierarchy(
        obs_man=obs_man,
        n_mid=3,
        top_dim=2,
        mid_kind="chain",
        mlp_hidden=(8,),
    )


def check_decomposition(obs_kind: str) -> float:
    model = _tiny_model(obs_kind)
    key = jax.random.PRNGKey(0)
    k_init, k_x = jax.random.split(key)
    params = model.initialize(k_init, location=0.0, shape=0.2)

    # Draw joint samples from the *model* so points are in-support and finite.
    joint = model.sample(k_x, params, 16)
    od, md = model.obs_man.data_dim, model.mid_man.data_dim
    xs, ys, zs = joint[:, :od], joint[:, od : od + md], joint[:, od + md :]

    def one(x: Array, y: Array, z: Array) -> tuple[Array, Array]:
        f = model.log_density_joint(params, x, y, z) - model.log_q(params, x, y, z)
        decomp = model.learning_signal(params, x, y, z) + model.conjugation_baseline(
            params, x
        )
        return f, decomp

    f, decomp = jax.vmap(one)(xs, ys, zs)
    max_err = float(jnp.max(jnp.abs(f - decomp)))
    print(f"[{obs_kind:8s}] decomposition max|f - (rY+rZ*-rZinner+c)| = {max_err:.2e}")
    return max_err


def check_elbo_gradient() -> float:
    """Estimator unbiasedness: E_key[autodiff grad] == d/dp E_key[elbo].

    The score-function estimator is unbiased but not deterministic in the key, so
    a single fixed-key finite difference would compare a pathwise gradient (the
    resampled Gaussian moving with the key) against a score-function gradient.
    Instead we average both sides over many keys with common random numbers.
    """
    model = _tiny_model("normal")
    key = jax.random.PRNGKey(1)
    k_init, k_elbo, k_keys = jax.random.split(key, 3)
    params = model.initialize(k_init, location=0.0, shape=0.2)
    x = model.sample(k_elbo, params, 1)[0, : model.obs_man.data_dim]

    n_keys, n_samples = 4096, 32
    keys = jax.random.split(k_keys, n_keys)

    def mean_elbo(p: Array) -> Array:
        return jnp.mean(jax.vmap(lambda k: model.elbo_at(k, p, x, n_samples))(keys))

    g = jax.grad(mean_elbo)(params)  # E_key[ score-function grad ]

    eps = 1e-4
    idxs = [0, model.top_var.dim, model.dim - 1]  # top, lower, recog blocks
    max_err = 0.0
    for i in idxs:
        e = jnp.zeros_like(params).at[i].set(eps)
        fd = (mean_elbo(params + e) - mean_elbo(params - e)) / (2 * eps)
        err = float(jnp.abs(fd - g[i]))
        max_err = max(max_err, err)
        print(f"  grad[{i:3d}]: autodiff {g[i]:+.5f}  fd {fd:+.5f}  |d| {err:.2e}")
    print(f"[gradient] max |E[autodiff] - fd E[elbo]| = {max_err:.2e}  (MC noise)")
    return max_err


def check_lower_bound() -> None:
    model = _tiny_model("normal")
    key = jax.random.PRNGKey(2)
    k_init, k_x, k_elbo, k_iwae = jax.random.split(key, 4)
    params = model.initialize(k_init, location=0.0, shape=0.3)
    x = model.sample(k_x, params, 1)[0, : model.obs_man.data_dim]

    elbo = float(model.elbo_at(k_elbo, params, x, n_samples=256))

    # IWAE(K): log-mean importance weight p(x,y,z)/q(y,z|x).
    ys, zs = model.sample_posterior(k_iwae, params, x, 2000)

    def log_w(y: Array, z: Array) -> Array:
        return model.log_density_joint(params, x, y, z) - model.log_q(params, x, y, z)

    lw = jax.vmap(log_w)(ys, zs)
    iwae = float(jax.scipy.special.logsumexp(lw) - jnp.log(lw.shape[0]))
    print(f"[bound]   ELBO {elbo:.4f}  <=  IWAE {iwae:.4f}  (gap {iwae - elbo:.4f})")


def check_marginal_elbo() -> tuple[float, float]:
    """Exact-N estimator: value identity at shared z's + pathwise grad vs FD."""
    model = build_conv_boltzmann_gaussian_hierarchy(
        Normal(36, Diagonal()),
        in_lattice=(3, 3),
        stride=(2, 2),
        kernel_shape=(3, 2),
        top_dim=3,
        prior_graph="chordal",
        mlp_hidden=(16,),
    )
    key = jax.random.PRNGKey(3)
    k_init, k_x, k_z, k_y = jax.random.split(key, 4)
    params = model.initialize(k_init, location=0.0, shape=0.3)
    x = 0.5 * jax.random.normal(k_x, (36,))

    # Value identity: with r_Y == 0 pointwise, the sampled-y assembly at the
    # SAME z samples must equal the marginal estimator exactly.
    n = 64
    zs = model.reparam_top_samples(k_z, params, x, n)
    marginal = model.marginal_elbo_at(k_z, params, x, n)

    def with_y(subkey: Array, z: Array) -> Array:
        y = model.mid_man.sample(subkey, model.posterior_mid_at(params, x, z), 1)[0]
        return model.learning_signal(params, x, y, z)

    sampled = model.conjugation_baseline(params, x) + jnp.mean(
        jax.vmap(with_y)(jax.random.split(k_y, n), zs)
    )
    v_err = float(jnp.abs(marginal - sampled))
    print(f"[marginal] |marginal - sampled-y assembly at shared z| = {v_err:.2e}")

    # Pathwise gradient vs fixed-key finite differences: the marginal estimator
    # is a deterministic function of (params, key), so FD is exact up to O(eps^2).
    def f(p: Array) -> Array:
        return model.marginal_elbo_at(k_z, p, x, 16)

    g = jax.grad(f)(params)
    eps = 1e-5
    g_err = 0.0
    for i in [0, model.top_var.dim, model.dim - 1]:  # top, lower, recog blocks
        e = jnp.zeros_like(params).at[i].set(eps)
        fd = (f(params + e) - f(params - e)) / (2 * eps)
        err = float(jnp.abs(fd - g[i]) / (1.0 + jnp.abs(fd)))
        g_err = max(g_err, err)
        print(f"  grad[{i:3d}]: pathwise {g[i]:+.6f}  fd {fd:+.6f}  rel|d| {err:.2e}")
    print(f"[marginal] max rel grad err = {g_err:.2e}  (deterministic; no MC noise)")
    return v_err, g_err


def main() -> None:
    print("=== Hierarchical variational-conjugation validation ===")
    e1 = check_decomposition("normal")
    e2 = check_decomposition("binomial")
    print()
    g = check_elbo_gradient()
    print()
    check_lower_bound()
    print()
    mv, mg = check_marginal_elbo()
    print()
    ok = e1 < 1e-6 and e2 < 1e-6 and g < 5e-2 and mv < 1e-9 and mg < 1e-4
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
