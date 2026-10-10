"""Check the canonical circuit against brute-force integration, at 4 neurons and a one-dimensional $x$.

The brute force enumerates the neurons and integrates $x$ and $z$ on grids. It uses only sufficient
statistics and the circuit's joint densities, not the closed forms the diagnostics rely on. For generative
couplings on no pairs, a chain and all pairs, checked:

1. the likelihood of the $X - N$ harmonium equals that of :class:`~goal.models.BoltzmannLGM` at the same
   parameters, at every state of the neurons;
2. the residual of the $X - N$ level is zero, the generative model over the neurons has no couplings off
   $E$, and the exact sampler of the neurons matches the enumerated means;
3. the exact $\\log p(x)$ against the unnormalized joint of the graphical harmonium on grids;
4. $\\log \\tilde p(x)$ and the ELBO by quadrature against the circuit's own joint and recognition densities
   on a grid, and the divergence of the recognition model from the exact posterior likewise;
5. the mean of the ELBO estimator over many keys, and of its gradient, against the quadrature ELBO.

Parameters are the initialization moved by noise (smaller on the map, which keeps the precision of $z$
positive), with $\\rho_Z$ a map, so that every block is nonzero.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from goal.geometry import PositiveDefinite
from goal.models import BoltzmannLGM

from ..shared import jax_cli
from .model import CanonicalCircuit, Couplings, canonical_circuit

jax.config.update("jax_enable_x64", True)

N_NEURONS, N_NODES = 4, 40
ZS = jnp.linspace(-12.0, 12.0, 1201)[:, None]
XS = jnp.linspace(-8.0, 8.0, 801)[:, None]


def report(name: str, err: float, tol: float) -> bool:
    ok = err < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {err:.2e} (tol {tol:.0e})")
    return ok


def grid_logsumexp(vals: Array, grid: Array) -> Array:
    """$\\log \\int e^{f}$ by the trapezoid rule, on the last axis."""
    dx = grid[1, 0] - grid[0, 0]
    w = jnp.full(grid.shape[0], dx).at[0].set(dx / 2).at[-1].set(dx / 2)
    return logsumexp(vals + jnp.log(w), axis=-1)


def harmonium_log_joint(circuit: CanonicalCircuit, params: Array) -> Any:
    """Unnormalized $\\log p(x, n, z)$ of the graphical harmonium, from its sufficient statistics."""
    hrm = circuit.gen_hrm
    hrm_params, _ = circuit.split_coords(params)
    obs_params, xn_params, pop_params = hrm.split_coords(hrm_params)
    pop_hrm = circuit.dep.gen_hrm
    bias, nz_params, z_params = pop_hrm.split_coords(pop_params)
    (xn_map,) = hrm.crs_maps
    (nz_map,) = pop_hrm.crs_maps
    xn_mat, nz_mat = xn_map.to_matrix(xn_params), nz_map.to_matrix(nz_params)
    obs, lat = circuit.obs_man, circuit.lat_man

    def log_joint(x: Array, n: Array, z: Array) -> Array:
        s_z = lat.sufficient_statistic(z)
        return (
            obs.sufficient_statistic(x) @ obs_params
            + obs.log_base_measure(x)
            + x @ xn_mat @ n
            + pop_hrm.obs_man.sufficient_statistic(n) @ bias
            + n @ nz_mat @ s_z
            + s_z @ z_params
            + lat.log_base_measure(z)
        )

    return log_joint


def check(couplings: Couplings) -> bool:
    print(f"couplings: {couplings}")
    circuit = canonical_circuit(1, N_NEURONS, couplings, (5,))
    key = jax.random.PRNGKey(1)
    train_x = jax.random.normal(key, (200, 1))
    u = circuit.initialize_tied(jax.random.PRNGKey(2), train_x, 2.0, 0.5)
    u = {
        k: v
        + (0.02 if k == "cnj" else 0.2)
        * jax.random.normal(jax.random.fold_in(key, i), v.shape)
        for i, (k, v) in enumerate(u.items())
    }
    params = circuit.tie(u)
    states = circuit.states
    x = jnp.array([0.4])
    oks: list[bool] = []

    # 1. The likelihood against BoltzmannLGM
    lgm = BoltzmannLGM(1, PositiveDefinite(), N_NEURONS)
    lkl = circuit.likelihood_function(params)
    ours = jax.vmap(lambda n: circuit.likelihood_at(params, jnp.append(n, 0.0)))(states)
    theirs = jax.vmap(
        lambda n: lgm.lkl_fun_man(lkl, lgm.pst_man.sufficient_statistic(n))
    )(states)
    oks.append(
        report(
            "likelihood vs BoltzmannLGM", float(jnp.max(jnp.abs(ours - theirs))), 1e-12
        )
    )

    # 2. The X - N level is exactly conjugate, and the generative couplings lie on E
    r_n = jax.vmap(lambda n: circuit.conjugation_residual(params, jnp.append(n, 0.0)))(
        states
    )
    oks.append(report("residual r_N", float(jnp.max(jnp.abs(r_n))), 1e-10))
    rows, cols = np.triu_indices(N_NEURONS, 1)
    off_e = np.array([(int(i), int(j)) not in circuit.edges for i, j in zip(rows, cols)])
    gen = circuit.dep.likelihood_at(circuit.conjugated_prior_params(params), jnp.array([0.7]))
    _, gen_off = circuit.neurons.split_couplings(gen)
    oks.append(
        report(
            "generative couplings off E",
            float(jnp.max(jnp.abs(gen_off[off_e]))) if off_e.any() else 0.0,
            1e-12,
        )
    )

    # 2b. The exact sampler of the neurons
    samples = circuit.neurons.sample(jax.random.PRNGKey(4), gen, 100000)
    probs = jax.nn.softmax(jax.vmap(circuit.neurons.sufficient_statistic)(states) @ gen)
    oks.append(
        report(
            "sampler means (in standard errors)",
            float(
                jnp.max(
                    jnp.abs(jnp.mean(samples, 0) - probs @ states)
                    / jnp.sqrt(probs @ states * (1 - probs @ states) / 100000)
                )
            ),
            4.5,
        )
    )

    # 3. Exact log p(x) against grids over x and z
    log_joint = harmonium_log_joint(circuit, params)

    def log_nz(xv: Array) -> Array:
        vals = jax.vmap(lambda n: jax.vmap(lambda z: log_joint(xv, n, z))(ZS))(states)
        return logsumexp(grid_logsumexp(vals, ZS))

    log_unnorm = jax.lax.map(log_nz, XS, batch_size=100)
    brute = log_nz(x) - grid_logsumexp(log_unnorm, XS)
    exact = circuit.exact_log_observable_density(params, x)
    oks.append(report("log p(x)", float(jnp.abs(brute - exact)), 1e-8))

    # 4. log p~(x), ELBO and divergence against the circuit's densities on a grid over z
    def joint(n: Array, z: Array) -> Array:
        return circuit.log_density(params, jnp.concatenate([x, n, z]))

    def recog(n: Array, z: Array) -> Array:
        return circuit.recognition_log_density(params, x, jnp.concatenate([n, z]))

    lp = jax.vmap(lambda n: jax.vmap(lambda z: joint(n, z))(ZS))(states)
    lq = jax.vmap(lambda n: jax.vmap(lambda z: recog(n, z))(ZS))(states)
    log_tilde_brute = logsumexp(grid_logsumexp(lp, ZS))
    q = jnp.exp(lq)
    dz = float(ZS[1, 0] - ZS[0, 0])
    elbo_brute = jnp.sum(q * (lp - lq)) * dz
    post = jax.vmap(lambda n: jax.vmap(lambda z: log_joint(x, n, z))(ZS))(states)
    post = post - logsumexp(grid_logsumexp(post, ZS))
    kl_brute = jnp.sum(q * (lq - post)) * dz
    log_tilde, elbo = circuit.variational_bounds(params, x, N_NODES)
    kl = circuit.recognition_divergence(params, x, N_NODES)
    oks.append(report("log p~(x)", float(jnp.abs(log_tilde - log_tilde_brute)), 1e-6))
    oks.append(report("ELBO", float(jnp.abs(elbo - elbo_brute)), 1e-6))
    oks.append(report("KL(q || p)", float(jnp.abs(kl - kl_brute)), 1e-6))
    print(
        f"  log p(x) {float(exact):.4f}, log p~(x) {float(log_tilde):.4f}, ELBO {float(elbo):.4f}, KL {float(kl):.4f}"
    )

    # 5. The ELBO estimator and its gradient
    n_keys, n_samples = 400, 16
    keys = jax.random.split(jax.random.PRNGKey(3), n_keys)

    def estimate(p: Array, k: Array) -> Array:
        return circuit.elbo_at(k, p, x, n_samples)

    vals, grads = jax.vmap(jax.value_and_grad(estimate), in_axes=(None, 0))(
        params, keys
    )
    exact_grad = jax.grad(lambda p: circuit.variational_bounds(p, x, N_NODES)[1])(
        params
    )
    se = float(jnp.std(vals) / np.sqrt(n_keys))
    oks.append(
        report(
            "ELBO estimator (in standard errors)",
            float(jnp.abs(jnp.mean(vals) - elbo)) / max(se, 1e-12),
            4.0,
        )
    )
    g_mean, g_se = jnp.mean(grads, axis=0), jnp.std(grads, axis=0) / np.sqrt(n_keys)
    z_scores = jnp.abs(g_mean - exact_grad) / jnp.maximum(g_se, 1e-8)
    rel = jnp.linalg.norm(g_mean - exact_grad) / jnp.linalg.norm(exact_grad)
    print(
        f"  gradient: relative error {float(rel):.2e}, largest z-score {float(jnp.max(z_scores)):.2f} over {params.size} coordinates"
    )
    oks.append(report("gradient (largest z-score)", float(jnp.max(z_scores)), 5.0))
    return all(oks)


def main() -> None:
    jax_cli()
    results = [check(c) for c in ("independent", "chain", "full")]
    print("all checks passed" if all(results) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
