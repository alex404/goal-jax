"""Check the canonical circuit's computations against brute-force integration.

The brute force is written from the densities directly, in numpy, without the library: neurons are
enumerated, $z$ is integrated on a fine grid, and $x$ is integrated in closed form for the
log-partition function and on a grid for the normalization of $p_X$. Checked:

1. the logits of the population code against the tuning-curve parameterization;
2. the exact log-partition function and $\\log p_X$;
3. $\\int p_X = 1$;
4. the log-likelihood identity by quadrature, against the exact value;
5. the exact posterior mean against the grid;
6. the least-squares residual variance, against the variance of the residual at its own nodes;
7. $\\log \\tilde p_X$ of the variational model by quadrature, against a grid over $z$, and that the ELBO
   bounds it;
8. that the prior over the neurons has the stored biases and noise correlations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from ..shared import jax_cli
from .model import CanonicalCircuit


def brute_force(
    u: dict, xs: np.ndarray, n_neurons: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Brute-force log-partition function, log-likelihoods and posterior means of $z$."""
    u = {k: np.asarray(v) for k, v in u.items()}

    def chol(c: np.ndarray) -> np.ndarray:
        low = np.tril(c, -1) + np.diag(np.exp(np.diag(c)))
        return low @ low.T

    prc_x, mu_x, w = chol(u["x_chol"]), u["x_mean"], u["xn"]
    a = chol(u["a_chol"])[0, 0]
    mus = u["preferred"][:, 0]
    lam_z, m_z = chol(u["z_chol"])[0, 0], u["z_mean"][0]
    rows, cols = np.triu_indices(n_neurons, 1)
    cpl = np.zeros((n_neurons, n_neurons))
    cpl[rows, cols] = u["couplings"]
    states = np.array(
        [[(s >> i) & 1 for i in range(n_neurons)] for s in range(2**n_neurons)], float
    )
    zs = np.linspace(-20, 20, 40001)
    dz = zs[1] - zs[0]
    # log of exp(s_N theta_N + n Theta_NZ s_Z(z) + s_Z(z) theta_Z) * base(z), shape (states, grid)
    logits = u["peak"][:, None] - 0.5 * a * (zs[None, :] - mus[:, None]) ** 2
    log_nz = (
        states @ logits
        + np.einsum("si,ij,sj->s", states, cpl, states)[:, None]
        - 0.5 * lam_z * zs[None, :] ** 2
        + lam_z * m_z * zs[None, :]
        - 0.5 * np.log(2 * np.pi)
    )
    h = prc_x @ mu_x
    d_x = len(mu_x)

    def log_int_x(h_n: np.ndarray) -> np.ndarray:
        cov = np.linalg.inv(prc_x)
        return (
            0.5 * np.einsum("si,ij,sj->s", h_n, cov, h_n)
            - 0.5 * np.linalg.slogdet(prc_x)[1]
        )

    log_psi = logsumexp(
        log_nz + log_int_x(h[None, :] + states @ w.T)[:, None]
    ) + np.log(dz)
    lls, post_means = [], []
    for x in xs:
        log_x = -0.5 * x @ prc_x @ x + x @ h - 0.5 * d_x * np.log(2 * np.pi)
        log_joint = log_nz + (states @ (w.T @ x))[:, None] + log_x
        lls.append(logsumexp(log_joint) + np.log(dz) - log_psi)
        pz = np.exp(logsumexp(log_joint, axis=0) - logsumexp(log_joint))
        post_means.append(np.sum(pz * zs))
    return float(log_psi), np.array(lls), np.array(post_means)


def main() -> None:
    jax_cli()
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)
    n_neurons = 6
    circuit = CanonicalCircuit(
        obs_dim=2,
        n_neurons=n_neurons,
        lat_dim=1,
        noise_correlations="harmonium",
        latent=True,
        exact_reference=False,
        n_nodes=40,
    )
    k_data, k_u, k_x = jax.random.split(key, 3)
    data = jax.random.normal(k_data, (100, 2))
    u = circuit.initialize(k_u, data)
    keys = jax.random.split(k_u, len(u))
    u = {
        k: v + 0.3 * jax.random.normal(kk, v.shape)
        for (k, v), kk in zip(u.items(), keys)
    }
    params = circuit.constrain(u)
    xs = jax.random.normal(k_x, (5, 2))

    # 1. Logits of the population code
    _, pch_params = circuit.split_params(params)
    z = jnp.array([0.7])
    logits = circuit.blz_man.split_couplings(circuit.pch.likelihood_at(pch_params, z))[
        0
    ]
    a = jnp.exp(u["a_chol"][0, 0]) ** 2
    expected = u["peak"] - 0.5 * a * (z[0] - u["preferred"][:, 0]) ** 2
    print(f"logits: max error {jnp.max(jnp.abs(logits - expected)):.2e}")

    # 2-3. Log-partition function and log-likelihoods
    log_psi, lls, post_means = brute_force(u, np.asarray(xs), n_neurons)
    print(
        f"log-partition: {circuit.log_partition_function(params):.10f} vs {log_psi:.10f}"
    )
    exact = jax.vmap(circuit.log_observable_density, in_axes=(None, 0))(params, xs)
    print(f"log p_X: max error {np.max(np.abs(np.asarray(exact) - lls)):.2e}")
    grid = np.linspace(-30, 30, 1201)
    gx = jnp.asarray(np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2))
    dens = jnp.exp(
        jax.vmap(circuit.log_observable_density, in_axes=(None, 0))(params, gx)
    )
    print(f"integral of p_X: {float(jnp.sum(dens)) * (grid[1] - grid[0]) ** 2:.6f}")

    # 4. The identity
    ident, ess_q, ess_p = jax.vmap(circuit.log_likelihood_identity, in_axes=(None, 0))(
        params, xs
    )
    print(f"identity: max error {np.max(np.abs(np.asarray(ident) - lls)):.2e}")
    print(
        f"relative ESS at recognition nodes: {np.round(np.asarray(ess_q), 3)}, prior: {float(ess_p[0]):.3f}"
    )

    # 5. Posterior means
    pm = jax.vmap(circuit.exact_posterior_means, in_axes=(None, 0))(params, xs)[:, 0]
    print(
        f"posterior mean: max error {np.max(np.abs(np.asarray(pm) - post_means)):.2e}"
    )

    # 6. Residual variance
    _, pch_params = circuit.split_params(params)
    _, nz_params, z_params = circuit.pch.split_coords(pch_params)
    beta, q_params, var = circuit.recognition(params, xs[0])
    zs, ws = circuit.quadrature(q_params)
    rs = circuit.residuals(nz_params, beta, q_params - z_params, zs)
    direct = jnp.sum(ws * rs**2) - jnp.sum(ws * rs) ** 2
    print(f"residual variance: least squares {var:.6e}, direct {direct:.6e}")

    # 7. The variational model, on a grid over z, and its ELBO
    lgm_params, _ = circuit.split_params(params)
    beta_p, p_params, _ = circuit.generative_prior(params)
    zg = jnp.linspace(-15.0, 15.0, 6001)
    stats = jax.vmap(circuit.blz_man.sufficient_statistic)(circuit.blz_man.states)
    log_pz = jax.vmap(circuit.lat_man.log_density, in_axes=(None, 0))(
        p_params, zg[:, None]
    )
    log_cond = jax.vmap(
        lambda z: jax.nn.log_softmax(
            stats
            @ circuit.pch.lkl_fun_man(
                circuit.pch.lkl_fun_man.join_coords(beta_p, nz_params),
                circuit.lat_man.sufficient_statistic(z),
            )
        )
    )(zg[:, None])
    lkl_params = jax.vmap(circuit.lgm.likelihood_at, in_axes=(None, 0))(
        lgm_params, circuit.blz_man.states
    )

    def log_tilde_grid(x: jax.Array) -> jax.Array:
        log_x = jax.vmap(circuit.obs_man.log_density, in_axes=(0, None))(lkl_params, x)
        log_joint = log_pz[:, None] + log_cond + log_x[None, :]
        return jax.scipy.special.logsumexp(log_joint) + jnp.log(zg[1] - zg[0])

    grid_tilde = jax.vmap(log_tilde_grid)(xs)
    quad_tilde, elbo = jax.vmap(circuit.variational_bounds, in_axes=(None, 0))(
        params, xs
    )
    print(f"log p~_X: max error {jnp.max(jnp.abs(quad_tilde - grid_tilde)):.2e}")
    print(
        f"ELBO <= log p~_X: {bool(jnp.all(elbo <= quad_tilde))}, gaps {np.round(np.asarray(quad_tilde - elbo), 4)}"
    )
    coarse = np.linspace(-12, 12, 121)
    cx = jnp.asarray(np.stack(np.meshgrid(coarse, coarse), axis=-1).reshape(-1, 2))
    tilde_dens = jnp.exp(jax.lax.map(log_tilde_grid, cx, batch_size=64))
    print(
        f"integral of p~_X: {float(jnp.sum(tilde_dens)) * (coarse[1] - coarse[0]) ** 2:.4f}"
    )

    # 8. Noise correlations: the prior over the neurons has the stored biases and couplings
    biases = u["peak"] - 0.5 * a * u["preferred"][:, 0] ** 2
    rows, cols = np.triu_indices(n_neurons, 1)
    for kind, mask in (
        ("none", np.zeros(rows.size, bool)),
        ("chain", cols == rows + 1),
    ):
        noisy = CanonicalCircuit(
            obs_dim=2,
            n_neurons=n_neurons,
            lat_dim=1,
            noise_correlations=kind,
            latent=True,
            exact_reference=True,
        )
        cpl = jax.random.normal(k_x, (int(mask.sum()),))
        lgm_params, _ = noisy.split_params(noisy.constrain({**u, "couplings": cpl}))
        diag, off = noisy.blz_man.split_couplings(noisy.lgm.prior(lgm_params))
        expected = jnp.zeros(rows.size).at[np.flatnonzero(mask)].set(cpl)
        print(
            f"noise correlations {kind}: coupling error {jnp.max(jnp.abs(off - expected)):.2e}, "
            f"bias error {jnp.max(jnp.abs(diag - biases)):.2e}"
        )


if __name__ == "__main__":
    main()
