"""Train the three-layer canonical circuit on points near a curve, sweeping the conjugation penalty.

The circuit $X - N - Z$ (:mod:`.model`) is fit by maximizing either the exact log-likelihood of the
harmonium or the ELBO of the variational model, minus $\\lambda$ times the residual variances of the population code edge, at the recognition model and at
the prior. $\\lambda$ is increased linearly from zero over the first half of training, and each final
value is one point on the frontier between fit and conjugation. Each seed gives one initialization,
shared by the runs at every $\\lambda$.

``--experiment`` selects the circuit and the objective, and the results go to a subdirectory of that name.
"""

import argparse
from dataclasses import dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from ..shared import example_paths, jax_cli
from .model import CanonicalCircuit, NoiseCorrelations, Unconstrained
from .types import Results, RunResult


@dataclass(frozen=True)
class Experiment:
    """A dataset, a circuit and an objective."""

    data: Literal["arc", "bump", "bump8"]
    noise_correlations: NoiseCorrelations
    latent: bool
    use_elbo: bool
    lams: tuple[float, ...]


SWEEP = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
BUMP_SWEEP = (0.0, 0.3, 1.0, 3.0)

EXPERIMENTS = {
    "arc_none_ll": Experiment("arc", "none", True, False, SWEEP),
    "arc_none_elbo": Experiment("arc", "none", True, True, SWEEP),
    "arc_harmonium_ll": Experiment("arc", "harmonium", True, False, SWEEP),
    "bump_none_elbo": Experiment("bump", "none", True, True, BUMP_SWEEP),
    "bump_chain_elbo": Experiment("bump", "chain", True, True, BUMP_SWEEP),
    "bump_full_elbo": Experiment("bump", "full", True, True, BUMP_SWEEP),
    "bump_chain_nolatent": Experiment("bump", "chain", False, False, (0.0,)),
    "bump8_chain_elbo": Experiment("bump8", "chain", True, True, BUMP_SWEEP),
    "bump8_full_elbo": Experiment("bump8", "full", True, True, BUMP_SWEEP),
    "bump8_chain_nolatent": Experiment("bump8", "chain", False, False, (0.0,)),
}


def curve(data: str, ts: Array) -> Array:
    """Noiseless points of a dataset's curve at coordinates ``ts``.

    ``"arc"`` is a three-quarter circle in $\\mathbb R^2$. ``"bump"`` is a one-dimensional retina of
    16 pixels on $[0, 1]$ seeing a Gaussian bump of width $0.08$ at position $t$, and ``"bump8"`` one of
    8 pixels seeing a bump of width $0.15$.
    """
    if data == "arc":
        return jnp.stack([jnp.sin(ts), jnp.cos(ts)], axis=1)
    n_pixels, width = (16, 0.08) if data == "bump" else (8, 0.15)
    centres = jnp.linspace(0.0, 1.0, n_pixels)
    return jnp.exp(-((ts[:, None] - centres[None, :]) ** 2) / (2 * width**2))


def curve_range(data: str) -> tuple[float, float]:
    return (-0.75 * np.pi, 0.75 * np.pi) if data == "arc" else (0.0, 1.0)


def curve_data(data: str, key: Array, n: int, noise: float) -> tuple[Array, Array]:
    """Noisy points near a dataset's curve, with their curve coordinate $t$."""
    k_t, k_e = jax.random.split(key)
    lo, hi = curve_range(data)
    t = jax.random.uniform(k_t, (n,), minval=lo, maxval=hi)
    xs = curve(data, t)
    return xs + noise * jax.random.normal(k_e, xs.shape), t


def off_curve_fraction(data: str, xs: Array, noise: float) -> float:
    """Fraction of points whose mean squared distance to the curve exceeds twice the noise variance."""
    lo, hi = curve_range(data)
    ref = curve(data, jnp.linspace(lo, hi, 1000))
    dists = jax.lax.map(
        lambda x: jnp.min(jnp.mean((ref - x) ** 2, axis=1)), xs, batch_size=256
    )
    return float(jnp.mean(dists > 2 * noise**2))


def objective(
    circuit: CanonicalCircuit, u: Unconstrained, xs: Array, lam: Array, use_elbo: bool
) -> tuple[Array, tuple[Array, Array, Array]]:
    """Negative of the mean fit term minus the penalties, and the three terms.

    The fit term is the exact log-likelihood of the harmonium, or the ELBO of the variational model.
    """
    params = circuit.constrain(u)
    if use_elbo:
        fit = jnp.mean(
            jax.vmap(circuit.variational_bounds, in_axes=(None, 0))(params, xs)[1]
        )
    else:
        fit = jnp.mean(
            jax.vmap(circuit.log_observable_density, in_axes=(None, 0))(params, xs)
        )
    var_q = jnp.mean(jax.vmap(lambda x: circuit.recognition(params, x)[2])(xs))
    _, _, var_p = circuit.generative_prior(params)
    return -(fit - lam * (var_q + var_p)), (fit, var_q, var_p)


def spearman(a: Array, b: Array) -> float:
    ra = jnp.argsort(jnp.argsort(a)).astype(float)
    rb = jnp.argsort(jnp.argsort(b)).astype(float)
    return float(jnp.corrcoef(ra, rb)[0, 1])


def affine_r2(xs: Array, ys: Array) -> list[float]:
    """Coefficient of determination of an affine regression of each column of ``ys`` on ``xs``."""
    design = jnp.concatenate([xs, jnp.ones((xs.shape[0], 1))], axis=1)
    coef, *_ = jnp.linalg.lstsq(design, ys)
    res = ys - design @ coef
    return list(map(float, 1 - jnp.var(res, axis=0) / jnp.var(ys, axis=0)))


def measure(
    circuit: CanonicalCircuit,
    key: Array,
    u: Unconstrained,
    test_x: Array,
    test_t: Array,
    data: str,
    noise: float,
) -> dict[str, Any]:
    """Final measurements on the test set."""
    params = circuit.constrain(u)
    lat = circuit.lat_man
    exact = jax.vmap(circuit.log_observable_density, in_axes=(None, 0))(params, test_x)
    log_tilde, elbo = jax.vmap(circuit.variational_bounds, in_axes=(None, 0))(
        params, test_x
    )
    ident, ess_q, ess_p = jax.vmap(circuit.log_likelihood_identity, in_axes=(None, 0))(
        params, test_x
    )
    _, q_params, var_q = jax.vmap(circuit.recognition, in_axes=(None, 0))(
        params, test_x
    )
    _, p_params, var_p = circuit.generative_prior(params)

    # Exact posteriors, moment matched
    ex_means = jax.vmap(circuit.exact_posterior_means, in_axes=(None, 0))(
        params, test_x
    )
    ex_params = jax.vmap(lat.to_natural)(ex_means)
    q_means = jax.vmap(lat.to_mean)(q_params)
    kl = jax.vmap(lat.relative_entropy)(ex_params, q_params)

    def mean_sd(means: Array) -> tuple[Array, Array]:
        m, s2 = means[:, 0], means[:, 1]
        return m, jnp.sqrt(s2 - m**2)

    m_ex, sd_ex = mean_sd(ex_means)
    m_q, sd_q = mean_sd(q_means)

    # Tuning curves (prior logits theta^*_N + Theta_NZ s_Z(z), without couplings) and the prior over z
    zs = jnp.linspace(-4.0, 4.0, 201)
    lgm_params, pch_params = circuit.split_params(params)
    _, nz_params, _ = circuit.pch.split_coords(pch_params)
    lkl_params = circuit.pch.lkl_fun_man.join_coords(
        circuit.lgm.prior(lgm_params), nz_params
    )
    logits = jax.vmap(
        lambda z: circuit.blz_man.split_couplings(
            circuit.pch.lkl_fun_man(lkl_params, lat.sufficient_statistic(z))
        )[0]
    )(zs[:, None])
    state_params = circuit.state_latents(params)
    log_wts = circuit._state_log_weights(state_params, circuit.lgm.prior(lgm_params))  # pyright: ignore[reportPrivateUsage]
    wts = jax.nn.softmax(log_wts)
    dens = jax.vmap(
        lambda z: (
            wts @ jnp.exp(jax.vmap(lat.log_density, in_axes=(0, None))(state_params, z))
        )
    )(zs[:, None])
    gauss = jnp.exp(jax.vmap(lat.log_density, in_axes=(None, 0))(p_params, zs[:, None]))

    samples, sample_ns, _ = circuit.sample(key, params, 1000)
    lgm_params, _ = circuit.split_params(params)
    comp_means = jax.vmap(
        lambda n: circuit.obs_man.split_mean_covariance(
            circuit.obs_man.to_mean(circuit.lgm.likelihood_at(lgm_params, n))
        )[0]
    )(sample_ns)
    return {
        "final_test_ll": float(jnp.mean(exact)),
        "final_log_tilde": float(jnp.mean(log_tilde)),
        "final_elbo": float(jnp.mean(elbo)),
        "final_var_q": float(jnp.mean(var_q)),
        "final_var_p": float(var_p),
        "identity_error": float(jnp.mean(jnp.abs(ident - exact))),
        "ess_q": float(jnp.mean(ess_q)),
        "ess_p": float(jnp.mean(ess_p)),
        "kl_q": float(jnp.mean(kl)),
        "r2_q": affine_r2(test_x, q_params),
        "r2_exact": affine_r2(test_x, ex_params),
        "corr_t": spearman(m_ex, test_t),
        "off_curve": off_curve_fraction(data, samples, noise),
        "off_curve_means": off_curve_fraction(data, comp_means, noise),
        "samples": samples.tolist(),
        "post_mean_exact": m_ex.tolist(),
        "post_mean_q": m_q.tolist(),
        "post_sd_exact": sd_ex.tolist(),
        "post_sd_q": sd_q.tolist(),
        "tuning_z": zs.tolist(),
        "tuning_logits": logits.T.tolist(),
        "prior_density": dens.tolist(),
        "prior_gaussian": gauss.tolist(),
    }


def main() -> None:
    jax_cli()
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), required=True)
    experiment = parser.parse_known_args()[0].experiment
    exp = EXPERIMENTS[experiment]
    paths = example_paths(__file__)
    paths = replace(paths, results_dir=paths.results_dir / experiment)
    key = jax.random.PRNGKey(0)

    # Model and data
    obs_dim = {"arc": 2, "bump": 16, "bump8": 8}[exp.data]
    n_neurons = 10
    circuit = CanonicalCircuit(
        obs_dim=obs_dim,
        n_neurons=n_neurons,
        lat_dim=1,
        noise_correlations=exp.noise_correlations,
        latent=exp.latent,
        exact_reference=True,
    )
    n_train, n_test, noise = 2000, 500, 0.05
    use_elbo = exp.use_elbo

    # Training
    lams = exp.lams
    n_seeds = 3
    n_chunks, chunk_steps, batch_size, learning_rate = 50, 100, 128, 2e-2
    anneal_steps = n_chunks * chunk_steps // 2
    # The bias of Z is held at its initial value (standard normal), which fixes the scale and location
    # of z: otherwise rescaling z with the tuning curves and the bias leaves the likelihood unchanged
    fixed = {"z_mean", "z_chol"}

    k_train, k_test, k_init, k_run = jax.random.split(key, 4)
    train_x, _ = curve_data(exp.data, k_train, n_train, noise)
    test_x, test_t = curve_data(exp.data, k_test, n_test, noise)
    optimizer = optax.adam(learning_rate)

    def evaluate(u: Unconstrained) -> tuple[Array, ...]:
        params = circuit.constrain(u)
        train_ll = jnp.mean(
            jax.vmap(circuit.log_observable_density, in_axes=(None, 0))(params, train_x)
        )
        _, (test_ll, var_q, var_p) = objective(
            circuit, u, test_x, jnp.array(0.0), False
        )
        log_tilde, elbo = jax.vmap(circuit.variational_bounds, in_axes=(None, 0))(
            params, test_x
        )
        return train_ll, test_ll, jnp.mean(log_tilde), jnp.mean(elbo), var_q, var_p

    def train_chunk(
        carry: tuple[Unconstrained, Any, Array], lam_final: Array
    ) -> tuple[Unconstrained, Any, Array]:
        def step(
            carry: tuple[Unconstrained, Any, Array], _: None
        ) -> tuple[tuple[Unconstrained, Any, Array], None]:
            u, opt_state, step_key = carry
            step_key, k_batch = jax.random.split(step_key)
            idx = jax.random.choice(k_batch, n_train, (batch_size,), replace=False)
            count = optax.tree_utils.tree_get(opt_state, "count")
            lam = lam_final * jnp.minimum(1.0, count / anneal_steps)
            grads, _ = jax.grad(
                lambda v: objective(circuit, v, train_x[idx], lam, use_elbo),
                has_aux=True,
            )(u)
            grads = {
                k: jnp.zeros_like(g) if k in fixed else g for k, g in grads.items()
            }
            updates, opt_state = optimizer.update(grads, opt_state, u)
            return (optax.apply_updates(u, updates), opt_state, step_key), None

        carry, _ = jax.lax.scan(step, carry, None, chunk_steps)
        return carry

    train_chunk_jit = jax.jit(train_chunk)
    evaluate_jit = jax.jit(evaluate)

    runs: list[RunResult] = []
    steps = [chunk * chunk_steps for chunk in range(n_chunks + 1)]
    for seed, lam in [(seed, lam) for seed in range(n_seeds) for lam in lams]:
        u0 = circuit.initialize(jax.random.fold_in(k_init, seed), train_x)
        carry = (u0, optimizer.init(u0), jax.random.fold_in(k_run, seed))
        hist: dict[str, list[float]] = {
            name: []
            for name in ("train_ll", "test_ll", "log_tilde", "elbo", "var_q", "var_p")
        }
        for chunk in range(n_chunks + 1):
            if chunk > 0:
                carry = train_chunk_jit(carry, jnp.array(lam))
            for name, val in zip(hist, evaluate_jit(carry[0]), strict=True):
                hist[name].append(float(val))
            print(
                f"seed {seed}, lambda {lam:g}, step {chunk * chunk_steps}: "
                f"test ll {hist['test_ll'][-1]:.4f}, elbo {hist['elbo'][-1]:.4f}",
                flush=True,
            )
        print(
            f"seed {seed}, lambda {lam:5.2f}: test ll {hist['test_ll'][-1]:.4f}, "
            f"var_q {hist['var_q'][-1]:.2e}, var_p {hist['var_p'][-1]:.2e}",
            flush=True,
        )
        runs.append(
            {
                "lam": lam,
                "seed": seed,
                **hist,
                **measure(
                    circuit,
                    jax.random.fold_in(k_run, 1),
                    carry[0],
                    test_x,
                    test_t,
                    exp.data,
                    noise,
                ),
            }  # pyright: ignore[reportArgumentType]
        )

    results: Results = {
        "steps": steps,
        "train_x": train_x.tolist(),
        "test_x": test_x.tolist(),
        "test_t": test_t.tolist(),
        "data_off_curve": off_curve_fraction(exp.data, test_x, noise),
        "runs": runs,
    }
    paths.save_analysis(results)


if __name__ == "__main__":
    main()
