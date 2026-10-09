"""Train the canonical circuit on points near a curve by the ELBO, sweeping the conjugation penalty.

The circuit (:mod:`.model`) is fit by maximizing

$$\\mathcal L_\\lambda = \\hat{\\mathcal L} - \\lambda \\Big(\\sum_\\ell \\mathcal R^p_\\ell + \\sum_\\ell \\bar{\\mathcal R}^q_\\ell\\Big),$$

the Monte Carlo ELBO (:meth:`~goal.geometry.VariationalConjugated.mean_elbo`) minus the residual
variances of both levels under the model
(:meth:`~goal.geometry.VariationalConjugated.conjugation_residual_variances`) and under the recognition
model, averaged over the batch
(:meth:`~goal.geometry.VariationalConjugated.mean_recognition_residual_variances`). All three are library
estimators with score-function gradients. $\\lambda$ is increased linearly from zero over the first half
of training, and each final value is one point on the frontier between fit and conjugation. Each seed
gives one initialization, shared by the runs at every $\\lambda$. Nothing constrains the parameters: a run
whose loss becomes NaN is stopped, and its last finite parameters are measured. The ``*_exact``
experiments instead maximize the exact log-likelihood of the graphical harmonium, as a baseline.

``--experiment`` selects the data, the graph of couplings, the conjugation function and the objective, and the results go
to a subdirectory of that name.
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
from .model import CanonicalCircuit, Conjugation, Couplings, Tied, canonical_circuit
from .types import Results, RunResult


@dataclass(frozen=True)
class Experiment:
    """A dataset, a circuit, an objective and a sweep of penalty strengths.

    ``fit`` is the penalized ELBO, or the exact log-likelihood of the graphical harmonium by enumeration
    (a baseline for how much the harmonium gains from $z$; it leaves $\rho_Z$ at its initial value, so
    the variational measurements of those runs describe that $\rho_Z$).
    """

    data: Literal["arc", "bump", "bump8"]
    couplings: Couplings
    conjugation: Conjugation
    fit: Literal["elbo", "exact"]
    lams: tuple[float, ...]


SWEEP = (0.0, 0.3, 1.0, 3.0)

EXPERIMENTS = {
    "bump_chain_exact": Experiment("bump", "chain", "constant", "exact", (0.0,)),
    "bump_chain_constant": Experiment("bump", "chain", "constant", "elbo", SWEEP),
    "bump_chain_mlp": Experiment("bump", "chain", "mlp", "elbo", SWEEP),
    "bump8_chain_exact": Experiment("bump8", "chain", "constant", "exact", (0.0,)),
    "bump8_chain_constant": Experiment("bump8", "chain", "constant", "elbo", SWEEP),
    "bump8_chain_mlp": Experiment("bump8", "chain", "mlp", "elbo", SWEEP),
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
    circuit: CanonicalCircuit,
    u: Tied,
    key: Array,
    xs: Array,
    lam: Array,
    n_samples: int,
) -> tuple[Array, tuple[Array, Array, Array]]:
    """The negative penalized ELBO, and the ELBO and the summed residual variances under $q$ and $\\tilde p$."""
    params = circuit.tie(u)
    k_elbo, k_p, k_q = jax.random.split(key, 3)
    elbo = circuit.mean_elbo(k_elbo, params, xs, n_samples)
    var_p = sum(
        circuit.conjugation_residual_variances(k_p, params, n_samples),
        start=jnp.zeros(()),
    )
    var_q = sum(
        circuit.mean_recognition_residual_variances(k_q, params, xs, n_samples),
        start=jnp.zeros(()),
    )
    return -(elbo - lam * (var_q + var_p)), (elbo, var_q, var_p)


def exact_objective(circuit: CanonicalCircuit, u: Tied, xs: Array) -> Array:
    """The negative mean exact log-likelihood of the graphical harmonium."""
    params = circuit.tie(u)
    return -jnp.mean(
        jax.vmap(circuit.exact_log_observable_density, in_axes=(None, 0))(params, xs)
    )


def min_precision(circuit: CanonicalCircuit, params: Array, xs: Array) -> Array:
    """Smallest precision of $z$ over $p(z \\mid n)$ at every state, the prior $\\theta_Z + \\rho_Z$, and the recognition model at ``xs``: negative when a run has left the domain."""
    lat, dep = circuit.lat_man, circuit.dep

    def precision(nrm_params: Array) -> Array:
        _, prc = lat.split_location_precision(nrm_params)
        return lat.cov_man.to_matrix(prc)[0, 0]

    def z_params(dep_params: Array) -> Array:
        nrm_params, _ = dep.dep_man.split_coords(
            dep.conjugated_prior_params(dep_params)
        )
        return nrm_params

    hrm_params, _ = circuit.split_coords(params)
    _, _, pop_params = circuit.gen_hrm.split_coords(hrm_params)
    states = jax.vmap(dep.gen_hrm.posterior_at, in_axes=(None, 0))(
        pop_params, circuit.states
    )
    prior = z_params(circuit.conjugated_prior_params(params))
    recog = jax.vmap(lambda x: z_params(circuit.recognition_at(params, x)))(xs)
    return jnp.min(
        jnp.concatenate(
            [
                jax.vmap(precision)(states),
                precision(prior)[None],
                jax.vmap(precision)(recog),
            ]
        )
    )


def spearman(a: Array, b: Array) -> float:
    ra = jnp.argsort(jnp.argsort(a)).astype(float)
    rb = jnp.argsort(jnp.argsort(b)).astype(float)
    return float(jnp.corrcoef(ra, rb)[0, 1])


def measure(
    circuit: CanonicalCircuit,
    key: Array,
    u: Tied,
    test_x: Array,
    test_t: Array,
    data: str,
    noise: float,
    n_samples: int,
) -> dict[str, Any]:
    """Final measurements on the test set."""
    params = circuit.tie(u)
    lat, dep = circuit.lat_man, circuit.dep
    k_p, k_q, k_s = jax.random.split(key, 3)
    var_p = circuit.conjugation_residual_variances(k_p, params, n_samples)
    var_q = circuit.mean_recognition_residual_variances(k_q, params, test_x, n_samples)

    # Posterior over z: the exact mixture against the recognition Gaussian
    def posterior_moments(x: Array) -> tuple[Array, Array, Array, Array]:
        post = circuit.posterior_at(params, x)
        wts = jax.nn.softmax(circuit.state_log_weights(post))
        comps = jax.vmap(dep.gen_hrm.posterior_at, in_axes=(None, 0))(
            post, circuit.states
        )
        means = jax.vmap(lambda p: lat.split_mean_second_moment(lat.to_mean(p)))(comps)
        m_ex = wts @ means[0][:, 0]
        s2_ex = wts @ means[1][:, 0]
        q_params, _ = dep.dep_man.split_coords(
            dep.conjugated_prior_params(circuit.recognition_at(params, x))
        )
        m_q, s2_q = lat.split_mean_second_moment(lat.to_mean(q_params))
        return m_ex, jnp.sqrt(s2_ex - m_ex**2), m_q[0], jnp.sqrt(s2_q[0] - m_q[0] ** 2)

    m_ex, sd_ex, m_q, sd_q = jax.vmap(posterior_moments)(test_x)

    # Tuning curves: the logits theta_N + rho_N + Theta_NZ s_Z(z) at the prior, without couplings
    zs = jnp.linspace(-4.0, 4.0, 201)[:, None]
    dep_prior = circuit.conjugated_prior_params(params)
    logits = jax.vmap(
        lambda z: circuit.neurons.split_couplings(dep.likelihood_at(dep_prior, z))[0]
    )(zs)

    # The exact prior over z, a mixture over the states, against the Gaussian of the variational model
    hrm_params, _ = circuit.split_coords(params)
    _, _, pop_params = circuit.gen_hrm.split_coords(hrm_params)
    wts = jax.nn.softmax(circuit.exact_state_log_weights(params))
    comps = jax.vmap(dep.gen_hrm.posterior_at, in_axes=(None, 0))(
        pop_params, circuit.states
    )
    dens = jax.vmap(
        lambda z: wts @ jnp.exp(jax.vmap(lat.log_density, in_axes=(0, None))(comps, z))
    )(zs)
    p_params, _ = dep.dep_man.split_coords(dep.conjugated_prior_params(dep_prior))
    gauss = jnp.exp(jax.vmap(lat.log_density, in_axes=(None, 0))(p_params, zs))

    # Samples of the variational model, and the means of x at the sampled neurons
    obs_dim = circuit.obs_dim
    xnz = circuit.sample(k_s, params, 1000)
    comp_means = jax.vmap(
        lambda nz: circuit.obs_man.to_mean(circuit.likelihood_at(params, nz))[:obs_dim]
    )(xnz[:, obs_dim:])
    return {
        "final_var_p_levels": [float(v) for v in var_p],
        "final_var_q_levels": [float(v) for v in var_q],
        "corr_t": spearman(m_ex, test_t),
        "off_curve": off_curve_fraction(data, xnz[:, :obs_dim], noise),
        "off_curve_means": off_curve_fraction(data, comp_means, noise),
        "samples": xnz[:, :obs_dim].tolist(),
        "post_mean_exact": m_ex.tolist(),
        "post_mean_q": m_q.tolist(),
        "post_sd_exact": sd_ex.tolist(),
        "post_sd_q": sd_q.tolist(),
        "tuning_z": zs[:, 0].tolist(),
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
    circuit = canonical_circuit(obs_dim, 10, exp.couplings, exp.conjugation, (32,))
    n_train, n_test, noise = 2000, 500, 0.05
    z_range = 2.0

    # Training
    n_seeds = 3
    n_chunks, chunk_steps, batch_size, learning_rate = 50, 100, 128, 2e-2
    n_samples, n_eval_samples, n_nodes = 16, 64, 40
    anneal_steps = n_chunks * chunk_steps // 2

    k_train, k_test, k_init, k_run, k_eval = jax.random.split(key, 5)
    train_x, _ = curve_data(exp.data, k_train, n_train, noise)
    test_x, test_t = curve_data(exp.data, k_test, n_test, noise)
    optimizer = optax.adam(learning_rate)

    def evaluate(u: Tied) -> tuple[Array, ...]:
        params = circuit.tie(u)
        train_ll = jnp.mean(
            jax.vmap(circuit.exact_log_observable_density, in_axes=(None, 0))(
                params, train_x
            )
        )
        test_ll = jnp.mean(
            jax.vmap(circuit.exact_log_observable_density, in_axes=(None, 0))(
                params, test_x
            )
        )
        log_tilde, elbo = jax.vmap(circuit.variational_bounds, in_axes=(None, 0, None))(
            params, test_x, n_nodes
        )
        kl = jax.vmap(circuit.recognition_divergence, in_axes=(None, 0, None))(
            params, test_x, n_nodes
        )
        _, (_, var_q, var_p) = objective(
            circuit, u, k_eval, test_x, jnp.array(0.0), n_eval_samples
        )
        return (
            train_ll,
            test_ll,
            jnp.mean(log_tilde),
            jnp.mean(elbo),
            jnp.mean(kl),
            var_q,
            var_p,
            min_precision(circuit, params, test_x),
        )

    def train_chunk(
        carry: tuple[Tied, Any, Array], lam_final: Array
    ) -> tuple[tuple[Tied, Any, Array], Array]:
        def step(
            carry: tuple[Tied, Any, Array], _: None
        ) -> tuple[tuple[Tied, Any, Array], Array]:
            u, opt_state, step_key = carry
            step_key, k_batch, k_obj = jax.random.split(step_key, 3)
            idx = jax.random.choice(k_batch, n_train, (batch_size,), replace=False)
            count = optax.tree_utils.tree_get(opt_state, "count")
            lam = lam_final * jnp.minimum(1.0, count / anneal_steps)
            if exp.fit == "exact":
                loss, grads = jax.value_and_grad(
                    lambda v: exact_objective(circuit, v, train_x[idx])
                )(u)
            else:
                (loss, _), grads = jax.value_and_grad(
                    lambda v: objective(
                        circuit, v, k_obj, train_x[idx], lam, n_samples
                    ),
                    has_aux=True,
                )(u)
            updates, opt_state = optimizer.update(grads, opt_state, u)
            return (optax.apply_updates(u, updates), opt_state, step_key), loss  # pyright: ignore[reportReturnType]

        return jax.lax.scan(step, carry, None, chunk_steps)

    train_chunk_jit = jax.jit(train_chunk)
    evaluate_jit = jax.jit(evaluate)

    names = (
        "train_ll",
        "test_ll",
        "log_tilde",
        "elbo",
        "kl_q",
        "var_q",
        "var_p",
        "min_prc",
    )
    runs: list[RunResult] = []
    steps = [chunk * chunk_steps for chunk in range(n_chunks + 1)]
    for seed, lam in [(seed, lam) for seed in range(n_seeds) for lam in exp.lams]:
        u0 = circuit.initialize_tied(jax.random.fold_in(k_init, seed), train_x, z_range)
        carry = (u0, optimizer.init(u0), jax.random.fold_in(k_run, seed))
        hist: dict[str, list[float]] = {name: [] for name in names}
        stopped: int | None = None
        for chunk in range(n_chunks + 1):
            if chunk > 0:
                new_carry, losses = train_chunk_jit(carry, jnp.array(lam))
                if not bool(jnp.all(jnp.isfinite(losses))):
                    stopped = (chunk - 1) * chunk_steps + int(
                        jnp.argmin(jnp.isfinite(losses))
                    )
                    print(f"seed {seed}, lambda {lam:g}: NaN loss at step {stopped}")
                    break
                carry = new_carry
            for name, val in zip(names, evaluate_jit(carry[0]), strict=True):
                hist[name].append(float(val))
            print(
                f"seed {seed}, lambda {lam:g}, step {chunk * chunk_steps}: "
                f"test ll {hist['test_ll'][-1]:.4f}, log p~ {hist['log_tilde'][-1]:.4f}, "
                f"elbo {hist['elbo'][-1]:.4f}, kl {hist['kl_q'][-1]:.2e}, "
                f"var_q {hist['var_q'][-1]:.2e}, var_p {hist['var_p'][-1]:.2e}, "
                f"min prc {hist['min_prc'][-1]:.3f}",
                flush=True,
            )
        runs.append(
            {
                "lam": lam,
                "seed": seed,
                "stopped": stopped,
                **hist,
                **{f"final_{name}": hist[name][-1] for name in names},
                **measure(
                    circuit,
                    jax.random.fold_in(k_eval, 1),
                    carry[0],
                    test_x,
                    test_t,
                    exp.data,
                    noise,
                    n_eval_samples,
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
