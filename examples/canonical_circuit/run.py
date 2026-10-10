"""Fit the canonical circuit to a bump on a retina, by brute-force maximum likelihood and by the penalized ELBO.

The data are points of a one-dimensional retina of 16 pixels on $[0, 1]$ seeing a Gaussian bump of width
$0.08$ at a uniformly distributed position $t$, with pixel noise. The circuit (:mod:`.model`) is fit over a
grid of conditions:

- the graph of the generative couplings of the neurons: none, a chain, or all pairs;
- with $z$, or without ($\\Theta_{NZ} = 0$, and $\\rho_Z = 0$, which is then exact). Without $z$ both
  levels are exactly conjugate, the ELBO is the log-likelihood and its gradient the brute-force one, so
  only the brute-force fit is run;
- by maximizing the exact log-likelihood of the graphical harmonium (``exact``, by enumeration of the
  neurons), or the penalized ELBO

  $$\\mathcal L_\\lambda = \\hat{\\mathcal L} - \\lambda \\Big(\\sum_\\ell \\mathcal R^p_\\ell + \\sum_\\ell
  \\bar{\\mathcal R}^q_\\ell\\Big)$$

  (``elbo``), the Monte Carlo ELBO minus the residual variances under the model and under the recognition
  model, all library estimators with score-function gradients;
- the penalty strength $\\lambda$ and the seed.

By default the ELBO is Algorithm 1 of the article as written: fixed $\\lambda$ and Adam at a constant
learning rate; a ramp of $\\lambda$, a warmup and a cosine decay are options. Both fits use the same steps
and learning-rate schedule. Nothing
constrains the parameters. An update with a non-finite gradient is skipped and counted;
a run whose parameters become non-finite is stopped, and its last finite parameters are measured.

Each run is saved to ``runs/<key>/`` of the results directory, where ``<key>`` names the condition and a
hash of the training settings its objective depends on, and is skipped when it exists (``--rerun``
overrides this). The flags select a subset of the grid and the training settings, and every run of the
current settings and penalty strengths found under ``runs/`` is collected into ``analysis.json``.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from ..shared import ExamplePaths, example_paths, jax_cli
from .model import CanonicalCircuit, Couplings, Tied, canonical_circuit
from .types import Results, RunResult

type Fit = Literal["exact", "elbo"]

N_PIXELS, WIDTH, NOISE = 16, 0.08, 0.05
N_NEURONS = 10


# Data


def curve(ts: Array) -> Array:
    """Noiseless images of bumps at positions ``ts``."""
    centres = jnp.linspace(0.0, 1.0, N_PIXELS)
    return jnp.exp(-((ts[:, None] - centres[None, :]) ** 2) / (2 * WIDTH**2))


def curve_data(key: Array, n: int) -> tuple[Array, Array]:
    """Noisy images of bumps at uniform positions, with the positions."""
    k_t, k_e = jax.random.split(key)
    t = jax.random.uniform(k_t, (n,))
    return curve(t) + NOISE * jax.random.normal(k_e, (n, N_PIXELS)), t


def off_curve(xs: Array) -> Array:
    """Whether the mean squared distance of each point to the noiseless curve exceeds twice the noise variance."""
    ref = curve(jnp.linspace(0.0, 1.0, 1000))
    dists = jax.lax.map(
        lambda x: jnp.min(jnp.mean((ref - x) ** 2, axis=1)), xs, batch_size=256
    )
    return dists > 2 * NOISE**2


# Conditions and settings


@dataclass(frozen=True)
class Condition:
    """One cell of the grid."""

    couplings: Couplings
    use_z: bool
    fit: Fit
    lam: float
    seed: int

    def key(self, signature: str) -> str:
        z = "z" if self.use_z else "noz"
        fit = "exact" if self.fit == "exact" else f"elbo{self.lam:g}"
        return f"{self.couplings}_{z}_{fit}_s{self.seed}_{signature}"


@dataclass(frozen=True)
class Training:
    """Training settings; :meth:`signature` hashes those the objective of a fit depends on."""

    steps: int
    chunk: int
    batch: int
    lr: float
    warmup: int
    decay: float
    ramp: float
    n_samples: int
    hidden: tuple[int, ...]
    obs_sd: float
    z_range: float

    def signature(self, fit: Fit) -> str:
        fields = asdict(self)
        if fit == "exact":
            for name in ("ramp", "n_samples", "hidden"):
                fields.pop(name)
        return hashlib.sha1(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:8]


# Objectives


def elbo_objective(
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


# Measurements


def spearman(a: Array, b: Array) -> float:
    ra = jnp.argsort(jnp.argsort(a)).astype(float)
    rb = jnp.argsort(jnp.argsort(b)).astype(float)
    return float(jnp.corrcoef(ra, rb)[0, 1])


def posterior_moments_z(
    circuit: CanonicalCircuit, params: Array, xs: Array
) -> tuple[Array, Array]:
    """Exact posterior mean and variance of $z$ at each datapoint, by enumeration."""
    dep, lat = circuit.dep, circuit.lat_man

    def one(x: Array) -> Array:
        post = circuit.posterior_at(params, x)
        wts = jax.nn.softmax(circuit.state_log_weights(post))
        comps = jax.vmap(dep.gen_hrm.posterior_at, in_axes=(None, 0))(
            post, circuit.states
        )
        mean, second = jax.vmap(
            lambda p: lat.split_mean_second_moment(lat.to_mean(p))
        )(comps)
        m = wts @ mean[:, 0]
        return jnp.stack([m, wts @ second.reshape(second.shape[0], -1)[:, 0] - m**2])

    moments = jax.lax.map(one, xs, batch_size=50)
    return moments[:, 0], moments[:, 1]


def posterior_mean_z(circuit: CanonicalCircuit, params: Array, xs: Array) -> Array:
    return posterior_moments_z(circuit, params, xs)[0]


def z_usage(post_mean: Array, post_var: Array) -> float:
    """Fraction of the variance of $z$ explained by $x$: $\\mathrm{Var}(\\mathbb E[z \\mid x]) / (\\mathrm{Var}(\\mathbb E[z \\mid x]) + \\mathbb E[\\mathrm{Var}(z \\mid x)])$ over the test set, under the exact posterior; zero when $z$ is not used."""
    between = jnp.var(post_mean)
    return float(between / (between + jnp.mean(post_var)))


def bounds(
    circuit: CanonicalCircuit, params: Array, xs: Array, n_nodes: int
) -> dict[str, Array]:
    """Mean exact $\\log p(x)$, $\\log \\tilde p(x)$, ELBO and divergence of $q$ from the exact posterior."""
    ll = jax.lax.map(
        lambda x: circuit.exact_log_observable_density(params, x), xs, batch_size=50
    )
    log_tilde, elbo = jax.lax.map(
        lambda x: circuit.variational_bounds(params, x, n_nodes), xs, batch_size=50
    )
    kl = jax.lax.map(
        lambda x: circuit.recognition_divergence(params, x, n_nodes),
        xs,
        batch_size=50,
    )
    return {
        "test_ll": jnp.mean(ll),
        "log_tilde": jnp.mean(log_tilde),
        "elbo": jnp.mean(elbo),
        "kl_q": jnp.mean(kl),
    }


def sample_quality(
    circuit: CanonicalCircuit, key: Array, params: Array, n_show: int
) -> dict[str, Any]:
    """Mass off the curve of the harmonium and of $\\tilde p$, by enumeration, and means of sampled states.

    A state $n$ is off the curve when its mean $\\mathbb E[x \\mid n]$ is (:func:`off_curve`). Under the
    harmonium $p(n)$ is exact; under $\\tilde p$ it is $\\int \\tilde p(z) p(n \\mid z)$, by quadrature.
    """
    dep, lat, states = circuit.dep, circuit.lat_man, circuit.states
    log_w = circuit.exact_state_log_weights(params)
    means = jax.vmap(
        lambda n: circuit.obs_man.to_mean(circuit.likelihood_at(params, jnp.append(n, 0.0)))[
            : circuit.obs_dim
        ]
    )(states)
    off = off_curve(means)
    dep_prior = circuit.conjugated_prior_params(params)
    z_params, _ = dep.dep_man.split_coords(dep.conjugated_prior_params(dep_prior))
    loc, prc = lat.split_location_precision(z_params)
    var = 1.0 / lat.cov_man.to_matrix(prc)[0, 0]
    us, ws = np.polynomial.hermite_e.hermegauss(60)
    zs = var * loc[0] + jnp.sqrt(var) * jnp.asarray(us)
    p_tilde = jnp.asarray(ws / ws.sum()) @ jax.vmap(
        lambda z: jnp.exp(
            jax.vmap(dep.obs_man.log_density, in_axes=(None, 0))(
                dep.likelihood_at(dep_prior, z[None]), states
            )
        )
    )(zs)
    shown = jax.random.categorical(key, log_w, shape=(n_show,))
    return {
        "off_curve": float(jax.nn.softmax(log_w) @ off),
        "off_curve_tilde": float(p_tilde @ off),
        "sample_means": means[shown].tolist(),
    }


def tuning(circuit: CanonicalCircuit, params: Array) -> dict[str, Any]:
    """Firing probabilities $p(n_i = 1 \\mid z)$ of the generative model, couplings included, and the density of $\\tilde p(z)$."""
    dep, lat = circuit.dep, circuit.lat_man
    dep_prior = circuit.conjugated_prior_params(params)
    zs = jnp.linspace(-3.0, 3.0, 121)
    stats = jax.vmap(circuit.neurons.sufficient_statistic)(circuit.states)

    def rates(z: Array) -> Array:
        lkl = dep.likelihood_at(dep_prior, z[None])
        return jax.nn.softmax(stats @ lkl) @ circuit.states

    z_params, _ = dep.dep_man.split_coords(dep.conjugated_prior_params(dep_prior))
    dens = jax.vmap(lambda z: jnp.exp(lat.log_density(z_params, z[None])))(zs)
    return {
        "tuning_z": zs.tolist(),
        "tuning_rates": jax.vmap(rates)(zs).T.tolist(),
        "prior_density": dens.tolist(),
    }


# Training


def frozen_blocks(cond: Condition) -> set[str]:
    """Blocks held at their initial values: without $z$, $\\Theta_{NZ}$ and $\\rho_Z$ (zero)."""
    return set() if cond.use_z else {"nz_loc", "nz_prc", "cnj"}


def initialize(
    circuit: CanonicalCircuit, cond: Condition, cfg: Training, train_x: Array
) -> Tied:
    u = circuit.initialize_tied(
        jax.random.PRNGKey(cond.seed), train_x, cfg.z_range, cfg.obs_sd
    )
    if not cond.use_z:
        u = {
            **u,
            "nz_loc": jnp.zeros_like(u["nz_loc"]),
            "nz_prc": jnp.zeros_like(u["nz_prc"]),
        }
    return u


def train(
    circuit: CanonicalCircuit,
    cond: Condition,
    cfg: Training,
    train_x: Array,
    test_x: Array,
    test_t: Array,
) -> tuple[Tied, dict[str, list[float]], int | None, int]:
    """Train one condition, with a history of measurements on the test set after each chunk.

    An update whose gradient is not finite is skipped (``optax.apply_if_finite``), and the number of
    skipped steps is returned. Training stops if the parameters become non-finite, which happens after
    50 consecutive skipped steps.
    """
    n_train, n_nodes, n_eval_samples = train_x.shape[0], 40, 64
    frozen = frozen_blocks(cond)
    schedule = optax.join_schedules(
        [
            optax.linear_schedule(0.0, cfg.lr, cfg.warmup),
            optax.cosine_decay_schedule(cfg.lr, cfg.steps - cfg.warmup, cfg.decay),
        ],
        [cfg.warmup],
    )
    optimizer = optax.apply_if_finite(optax.adam(schedule), max_consecutive_errors=50)
    ramp = max(1.0, cfg.ramp * cfg.steps)

    def loss_fn(u: Tied, key: Array, xs: Array, lam: Array) -> Array:
        if cond.fit == "exact":
            return exact_objective(circuit, u, xs)
        loss, _ = elbo_objective(circuit, u, key, xs, lam, cfg.n_samples)
        return loss

    def step(
        carry: tuple[Tied, Any, Array, Array], _: None
    ) -> tuple[tuple[Tied, Any, Array, Array], Array]:
        u, opt_state, key, count = carry
        key, k_batch, k_obj = jax.random.split(key, 3)
        idx = jax.random.choice(k_batch, n_train, (cfg.batch,), replace=False)
        lam = cond.lam * jnp.minimum(1.0, count / ramp)
        loss, grads = jax.value_and_grad(loss_fn)(u, k_obj, train_x[idx], lam)
        grads = {k: jnp.zeros_like(g) if k in frozen else g for k, g in grads.items()}
        updates, opt_state = optimizer.update(grads, opt_state, u)
        return (optax.apply_updates(u, updates), opt_state, key, count + 1), loss  # pyright: ignore[reportReturnType]

    @jax.jit
    def chunk(
        carry: tuple[Tied, Any, Array, Array],
    ) -> tuple[tuple[Tied, Any, Array, Array], Array]:
        return jax.lax.scan(step, carry, None, cfg.chunk)

    @jax.jit
    def evaluate(u: Tied) -> tuple[dict[str, Array], Array]:
        params = circuit.tie(u)
        _, (_, var_q, var_p) = elbo_objective(
            circuit, u, jax.random.PRNGKey(1), test_x, jnp.array(0.0), n_eval_samples
        )
        return (
            {**bounds(circuit, params, test_x, n_nodes), "var_q": var_q, "var_p": var_p},
            posterior_mean_z(circuit, params, test_x),
        )

    u = initialize(circuit, cond, cfg, train_x)
    carry = (u, optimizer.init(u), jax.random.PRNGKey(1000 + cond.seed), jnp.array(0.0))
    history: dict[str, list[float]] = {}
    stopped: int | None = None
    for i in range(cfg.steps // cfg.chunk + 1):
        if i > 0:
            new_carry, _ = chunk(carry)
            finite = jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(lambda a: jnp.all(jnp.isfinite(a)), new_carry[0])
            )
            if not all(bool(f) for f in finite):
                stopped = (i - 1) * cfg.chunk
                print(f"  non-finite parameters in the chunk from step {stopped}", flush=True)
                break
            carry = new_carry
        vals, post_mean = evaluate(carry[0])
        record = {k: float(v) for k, v in vals.items()}
        record["corr_t"] = spearman(post_mean, test_t)
        for k, v in record.items():
            history.setdefault(k, []).append(v)
        print(
            f"  step {i * cfg.chunk}: log p {record['test_ll']:.3f}, log p~ {record['log_tilde']:.3f}, "
            f"elbo {record['elbo']:.3f}, var_q {record['var_q']:.2e}, var_p {record['var_p']:.2e}, "
            f"spearman {record['corr_t']:+.2f}",
            flush=True,
        )
    skipped = int(carry[1].total_notfinite)  # pyright: ignore[reportAttributeAccessIssue]
    if skipped:
        print(f"  skipped {skipped} steps with non-finite gradients", flush=True)
    return carry[0], history, stopped, skipped


def measure(
    circuit: CanonicalCircuit, u: Tied, test_x: Array, test_t: Array
) -> dict[str, Any]:
    """Final measurements on the test set."""
    params = circuit.tie(u)
    key = jax.random.PRNGKey(2)
    k_p, k_q, k_s = jax.random.split(key, 3)
    final = {k: float(v) for k, v in bounds(circuit, params, test_x, 40).items()}
    post_mean, post_var = posterior_moments_z(circuit, params, test_x)
    var_p = circuit.conjugation_residual_variances(k_p, params, 64)
    var_q = jax.lax.map(
        lambda kx: jnp.stack(
            circuit.recognition_residual_variances_at(kx[0], params, kx[1], 64)
        ),
        (jax.random.split(k_q, test_x.shape[0]), test_x),
        batch_size=50,
    )
    return {
        **{f"final_{k}": v for k, v in final.items()},
        "final_var_p_levels": [float(v) for v in var_p],
        "final_var_q_levels": [float(v) for v in jnp.mean(var_q, axis=0)],
        "final_corr_t": spearman(post_mean, test_t),
        "post_mean": post_mean.tolist(),
        "post_var": post_var.tolist(),
        "z_usage": z_usage(post_mean, post_var),
        **sample_quality(circuit, k_s, params, 300),
        **tuning(circuit, params),
    }


# Main


def run_paths(paths: ExamplePaths, key: str) -> ExamplePaths:
    return replace(paths, results_dir=paths.results_dir / "runs" / key)


def remeasure(paths: ExamplePaths) -> None:
    """Recompute ``post_mean``, ``post_var`` and ``z_usage`` of every cached run from its parameters."""
    _, k_test = jax.random.split(jax.random.PRNGKey(0))
    test_x, _ = curve_data(k_test, 500)
    runs_dir = paths.results_dir / "runs"
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        rp = run_paths(paths, run_dir.name)
        run = rp.load_analysis()
        circuit = canonical_circuit(
            N_PIXELS, N_NEURONS, run["couplings"], tuple(run["training"]["hidden"])
        )
        params = circuit.tie({k: jnp.asarray(v) for k, v in run["params"].items()})
        post_mean, post_var = posterior_moments_z(circuit, params, test_x)
        run.update(
            post_mean=post_mean.tolist(),
            post_var=post_var.tolist(),
            z_usage=z_usage(post_mean, post_var),
        )
        rp.save_analysis(run)
        print(f"{run['key']}: z usage {run['z_usage']:.3f}")


def summarize(paths: ExamplePaths, cfg: Training) -> None:
    """Print the final measurements of every cached run, with its settings where they differ from ``cfg``."""
    runs_dir = paths.results_dir / "runs"
    default = asdict(cfg)
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        run = run_paths(paths, run_dir.name).load_analysis()
        diff = {
            k: v
            for k, v in run["training"].items()
            if (list(v) if isinstance(v, list) else v)
            != (list(default[k]) if isinstance(default[k], tuple) else default[k])
        }
        print(
            f"{run['key'][:-9]:32s} log p {run['final_test_ll']:7.3f}  p~ {run['final_log_tilde']:7.3f}  "
            f"elbo {run['final_elbo']:7.3f}  usage {run.get('z_usage', float('nan')):.2f}  "
            f"off {run['off_curve']:.2f}  stopped {run['stopped']}  skipped {run.get('skipped', 0)}  {diff or ''}"
        )


def main() -> None:
    jax_cli()
    jax.config.update("jax_default_matmul_precision", "highest")
    parser = argparse.ArgumentParser()
    parser.add_argument("--couplings", nargs="+", default=["independent", "chain", "full"])
    parser.add_argument("--z", nargs="+", default=["with", "without"], choices=["with", "without"])
    parser.add_argument("--fits", nargs="+", default=["exact", "elbo"], choices=["exact", "elbo"])
    parser.add_argument("--lams", nargs="+", type=float, default=[0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rerun", action="store_true", help="rerun cached runs")
    parser.add_argument("--no-collect", action="store_true", help="do not rewrite analysis.json")
    parser.add_argument("--summary", action="store_true", help="print every cached run and exit")
    parser.add_argument("--remeasure", action="store_true", help="recompute the posterior moments of z of every cached run")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--decay", type=float, default=1.0, help="final fraction of the learning rate (1: constant)")
    parser.add_argument("--ramp", type=float, default=0.0, help="fraction of training over which lambda ramps up (0: fixed)")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--hidden", nargs="+", type=int, default=[32])
    parser.add_argument("--obs-sd", type=float, default=0.1)
    parser.add_argument("--z-range", type=float, default=2.0)
    args = parser.parse_known_args()[0]

    cfg = Training(
        steps=args.steps,
        chunk=args.chunk,
        batch=args.batch,
        lr=args.lr,
        warmup=args.warmup,
        decay=args.decay,
        ramp=args.ramp,
        n_samples=args.n_samples,
        hidden=tuple(args.hidden),
        obs_sd=args.obs_sd,
        z_range=args.z_range,
    )
    paths = example_paths(__file__)
    if args.summary:
        summarize(paths, cfg)
        return
    if args.remeasure:
        remeasure(paths)
        return
    k_train, k_test = jax.random.split(jax.random.PRNGKey(0))
    train_x, _ = curve_data(k_train, 2000)
    test_x, test_t = curve_data(k_test, 500)

    conditions = [
        Condition(couplings, z == "with", fit, lam if fit == "elbo" else 0.0, seed)
        for couplings in args.couplings
        for z in args.z
        for fit in args.fits
        for lam in (args.lams if fit == "elbo" else [0.0])
        for seed in args.seeds
        if z == "with" or fit == "exact"
    ]
    for cond in conditions:
        key = cond.key(cfg.signature(cond.fit))
        rp = run_paths(paths, key)
        if rp.analysis_path.exists() and not args.rerun:
            print(f"{key}: cached")
            continue
        print(f"{key}:", flush=True)
        circuit = canonical_circuit(N_PIXELS, N_NEURONS, cond.couplings, cfg.hidden)
        u, history, stopped, skipped = train(circuit, cond, cfg, train_x, test_x, test_t)
        run: RunResult = {
            "key": key,
            "couplings": cond.couplings,
            "use_z": cond.use_z,
            "fit": cond.fit,
            "lam": cond.lam,
            "seed": cond.seed,
            "training": asdict(cfg),
            "stopped": stopped,
            "skipped": skipped,
            "steps": [i * cfg.chunk for i in range(len(history["test_ll"]))],
            "history": history,
            "params": {k: v.tolist() for k, v in u.items()},
            **measure(circuit, u, test_x, test_t),
        }  # pyright: ignore[reportAssignmentType]
        rp.save_analysis(run)

    if args.no_collect:
        return

    # Collect every run of the current settings
    signatures = {fit: cfg.signature(fit) for fit in ("exact", "elbo")}
    runs: list[RunResult] = []
    runs_dir = paths.results_dir / "runs"
    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        run = run_paths(paths, run_dir.name).load_analysis()
        if run["key"].endswith(signatures[run["fit"]]) and (
            run["fit"] == "exact" or run["lam"] in args.lams
        ):
            runs.append(run)
    results: Results = {
        "training": asdict(cfg),
        "data_x": train_x[:300].tolist(),
        "test_t": test_t.tolist(),
        "runs": runs,
    }
    paths.save_analysis(results)
    print(f"collected {len(runs)} runs")


if __name__ == "__main__":
    main()
