"""Fit the one-layer canonical circuit to a bump on a retina, in both variants, by the penalized ELBO.

The data are points of a one-dimensional retina of 16 pixels on $[0, 1]$ seeing a Gaussian bump of width
$0.08$ at a uniformly distributed position $t$, with pixel noise. The circuit has 8 neurons with
generative couplings on a chain and a one-dimensional $z$, and is trained in its generative coordinates
(:meth:`~goal.models.CanonicalCircuit.tie`). Three fits:

- **no z**: the partially exact circuit with $\\Theta_{NZ} = 0$ and $\\rho_Z = 0$. Both levels are then
  exactly conjugate, the ELBO is the log-likelihood, and the fit is the floor that $z$ must improve on;
- **partially exact**: dense neurons and the exact conjugation parameters of the $X - N$ level; only
  $\\rho_Z$ is learned. This is the baseline of the approximate variant;
- **approximate**: chordal neurons on the chain, with $\\rho_N$ and $\\rho_Z$ learned.

Each fit maximizes the penalized ELBO

$$\\mathcal L_\\lambda = \\hat{\\mathcal L} - \\lambda \\Big(\\sum_\\ell \\mathcal R^p_\\ell + \\sum_\\ell
\\bar{\\mathcal R}^q_\\ell\\Big),$$

the Monte Carlo ELBO minus the residual variances under the model and under the recognition model. With
$z$, a warm-up first trains only the conjugation functions, which start far from the conjugation
parameters of the initial model. Everything is measured exactly on the test set: the neurons by
enumeration and $z$ by Gauss-Hermite quadrature give $\\log \\tilde p(x)$ of the directed model, its ELBO,
and their gap, the divergence of the recognition model from the posterior of $\\tilde p$.
"""

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jax.scipy.special import logsumexp

from goal.geometry import PositiveDefinite
from goal.models import CanonicalCircuit, canonical_circuit

from ..shared import example_paths, jax_cli
from .types import FitResult, Results

N_PIXELS, WIDTH, NOISE = 16, 0.08, 0.05
N_NEURONS = 8
EDGES = tuple((i, i + 1) for i in range(N_NEURONS - 1))
STATES = jnp.array(list(itertools.product([0.0, 1.0], repeat=N_NEURONS)))


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


# Model


def build(variant: str) -> CanonicalCircuit:
    return canonical_circuit(
        N_PIXELS,
        PositiveDefinite(),
        (N_NEURONS,),
        (EDGES,),
        (1,),
        variant,  # pyright: ignore[reportArgumentType]
        (32,),
    )


def initialize(circuit: CanonicalCircuit, use_z: bool, key: Array) -> Array:
    """Generative coordinates from the geometry of the data: neuron $i$ draws a bump at position $c_i$, tuned to $z$ around $4 c_i - 2$.

    The positions $c_i$ tile $[0, 1]$, so the mean of $x$ given that only neuron $i$ fires is the bump
    at $c_i$, with pixel noise of s.d. $0.2$: the noise of the data would make the conjugation parameters
    of the $X - N$ level large, far from the zero at which the learned ones start. The tuning curves tile $[-2, 2]$, neighbours overlapping
    by one spacing $s$ (precision $1 / s^2$), and the generative logit peaks at $-1$, with no
    couplings. Every conjugation function starts at zero: the last layer of each perceptron is zero.
    Without $z$, $\\Theta_{NZ}$ is zero.
    """
    (gen_nrn,) = circuit.generative_neuron_mans
    obs_nrm = circuit.gb_hrm.obs_man
    prc = jnp.eye(N_PIXELS) / 0.2**2
    obs = obs_nrm.join_location_precision(
        jnp.zeros(N_PIXELS), obs_nrm.cov_man.from_matrix(prc)
    )
    (xn_map,) = circuit.gb_hrm.crs_maps
    xn = xn_map.from_matrix(curve(jnp.linspace(0.0, 1.0, N_NEURONS)).T / 0.2**2)
    preferred = jnp.linspace(-2.0, 2.0, N_NEURONS)
    width_prc = (N_NEURONS - 1) ** 2 / 4.0**2
    n_couplings = len(gen_nrn.junction_tree.edges)
    neurons = gen_nrn.join_couplings(
        -1.0 - 0.5 * width_prc * preferred**2, jnp.zeros(n_couplings)
    )
    scale = 1.0 if use_z else 0.0
    nz_loc = scale * width_prc * preferred
    nz_prc = jnp.full(N_NEURONS, scale * width_prc)
    *_, cnj_man = circuit.generative_man.elm_mans
    cnj = []
    for j, mlp in enumerate(cnj_man.elm_mans):  # pyright: ignore[reportAttributeAccessIssue]
        if mlp.dim == 0:
            cnj.append(jnp.zeros(0))
            continue
        last = mlp.layer_dims[-2] * mlp.layer_dims[-1] + mlp.layer_dims[-1]
        cnj.append(
            mlp.glorot_initialize(jax.random.fold_in(key, j)).at[-last:].set(0.0)
        )
    return circuit.generative_man.join_coords(
        obs, xn, neurons, nz_loc, nz_prc, jnp.concatenate(cnj)
    )


def trained_mask(circuit: CanonicalCircuit, blocks: set[int]) -> Array:
    """One for each generative coordinate in the blocks ``blocks`` (by index), zero elsewhere."""
    dims = [man.dim for man in circuit.generative_man.elm_mans]
    return jnp.concatenate(
        [jnp.full(dim, i in blocks, dtype=float) for i, dim in enumerate(dims)]
    )


def penalized_elbo(
    circuit: CanonicalCircuit, gen: Array, key: Array, xs: Array, lam: float
) -> Array:
    """The negative penalized ELBO."""
    params = circuit.tie(gen)
    k_elbo, k_p, k_q = jax.random.split(key, 3)
    elbo = circuit.mean_elbo(k_elbo, params, xs, 8)
    var_p = sum(circuit.conjugation_residual_variances(k_p, params, 8), start=0.0)
    var_q = sum(
        circuit.mean_recognition_residual_variances(k_q, params, xs, 8), start=0.0
    )
    return -(elbo - lam * (var_q + var_p))


# Exact measurements: the neurons by enumeration, z by quadrature


def z_marginal(circuit: CanonicalCircuit, dep_params: Array) -> Array:
    """Natural parameters of the marginal over $z$ of a point of the deep model: $\\theta_Z + \\rho_Z$ at its biases."""
    pop = circuit.deep
    z_params, _ = pop.dep_man.split_coords(pop.conjugated_prior_params(dep_params))
    return z_params


def gaussian_moments(circuit: CanonicalCircuit, z_params: Array) -> tuple[Array, Array]:
    """Mean and variance of the one-dimensional Gaussian over $z$ at natural parameters."""
    lat = circuit.deep.pop_hrm.pst_man
    loc, prc = lat.split_location_precision(z_params)
    var = 1.0 / lat.cov_man.to_matrix(prc)[0, 0]
    return var * loc[0], var


def variational_bounds(
    circuit: CanonicalCircuit, params: Array, x: Array, n_nodes: int
) -> tuple[Array, Array]:
    """$\\log \\tilde p(x)$ and the ELBO of the variational model, with $z$ by quadrature over $q(z \\mid x)$ and $n$ by enumeration.

    With $f$ the summed residual, $\\log \\tilde p(x) = c(x) + \\log \\mathbb E_q[e^f]$ and $\\mathcal
    L(x) = c(x) + \\mathbb E_q[f]$. The residual of the $X - N$ level depends on $n$ and that of the
    population code level on $z$, and $q(n, z \\mid x) = q(z \\mid x) p(n \\mid z, \\beta)$ with $\\beta$
    the recognition biases.
    """
    pop = circuit.deep
    dep_prior = circuit.conjugated_prior_params(params)
    dep_post = circuit.recognition_at(params, x)
    mean, var = gaussian_moments(circuit, z_marginal(circuit, dep_post))
    us, ws = np.polynomial.hermite_e.hermegauss(n_nodes)
    zs = (mean + jnp.sqrt(var) * jnp.asarray(us))[:, None]
    r_n = jax.vmap(lambda n: circuit.conjugation_residual(params, jnp.append(n, 0.0)))(
        STATES
    )

    def at_node(z: Array) -> tuple[Array, Array]:
        lkl = pop.likelihood_at(dep_post, z)
        log_pn = jax.vmap(pop.obs_man.log_density, in_axes=(None, 0))(lkl, STATES)
        r_z = pop.conjugation_residual(dep_prior, z) - pop.conjugation_residual(
            dep_post, z
        )
        return r_z + jnp.exp(log_pn) @ r_n, r_z + logsumexp(log_pn + r_n)

    means, log_means = jax.vmap(at_node)(zs)
    c = circuit.conjugation_baseline(params, x)
    log_ws = jnp.log(jnp.asarray(ws / ws.sum()))
    return c + logsumexp(log_ws + log_means), c + jnp.exp(log_ws) @ means


def measure(circuit: CanonicalCircuit, gen: Array, xs: Array) -> dict[str, Array]:
    """Mean $\\log \\tilde p(x)$ and ELBO on ``xs``, and the mean and variance of $q(z \\mid x)$ at each point."""
    params = circuit.tie(gen)
    log_tilde, elbo = jax.lax.map(
        lambda x: variational_bounds(circuit, params, x, 20), xs, batch_size=50
    )
    q_mean, q_var = jax.lax.map(
        lambda x: gaussian_moments(
            circuit, z_marginal(circuit, circuit.recognition_at(params, x))
        ),
        xs,
        batch_size=50,
    )
    return {
        "log_tilde": jnp.mean(log_tilde),
        "elbo": jnp.mean(elbo),
        "q_mean": q_mean,
        "q_var": q_var,
    }


def tuning(circuit: CanonicalCircuit, gen: Array) -> tuple[Array, Array]:
    """Firing probabilities $p(n_i = 1 \\mid z)$ of the generative model, couplings included, on a grid of $z$."""
    pop = circuit.deep
    dep_prior = circuit.conjugated_prior_params(circuit.tie(gen))
    zs = jnp.linspace(-3.0, 3.0, 121)

    def rates(z: Array) -> Array:
        lkl = pop.likelihood_at(dep_prior, z[None])
        log_pn = jax.vmap(pop.obs_man.log_density, in_axes=(None, 0))(lkl, STATES)
        return jnp.exp(log_pn) @ STATES

    return zs, jax.vmap(rates)(zs).T


def spearman(a: Array, b: Array) -> float:
    ra = jnp.argsort(jnp.argsort(a)).astype(float)
    rb = jnp.argsort(jnp.argsort(b)).astype(float)
    return float(jnp.corrcoef(ra, rb)[0, 1])


# Training


def fit(
    name: str,
    variant: str,
    use_z: bool,
    train_x: Array,
    test_x: Array,
    test_t: Array,
    steps: int,
    warmup: int,
    chunk: int,
    lam: float,
    lr: float,
) -> FitResult:
    """Warm up the conjugation functions (with $z$), then train every generative coordinate, measuring after each chunk."""
    circuit = build(variant)
    n_blocks = len(circuit.generative_man.elm_mans)
    cnj_block = n_blocks - 1
    frozen = set() if use_z else {3, 4, cnj_block}  # Theta_NZ and rho_Z without z
    all_mask = trained_mask(circuit, set(range(n_blocks)) - frozen)
    warmup_mask = trained_mask(circuit, {cnj_block})
    n_train = train_x.shape[0]

    def make_chunk(optimizer: Any, mask: Array, length: int) -> Any:
        def step(carry: tuple[Array, Any, Array], _: None) -> tuple[Any, None]:
            gen, opt_state, key = carry
            key, k_batch, k_obj = jax.random.split(key, 3)
            idx = jax.random.choice(k_batch, n_train, (64,), replace=False)
            grads = jax.grad(
                lambda g: penalized_elbo(circuit, g, k_obj, train_x[idx], lam)
            )(gen)
            updates, opt_state = optimizer.update(grads * mask, opt_state, gen)
            return (optax.apply_updates(gen, updates), opt_state, key), None

        return jax.jit(lambda carry: jax.lax.scan(step, carry, None, length)[0])

    evaluate = jax.jit(lambda gen: measure(circuit, gen, test_x))
    gen = initialize(circuit, use_z, jax.random.PRNGKey(0))
    if use_z and warmup > 0:
        w_opt = optax.adam(optax.cosine_decay_schedule(lr, warmup, 0.1))
        gen, *_ = make_chunk(w_opt, warmup_mask, warmup)(
            (gen, w_opt.init(gen), jax.random.PRNGKey(1))
        )
    optimizer = optax.adam(optax.cosine_decay_schedule(lr, steps, 0.1))
    run_chunk = make_chunk(optimizer, all_mask, chunk)
    carry = (gen, optimizer.init(gen), jax.random.PRNGKey(2))
    history: dict[str, list[float]] = {"step": [], "log_tilde": [], "elbo": []}
    vals = evaluate(carry[0])
    for i in range(steps // chunk + 1):
        if i > 0:
            carry = run_chunk(carry)
            vals = evaluate(carry[0])
        history["step"].append(i * chunk)
        history["log_tilde"].append(float(vals["log_tilde"]))
        history["elbo"].append(float(vals["elbo"]))
        print(
            f"  {name} step {i * chunk}: log p~ {history['log_tilde'][-1]:.3f}, "
            + f"elbo {history['elbo'][-1]:.3f}",
            flush=True,
        )
    q_mean, q_var = vals["q_mean"], vals["q_var"]
    explained = jnp.var(q_mean)
    zs, rates = tuning(circuit, carry[0])
    return FitResult(
        name=name,
        history=history,
        log_tilde=history["log_tilde"][-1],
        elbo=history["elbo"][-1],
        z_usage=float(explained / (explained + jnp.mean(q_var))) if use_z else 0.0,
        spearman_t=spearman(q_mean, test_t) if use_z else 0.0,
        q_mean=q_mean.tolist(),
        tuning_z=zs.tolist(),
        tuning_rates=rates.tolist(),
    )


def main() -> None:
    jax_cli()
    jax.config.update("jax_default_matmul_precision", "highest")
    paths = example_paths(__file__)
    k_train, k_test = jax.random.split(jax.random.PRNGKey(0))
    train_x, _ = curve_data(k_train, 2000)
    test_x, test_t = curve_data(k_test, 200)
    steps, warmup, chunk, lam, lr = 2000, 1000, 500, 0.01, 1e-2
    fits = [
        fit(
            name, variant, use_z, train_x, test_x, test_t, steps, warmup, chunk, lam, lr
        )
        for name, variant, use_z in (
            ("no z", "partially_exact", False),
            ("partially exact", "partially_exact", True),
            ("approximate", "approximate", True),
        )
    ]
    paths.save_analysis(Results(test_t=test_t.tolist(), fits=fits))


if __name__ == "__main__":
    main()
