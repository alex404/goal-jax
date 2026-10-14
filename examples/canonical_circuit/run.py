"""Fit a one-layer canonical circuit to bumps on a retina by maximizing the ELBO.

The data are images on a one-dimensional retina of 16 pixels on $[0, 1]$, each a Gaussian bump of width
$0.08$ at a uniformly distributed position $t$, with pixel noise of s.d. $0.05$. The circuit is the
approximate variant with one layer: a Gaussian observable with diagonal covariance, 16 binary neurons
with no couplings, and a one-dimensional latent $z$. With a diagonal covariance the correlations between
pixels must come from the neurons, and with neurons that are independent given $z$ the correlations
between neurons must come from $z$, so $z$ has to carry the position of the bump.

The circuit is trained by Adam on the Monte Carlo ELBO in its natural parameters, from
:meth:`~goal.geometry.exponential_family.variational.VariationalConjugated.initialize_from_sample` with
Glorot-initialized perceptrons. Reported: the test ELBO during training, against the log-likelihood of
Gaussians with diagonal and full covariance fit to the training set; the mean and s.d. of $q(z \\mid x)$
against the bump position; the tuning curves $p(n_i = 1 \\mid z)$; and samples of the circuit sorted by
$z$.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import BiasConjugationFunction, Diagonal, JunctionTree
from goal.models import (
    CanonicalCircuit,
    canonical_circuit,
    diagonal_normal,
    full_normal,
)

from ..shared import example_paths, jax_cli
from .types import Results

N_PIXELS, WIDTH, NOISE = 16, 0.08, 0.05
N_NEURONS = 16


def bumps(key: Array, n: int) -> tuple[Array, Array]:
    """Noisy images of bumps at uniform positions, with the positions."""
    k_t, k_e = jax.random.split(key)
    ts = jax.random.uniform(k_t, (n,))
    centres = jnp.linspace(0.0, 1.0, N_PIXELS)
    images = jnp.exp(-((ts[:, None] - centres[None, :]) ** 2) / (2 * WIDTH**2))
    return images + NOISE * jax.random.normal(k_e, images.shape), ts


def rank(xs: Array) -> Array:
    """Ranks of the entries of ``xs``, for the Spearman correlation."""
    return jnp.argsort(jnp.argsort(xs)).astype(float)


def initialize(circuit: CanonicalCircuit, key: Array, xs: Array) -> Array:
    """Harmonium parameters from the sample, with every perceptron Glorot-initialized and every constant conjugation function zero."""
    k_hrm, k_cnj = jax.random.split(key)
    hrm_params, _ = circuit.split_coords(circuit.initialize_from_sample(k_hrm, xs))
    cnj_fun_params: list[Array] = []
    for i, level in enumerate(circuit.lvl_mans):
        (cnj_fun,) = level.cnj_funs
        if not isinstance(cnj_fun, BiasConjugationFunction):
            cnj_fun_params.append(jnp.zeros(cnj_fun.dim))
            continue
        bia_map: Any = cnj_fun.bia_map
        cnj_fun_params.append(bia_map.glorot_initialize(jax.random.fold_in(k_cnj, i)))
    return circuit.join_coords(hrm_params, jnp.concatenate(cnj_fun_params))


def z_moments(circuit: CanonicalCircuit, dep_params: Array) -> tuple[Array, Array]:
    """Mean and variance of $z$ at a point of the deep model of the circuit."""
    pop = circuit.lvl_mans[1]
    z_params, _ = pop.dep_man.split_coords(pop.conjugated_prior_params(dep_params))
    loc, prc = pop.pst_man.split_location_precision(z_params)
    var = 1.0 / prc[0]
    return var * loc[0], var


def tuning(circuit: CanonicalCircuit, params: Array, zs: Array) -> Array:
    """Firing probability of each neuron given $z$ under the generative model, one row per point of ``zs``."""
    pop = circuit.lvl_mans[1]
    dep_prior = circuit.conjugated_prior_params(params)
    (pop_hrm,) = pop.gph_hrm.obs_hrms
    neurons = pop_hrm.obs_man

    def rates(z: Array) -> Array:
        diag, _ = neurons.split_couplings(
            neurons.to_mean(pop.likelihood_at(dep_prior, z[None]))
        )
        return diag

    return jax.vmap(rates)(zs)


def fit(
    circuit: CanonicalCircuit,
    params: Array,
    train_x: Array,
    test_x: Array,
    key: Array,
    steps: int,
    chunk: int,
    lr: float,
) -> tuple[Array, list[int], list[float]]:
    """Maximize the ELBO by Adam on minibatches, measuring the test ELBO after each chunk."""
    optimizer = optax.adam(optax.cosine_decay_schedule(lr, steps, 0.05))

    def step(carry: tuple[Array, Any, Array], _: None) -> tuple[Any, None]:
        params, opt_state, key = carry
        key, k_batch, k_elbo = jax.random.split(key, 3)
        idx = jax.random.choice(k_batch, train_x.shape[0], (64,), replace=False)
        grads = jax.grad(lambda p: -circuit.mean_elbo(k_elbo, p, train_x[idx], 8))(
            params
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, key), None

    run_chunk = jax.jit(lambda carry: jax.lax.scan(step, carry, None, chunk)[0])
    test_elbo = jax.jit(
        lambda p: circuit.mean_elbo(jax.random.PRNGKey(0), p, test_x, 64)
    )
    carry = (params, optimizer.init(params), key)
    history_steps, history_elbo = [0], [float(test_elbo(params))]
    for i in range(1, steps // chunk + 1):
        carry = run_chunk(carry)
        history_steps.append(i * chunk)
        history_elbo.append(float(test_elbo(carry[0])))
        print(f"  step {i * chunk}: test ELBO {history_elbo[-1]:.3f}", flush=True)
    return carry[0], history_steps, history_elbo


def main() -> None:
    jax_cli()
    paths = example_paths(__file__)
    k_train, k_test, k_init, k_fit, k_sample = jax.random.split(
        jax.random.PRNGKey(0), 5
    )
    train_x, _ = bumps(k_train, 2000)
    test_x, test_t = bumps(k_test, 200)

    # Gaussian baselines
    baselines: dict[str, float] = {}
    for name, nrm in (
        ("Gaussian, diagonal", diagonal_normal(N_PIXELS)),
        ("Gaussian, full", full_normal(N_PIXELS)),
    ):
        nrm_params = nrm.to_natural(nrm.average_sufficient_statistic(train_x))
        baselines[name] = float(nrm.average_log_density(nrm_params, test_x))

    # Circuit
    circuit = canonical_circuit(
        N_PIXELS,
        Diagonal(),
        (JunctionTree.from_edges(N_NEURONS, ()),),
        (1,),
        "approximate",
        (32,),
    )
    params = initialize(circuit, k_init, train_x)
    params, steps, elbos = fit(
        circuit, params, train_x, test_x, k_fit, steps=12000, chunk=500, lr=1e-2
    )

    # Recognition and tuning
    q_mean, q_var = jax.vmap(
        lambda x: z_moments(circuit, circuit.recognition_at(params, x))
    )(test_x)
    prior_mean, prior_var = z_moments(circuit, circuit.conjugated_prior_params(params))
    prior_sd = jnp.sqrt(prior_var)
    zs = jnp.linspace(prior_mean - 3 * prior_sd, prior_mean + 3 * prior_sd, 200)
    rates = tuning(circuit, params, zs)
    rho = float(jnp.corrcoef(rank(q_mean), rank(test_t))[0, 1])
    print(f"Spearman correlation of E_q[z | x] with the bump position: {rho:.3f}")

    # Samples, sorted by z
    samples = circuit.sample(k_sample, params, 40)
    samples = samples[jnp.argsort(samples[:, -1])]
    data_order = jnp.argsort(test_t[:40])

    paths.save_analysis(
        Results(
            steps=steps,
            elbo=elbos,
            baselines=baselines,
            test_t=test_t.tolist(),
            q_mean=q_mean.tolist(),
            q_sd=jnp.sqrt(q_var).tolist(),
            spearman=rho,
            tuning_z=zs.tolist(),
            tuning_rates=rates.T.tolist(),
            sample_images=samples[:, :N_PIXELS].tolist(),
            data_images=test_x[:40][data_order].tolist(),
        )
    )


if __name__ == "__main__":
    main()
