"""Fit a one-layer canonical circuit to bumps on a retina by maximizing the ELBO.

The data are images on a one-dimensional retina of 16 pixels on $[0, 1]$, each a Gaussian bump of width
$0.08$ at a uniformly distributed position $t$, with pixel noise of s.d. $0.05$. The circuit is the
approximate variant with one layer: a Gaussian observable with diagonal covariance, 16 binary neurons
with no couplings, and a one-dimensional latent $z$. With a diagonal covariance the correlations between
pixels must come from the neurons, and with neurons that are independent given $z$ the correlations
between neurons must come from $z$, so $z$ has to carry the position of the bump.

The prior of $z$ is fixed at a standard normal, as the latent prior of factor analysis is: the
neurons see $z$ only through $(z, z^2)$, so an affine change of $z$ is absorbed by their biases and
couplings. Since the prior is $\\theta_Z + \\rho_Z$, with $\\rho_Z$ evaluated at the neural biases of
the prior, this sets $\\theta_Z$ to the standard normal minus $\\rho_Z$. Left free, the prior precision
of $z$ is a small difference of $\\theta_Z$ and the output of the perceptron, which an Adam step can
make negative.

The circuit is trained by Adam in its natural parameters, from
:meth:`~goal.geometry.exponential_family.variational.VariationalConjugated.initialize_from_sample` with
Glorot-initialized perceptrons, twice from the same initialization: on the Monte Carlo ELBO alone, and
on the ELBO minus $0.03$ times the variance of the conjugation residual of each level, as in the HMoG
example. Reported for each fit: the test ELBO during training, against the log-likelihood of
Gaussians with diagonal and full covariance fit to the training set; the s.d. of the conjugation
residual of each level under the model and the recognition model; the mean and s.d. of $q(z \\mid x)$
against the bump position; the tuning curves $p(n_i = 1 \\mid z)$; and the test images sorted by $t$
beside samples of the circuit sorted by $z$.
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
from .types import FitResults, Results


def bumps(
    key: Array, n: int, n_pixels: int, width: float, noise: float
) -> tuple[Array, Array]:
    """Noisy images of bumps at uniform positions, with the positions."""
    k_t, k_e = jax.random.split(key)
    ts = jax.random.uniform(k_t, (n,))
    centres = jnp.linspace(0.0, 1.0, n_pixels)
    images = jnp.exp(-((ts[:, None] - centres[None, :]) ** 2) / (2 * width**2))
    return images + noise * jax.random.normal(k_e, images.shape), ts


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


def with_standard_prior(circuit: CanonicalCircuit, params: Array) -> Array:
    """The circuit parameters with $\\theta_Z$ replaced so that the prior of $z$ is a standard normal."""
    pop = circuit.lvl_mans[1]
    hrm_params, cnj_fun_params = circuit.split_coords(params)
    obs_params, int_params, pop_params = circuit.gph_hrm.split_coords(hrm_params)
    nrn_params, nz_params, z_params = pop.gph_hrm.split_coords(pop_params)

    def with_z(z_params: Array) -> Array:
        pop_params = pop.gph_hrm.join_coords(nrn_params, nz_params, z_params)
        hrm_params = circuit.gph_hrm.join_coords(obs_params, int_params, pop_params)
        return circuit.join_coords(hrm_params, cnj_fun_params)

    z_man = pop.pst_man
    std_params = z_man.to_natural(
        z_man.join_mean_covariance(jnp.zeros(z_man.data_dim), jnp.ones(z_man.data_dim))
    )
    # rho_Z at the prior does not depend on theta_Z
    rho_z, _ = pop.dep_man.split_coords(
        pop.conjugated_prior_params(
            circuit.conjugated_prior_params(with_z(jnp.zeros_like(z_params)))
        )
    )
    return with_z(std_params - rho_z)


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


def residual_variance(level: Any, params: Array, key: Array, n: int) -> Array:
    """Variance of the conjugation residual of ``level`` at ``n`` samples of its deep model at the prior, which carry no gradient."""
    zs = level.dep_man.sample(
        key, jax.lax.stop_gradient(level.conjugated_prior_params(params)), n
    )
    return jnp.var(jax.vmap(lambda z: level.conjugation_residual(params, z))(zs))


def conjugation_penalty(
    circuit: CanonicalCircuit, key: Array, params: Array, xs: Array
) -> Array:
    """Residual variances that move the conjugation functions alone, as in the HMoG example.

    The variance of the residual of the X-N level under the model, and of the N-Z level under the
    model and under the recognition model at each image of ``xs``. The harmonium parameters carry no
    gradient, so the penalty fits $\\rho$ to the current harmonium and does not pull the harmonium
    toward conjugation.
    """
    pop = circuit.lvl_mans[1]
    k_n, k_z, k_q = jax.random.split(key, 3)
    hrm_params, cnj_fun_params = circuit.split_coords(params)
    _, rho_z_params = circuit.snd_man.split_coords(cnj_fun_params)
    fixed = jax.lax.stop_gradient(params)

    def with_rho_z(pop_params: Array) -> Array:
        pop_hrm_params, _ = pop.split_coords(jax.lax.stop_gradient(pop_params))
        return pop.join_coords(pop_hrm_params, rho_z_params)

    var_n = residual_variance(
        circuit,
        circuit.join_coords(jax.lax.stop_gradient(hrm_params), cnj_fun_params),
        k_n,
        64,
    )
    var_z = residual_variance(
        pop, with_rho_z(circuit.conjugated_prior_params(fixed)), k_z, 64
    )
    var_zq = jnp.mean(
        jax.vmap(
            lambda k, x: residual_variance(
                pop, with_rho_z(circuit.recognition_at(fixed, x)), k, 16
            )
        )(jax.random.split(k_q, xs.shape[0]), xs)
    )
    return var_n + var_z + var_zq


def fit(
    circuit: CanonicalCircuit,
    params: Array,
    train_x: Array,
    test_x: Array,
    key: Array,
    penalty_weight: float,
    steps: int,
    chunk: int,
    lr: float,
) -> tuple[Array, list[int], list[float], dict[str, list[float]]]:
    """Maximize the ELBO minus ``penalty_weight`` times :func:`conjugation_penalty` by Adam on minibatches, with the prior of $z$ standard normal.

    After each chunk it measures the test ELBO and the s.d. of the conjugation residual of each level,
    under the model and under the recognition model at 64 test images.
    """
    optimizer = optax.adam(optax.cosine_decay_schedule(lr, steps, 0.05))

    def loss(key: Array, p: Array) -> Array:
        k_batch, k_elbo, k_pen = jax.random.split(key, 3)
        idx = jax.random.choice(k_batch, train_x.shape[0], (64,), replace=False)
        params = with_standard_prior(circuit, p)
        elbo = circuit.mean_elbo(k_elbo, params, train_x[idx], 8)
        if penalty_weight == 0:
            return -elbo
        penalty = conjugation_penalty(circuit, k_pen, params, train_x[idx])
        return -elbo + penalty_weight * penalty

    def step(carry: tuple[Array, Any, Array], _: None) -> tuple[Any, None]:
        params, opt_state, key = carry
        key, k_loss = jax.random.split(key)
        grads = jax.grad(lambda p: loss(k_loss, p))(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, key), None

    run_chunk = jax.jit(lambda carry: jax.lax.scan(step, carry, None, chunk)[0])
    test_elbo = jax.jit(
        lambda p: circuit.mean_elbo(
            jax.random.PRNGKey(0), with_standard_prior(circuit, p), test_x, 64
        )
    )

    @jax.jit
    def residual_sds(p: Array) -> Array:
        params = with_standard_prior(circuit, p)
        key = jax.random.PRNGKey(1)
        model_vars = circuit.conjugation_residual_variances(key, params, 256)
        rec_vars = circuit.mean_recognition_residual_variances(
            key, params, test_x[:64], 64
        )
        return jnp.sqrt(jnp.array([*model_vars, *rec_vars]))

    names = (
        "X-N level, model",
        "N-Z level, model",
        "X-N level, recognition",
        "N-Z level, recognition",
    )
    carry = (params, optimizer.init(params), key)
    history_steps, history_elbo = [], []
    history_residual: dict[str, list[float]] = {name: [] for name in names}
    for i in range(steps // chunk + 1):
        if i > 0:
            carry = run_chunk(carry)
        history_steps.append(i * chunk)
        history_elbo.append(float(test_elbo(carry[0])))
        for name, sd in zip(names, residual_sds(carry[0]).tolist(), strict=True):
            history_residual[name].append(sd)
        print(f"  step {i * chunk}: test ELBO {history_elbo[-1]:.3f}", flush=True)
    return (
        with_standard_prior(circuit, carry[0]),
        history_steps,
        history_elbo,
        history_residual,
    )


def summarize(
    circuit: CanonicalCircuit,
    params: Array,
    test_x: Array,
    test_t: Array,
    key: Array,
    history: tuple[list[int], list[float], dict[str, list[float]]],
) -> FitResults:
    """Recognition, tuning curves and samples of a fit circuit, with its training history."""
    steps, elbos, residual_sds = history
    q_mean, q_var = jax.vmap(
        lambda x: z_moments(circuit, circuit.recognition_at(params, x))
    )(test_x)
    prior_mean, prior_var = z_moments(circuit, circuit.conjugated_prior_params(params))
    prior_sd = jnp.sqrt(prior_var)
    zs = jnp.linspace(prior_mean - 3 * prior_sd, prior_mean + 3 * prior_sd, 200)
    rates = tuning(circuit, params, zs)
    rho = float(jnp.corrcoef(rank(q_mean), rank(test_t))[0, 1])
    print(f"Spearman correlation of E_q[z | x] with the bump position: {rho:.3f}")
    samples = circuit.sample(key, params, test_x.shape[0])
    samples = samples[jnp.argsort(samples[:, -1])]
    return FitResults(
        steps=steps,
        elbo=elbos,
        residual_sds=residual_sds,
        q_mean=q_mean.tolist(),
        q_sd=jnp.sqrt(q_var).tolist(),
        spearman=rho,
        tuning_z=zs.tolist(),
        tuning_rates=rates.T.tolist(),
        sample_z=samples[:, -1].tolist(),
        sample_images=samples[:, : test_x.shape[1]].tolist(),
    )


def main() -> None:
    jax_cli()
    paths = example_paths(__file__)
    k_train, k_test, k_init, k_fit, k_sample = jax.random.split(
        jax.random.PRNGKey(0), 5
    )
    n_pixels, width, noise, n_neurons = 16, 0.08, 0.05, 16
    train_x, _ = bumps(k_train, 2000, n_pixels, width, noise)
    test_x, test_t = bumps(k_test, 200, n_pixels, width, noise)

    # Gaussian baselines
    baselines: dict[str, float] = {}
    for name, nrm in (
        ("Gaussian, diagonal", diagonal_normal(n_pixels)),
        ("Gaussian, full", full_normal(n_pixels)),
    ):
        nrm_params = nrm.to_natural(nrm.average_sufficient_statistic(train_x))
        baselines[name] = float(nrm.average_log_density(nrm_params, test_x))

    # Circuit, fit with and without the conjugation penalty from the same initialization
    circuit = canonical_circuit(
        n_pixels,
        Diagonal(),
        (JunctionTree.from_edges(n_neurons, ()),),
        (1,),
        "approximate",
        (32,),
    )
    init_params = initialize(circuit, k_init, train_x)
    fits: dict[str, FitResults] = {}
    for name, penalty_weight in (("ELBO", 0.0), ("ELBO and conjugation", 0.03)):
        print(f"{name}:")
        params, steps, elbos, residual_sds = fit(
            circuit,
            init_params,
            train_x,
            test_x,
            k_fit,
            penalty_weight,
            steps=30000,
            chunk=1000,
            lr=3e-3,
        )
        fits[name] = summarize(
            circuit, params, test_x, test_t, k_sample, (steps, elbos, residual_sds)
        )

    data_order = jnp.argsort(test_t)
    paths.save_analysis(
        Results(
            baselines=baselines,
            test_t=test_t.tolist(),
            data_t=test_t[data_order].tolist(),
            data_images=test_x[data_order].tolist(),
            fits=fits,
        )
    )


if __name__ == "__main__":
    main()
