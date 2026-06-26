"""Chordal Boltzmann PPC: conjugacy and data-learning.

A population code of correlated binary neurons --- a Boltzmann machine with
pairwise couplings on a chordal graph --- encodes a continuous Gaussian latent.
The model is a real ``VariationalConjugated`` harmonium
(:class:`~goal.models.harmonium.population_codes.BoltzmannPopulationCode`), so we
can ask the two questions that matter together:

1. **Conjugacy.** Is the Boltzmann log-partition affine in the latent sufficient
   statistics, ``psi_G(theta_N + Theta . s_X(z)) ~= rho . s_X(z) + chi``? Fitting
   ``rho`` by least squares drives the conjugation residual to near zero, and a
   Bayesian-decoding check confirms the code still carries latent information.

2. **Data-learning.** Trained on data from an *independent* Bernoulli mixture
   (not the model itself, so the test isn't circular) by maximizing the ELBO with
   the conjugation residual regularized, does good fit co-exist with conjugacy?

The independent-Bernoulli ``diagonal`` code is the baseline the ``chordal`` code
is compared against throughout.
"""

from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.scipy.special import logsumexp

from goal.geometry.exponential_family.variational import conjugation_metrics
from goal.models import (
    BoltzmannPopulationCode,
    chordal_boltzmann_population_code,
    diagonal_boltzmann_population_code,
)

from ..shared import example_paths, jax_cli
from .types import FamilyResult, Results

# Geometry
SIDE = 3  # SIDE x SIDE neurons on a 2D grid
LAT_DIM = 2  # full-covariance Gaussian latent
LATENT_STD = 1.0  # std of the latent prior used for decoding

# Data: an independent Bernoulli mixture with spatially-local clusters
N_COMPONENTS = 4
BLOB_WIDTH = 1.0  # width of each cluster's activation blob over the grid
P_HI = 0.9  # peak within-cluster firing probability
P_LO = 0.05  # baseline firing probability
N_TRAIN = 2048
N_TEST = 512

# Training: ELBO - LAMBDA * Var_p[r]. rho is learned by gradient throughout (no
# closed-form regression in the model path --- that is a diagnostic only).
STEPS = 5000
BATCH = 128
LEARNING_RATE = 1e-2  # cosine-decayed to 0 over training so conjugation settles
MC_SAMPLES = 8  # Monte-Carlo z-samples for the ELBO
CONJ_SAMPLES = 64  # prior z-samples for the conjugation regularizer
LAMBDA = 20.0  # final conjugation-regularizer weight
CONJ_WARMUP = 1500  # steps to linearly ramp the conjugation weight 0 -> LAMBDA
LOG_EVERY = 100
METRIC_SAMPLES = 2000  # prior samples for conjugation_metrics
CORR_SAMPLES = 4000  # model samples for the correlation comparison

# Diagnostics
Z_GRID = 81  # points along the 1D latent sweep for the conjugacy curves
DECODE_GRID = 61  # grid for exact posterior decoding
N_DECODE = 300  # test latents for the decoding check


def grid_locations_2d(side: int) -> tuple[Array, list[tuple[int, int]]]:
    """``side x side`` neurons on a 2D grid plus 4-neighbour adjacency edges."""
    coords = jnp.linspace(-1.6, 1.6, side)
    pref = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1).reshape(-1, 2)
    edges: list[tuple[int, int]] = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                edges.append((i, r * side + c + 1))
            if r + 1 < side:
                edges.append((i, (r + 1) * side + c))
    return pref, edges


# --- Data: independent Bernoulli mixture ---------------------------------


def make_data(key: Array, pref: Array, n: int) -> tuple[Array, Array, Array, float]:
    """Spatially-structured Bernoulli mixture: train, held-out, corr, and ceiling.

    Each of ``N_COMPONENTS`` clusters lights up a Gaussian blob of neurons centred
    at a random grid point; neurons fire independently within a cluster, so the
    marginal pairwise correlations are *local* --- the regime where a chordal code
    can capture structure a diagonal one cannot. The model never sees its own
    samples, so a good fit is a genuine result. ``ceiling`` is the held-out mean
    log-likelihood under the true generator --- the best ELBO any model can reach.
    """
    k_centers, k_w, k_train, k_test = jax.random.split(key, 4)
    lo, hi = float(pref.min()), float(pref.max())
    centers = jax.random.uniform(k_centers, (N_COMPONENTS, pref.shape[1]), minval=lo, maxval=hi)
    d2 = jnp.sum((pref[None, :, :] - centers[:, None, :]) ** 2, axis=-1)
    probs = P_LO + (P_HI - P_LO) * jnp.exp(-d2 / (2 * BLOB_WIDTH**2))
    weights = jax.nn.softmax(0.5 * jax.random.normal(k_w, (N_COMPONENTS,)))

    def sample(k: Array, m: int) -> Array:
        kc, kb = jax.random.split(k)
        comp = jax.random.choice(kc, N_COMPONENTS, (m,), p=weights)
        return (jax.random.uniform(kb, (m, n)) < probs[comp]).astype(jnp.float64)

    train, test = sample(k_train, N_TRAIN), sample(k_test, N_TEST)
    log_comp = test[:, None, :] * jnp.log(probs)[None] + (1 - test[:, None, :]) * jnp.log1p(
        -probs
    )[None]
    ceiling = float(jnp.mean(logsumexp(jnp.log(weights)[None] + log_comp.sum(-1), axis=1)))
    return train, test, corr_of(train), ceiling


def corr_of(binary: Array) -> Array:
    """Pearson correlation matrix of binary samples (std floored for safety)."""
    centered = binary - jnp.mean(binary, axis=0)
    cov = (centered.T @ centered) / binary.shape[0]
    std = jnp.sqrt(jnp.maximum(jnp.diag(cov), 1e-8))
    return cov / jnp.outer(std, std)


def offdiag_rmse(a: Array, b: Array) -> float:
    """RMSE between the off-diagonal (pairwise) entries of two corr matrices."""
    iu = jnp.triu_indices(a.shape[0], 1)
    return float(jnp.sqrt(jnp.mean((a[iu] - b[iu]) ** 2)))


# --- Conjugacy diagnostics -----------------------------------------------


def conjugacy_curves(
    model: BoltzmannPopulationCode[Any],
    params0: Array,
    params: Array,
    z_pts: Array,
) -> tuple[Array, Array, Array, Array]:
    """Log-partition, its affine fit, and the residual (rho=0 vs fitted) on a sweep.

    The residual is ``affine - psi``, so ``affine = psi + residual``.
    """
    psi = jax.vmap(
        lambda z: model.obs_man.log_partition_function(model.likelihood_at(params, z))
    )(z_pts)
    r_after = jax.vmap(lambda z: model.conjugation_residual(params, z))(z_pts)
    r_before = jax.vmap(lambda z: model.conjugation_residual(params0, z))(z_pts)
    return psi, psi + r_after, r_before, r_after


def decode_rmse(
    model: BoltzmannPopulationCode[Any],
    params: Array,
    x_stars: Array,
    key: Array,
    grid: Array,
    log_prior: Array,
    lesion: bool,
) -> float:
    """Exact grid-posterior decoding RMSE; ``lesion`` freezes eta_N(x):=eta_N(0)."""
    eta_at = lambda z: model.likelihood_at(params, z)  # noqa: E731
    eta0 = eta_at(jnp.zeros(x_stars.shape[1]))
    eta_grid = jax.vmap(eta_at)(grid)
    psi_grid = jax.vmap(model.obs_man.log_partition_function)(eta_grid)
    keys = jax.random.split(key, x_stars.shape[0])

    def one(x_star: Array, k: Array) -> Array:
        eta_star = eta0 if lesion else eta_at(x_star)
        n = model.obs_man.sample(k, eta_star, n=1)[0]
        s_n = model.obs_man.sufficient_statistic(n)
        return jax.nn.softmax(log_prior + (eta_grid @ s_n - psi_grid)) @ grid

    est = jax.vmap(one)(x_stars, keys)
    return float(jnp.sqrt(jnp.mean((est - x_stars) ** 2)))


# --- Per-family pipeline -------------------------------------------------


def fit_family(
    model: BoltzmannPopulationCode[Any],
    name: str,
    train: Array,
    test: Array,
    data_corr: Array,
    z_pts: Array,
    grid: Array,
    log_prior: Array,
    x_stars: Array,
    key: Array,
) -> FamilyResult:
    """Fit one population code and collect every diagnostic for it."""
    k_init, k_train, k_eval, k_dec, k_corr = jax.random.split(key, 5)

    # Start from rho = 0 and learn it by gradient (no regression seeding).
    params = model.initialize(k_init, location=0.0, shape=0.3)
    params_init = params

    # Train: maximize ELBO with the conjugation residual regularized. Cosine LR
    # decay shrinks the late steps so the model settles into the conjugation
    # optimum instead of jittering around it.
    optimizer = optax.adam(optax.cosine_decay_schedule(LEARNING_RATE, STEPS))
    opt_state = optimizer.init(params)

    def loss_fn(params: Array, key: Array, batch: Array, conj_beta: Array) -> Array:
        e_key, c_key = jax.random.split(key)
        elbo = model.mean_elbo(e_key, params, batch, MC_SAMPLES)
        conj = model.prior_conjugation_loss(c_key, params, CONJ_SAMPLES)
        return -elbo + conj_beta * LAMBDA * conj

    def step(carry: tuple[Any, Any, Array], gstep: Array) -> tuple[tuple[Any, Any, Array], None]:
        params, opt_state, k = carry
        conj_beta = jnp.minimum(1.0, gstep / CONJ_WARMUP)  # linear ramp 0 -> 1
        k, b_key, l_key = jax.random.split(k, 3)
        batch = train[jax.random.choice(b_key, train.shape[0], (BATCH,))]
        _, grads = jax.value_and_grad(loss_fn)(params, l_key, batch, conj_beta)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, k), None

    elbo_train: list[float] = []
    elbo_test: list[float] = []
    conj_r2: list[float] = []
    # fixed eval keys so the logged curves track the model, not resampling noise
    k_etr, k_ete, k_r2 = jax.random.split(k_eval, 3)
    carry = (params, opt_state, k_train)
    for c in range(STEPS // LOG_EVERY):
        gsteps = jnp.arange(c * LOG_EVERY, (c + 1) * LOG_EVERY)
        carry, _ = jax.lax.scan(step, carry, gsteps)
        params = carry[0]
        elbo_train.append(float(model.mean_elbo(k_etr, params, train[:512], MC_SAMPLES)))
        elbo_test.append(float(model.mean_elbo(k_ete, params, test, MC_SAMPLES)))
        _, _, r2 = conjugation_metrics(model, k_r2, params, METRIC_SAMPLES)
        conj_r2.append(float(r2))

    # Conjugacy curves: the trained model's log-partition vs its affine fit, and
    # the residual before training (rho=0) vs after training.
    psi, affine, r_before, r_after = conjugacy_curves(model, params_init, params, z_pts)

    # Information-content check + learned correlations on the trained model
    d_rmse = decode_rmse(model, params, x_stars, k_dec, grid, log_prior, False)
    l_rmse = decode_rmse(model, params, x_stars, k_dec, grid, log_prior, True)
    model_corr = corr_of(model.sample(k_corr, params, CORR_SAMPLES)[:, : model.n_neurons])

    print(
        f"  {name:9s} | ELBO {elbo_train[0]:.3f}->{elbo_train[-1]:.3f} (test {elbo_test[-1]:.3f})"
        + f" | conj R^2 {conj_r2[0]:.3f}->{conj_r2[-1]:.3f}"
        + f" | decode {d_rmse:.3f} (lesion {l_rmse:.3f})"
        + f" | corr-RMSE {offdiag_rmse(model_corr, data_corr):.3f}"
    )
    return FamilyResult(
        name=name,
        psi_curve=psi.tolist(),
        affine_curve=affine.tolist(),
        residual_before=r_before.tolist(),
        residual_after=r_after.tolist(),
        decode_rmse=d_rmse,
        lesion_rmse=l_rmse,
        elbo_train=elbo_train,
        elbo_test=elbo_test,
        conj_r2=conj_r2,
        model_corr=model_corr.tolist(),
        corr_offdiag_rmse=offdiag_rmse(model_corr, data_corr),
    )


def main() -> None:
    jax_cli()
    jax.config.update("jax_enable_x64", True)
    paths = example_paths(__file__)

    pref, edges = grid_locations_2d(SIDE)
    n = pref.shape[0]
    key = jax.random.PRNGKey(0)
    k_data, k_chord, k_diag = jax.random.split(key, 3)
    train, test, data_corr, ceiling = make_data(k_data, pref, n)
    print(f"data ceiling (max mean log p) = {ceiling:.3f}")

    # shared diagnostic grids: a 1D latent sweep, a decoding grid + prior + tests
    z_pts = jnp.stack([jnp.linspace(-3.0, 3.0, Z_GRID), jnp.zeros(Z_GRID)], axis=-1)
    daxis = jnp.linspace(-3.0, 3.0, DECODE_GRID)
    dgx, dgy = jnp.meshgrid(daxis, daxis, indexing="ij")
    grid = jnp.stack([dgx.ravel(), dgy.ravel()], axis=-1)
    log_prior = -0.5 * jnp.sum(grid**2, axis=1) / LATENT_STD**2
    x_stars = LATENT_STD * jax.random.normal(jax.random.PRNGKey(1), (N_DECODE, LAT_DIM))

    models: dict[str, BoltzmannPopulationCode[Any]] = {
        "chordal": chordal_boltzmann_population_code(n, edges, LAT_DIM),
        "diagonal": diagonal_boltzmann_population_code(n, LAT_DIM),
    }
    fam_keys = {"chordal": k_chord, "diagonal": k_diag}

    print("Fitting Boltzmann population codes to Bernoulli-mixture data:")
    families = {
        name: fit_family(
            model, name, train, test, data_corr, z_pts, grid, log_prior,
            x_stars, fam_keys[name],
        )
        for name, model in models.items()
    }

    paths.save_analysis(
        Results(
            n_neurons=n,
            n_latent=LAT_DIM,
            n_components=N_COMPONENTS,
            lambda_conj=float(LAMBDA),
            conj_warmup=CONJ_WARMUP,
            prior_std=float(LATENT_STD),
            ceiling=ceiling,
            z_grid=z_pts[:, 0].tolist(),
            steps=[(c + 1) * LOG_EVERY for c in range(STEPS // LOG_EVERY)],
            data_corr=data_corr.tolist(),
            families=families,
        )
    )


if __name__ == "__main__":
    main()
