"""WP2 test: iterate a chordal-Boltzmann spike layer with continuous variables.

Trains the three-level hierarchy ``p(x, y, z) = p(x|y) p(y|z) p(z)`` from
:mod:`.hierarchical` on a controlled continuous dataset, where

- ``x`` -- continuous observations (a mixture of Gaussians in ``R^D``),
- ``y`` -- a binary Boltzmann spike population (the middle layer under test),
- ``z`` -- a continuous Gaussian top latent.

This is one "iteration" of the grant's spike/continuous stack: continuous data
is encoded into spikes and back out of a continuous latent. The data comes from
an *independent* generator (not the model), so a good fit is a genuine result.

We compare three middle layers -- ``diagonal`` (independent spikes),
``chain`` (nearest-neighbour couplings), and ``chordal`` (a 2-D grid) -- on:

- held-out ELBO (vs the true-generator log-likelihood ceiling),
- the generative conjugation residual variances ``Var[r_Y]``, ``Var[r*_Z]``,
- the recognition inner-residual variance ``Var[r_inner]`` (MLP tractability),
- reconstruction MSE.

Run::

    uv run python -m examples.variational_mnist.hierarchical_experiment
"""

import json
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from goal.geometry import Diagonal  # noqa: E402
from goal.models import Normal  # noqa: E402

from ..shared import example_paths  # noqa: E402
from .hierarchical import (  # noqa: E402
    VariationalHierarchical,
    build_boltzmann_gaussian_hierarchy,
)

# Geometry
OBS_DIM = 10
N_MID = 16
TOP_DIM = 4
GRID_SIDE = 4  # 4x4 = 16 for the chordal grid

# Data: independent mixture of Gaussians in R^OBS_DIM
N_MODES = 6
MODE_SPREAD = 3.0
MODE_STD = 0.6
N_TRAIN = 4096
N_TEST = 1024

# Training
STEPS = 4000
BATCH = 128
LR = 2e-3
LR_WARMUP = 400  # linear LR warmup steps (stabilizes the early score-fn variance)
GRAD_CLIP = 1.0  # global-norm gradient clip
MC_SAMPLES = 4
CONJ_SAMPLES = 32
LAMBDA_GEN = 5.0  # weight on generative conjugation residual variance
LAMBDA_INNER = 1.0  # weight on recognition inner-residual variance
WARMUP = 1500  # linear ramp of LAMBDA_GEN
LOG_EVERY = 250
EVAL_SAMPLES = 8


def make_data(key: Array) -> tuple[Array, Array, float]:
    """Mixture of ``N_MODES`` Gaussians in ``R^OBS_DIM``; returns train, test, ceiling."""
    k_mean, k_w, k_train, k_test = jax.random.split(key, 4)
    means = MODE_SPREAD * jax.random.normal(k_mean, (N_MODES, OBS_DIM))
    weights = jax.nn.softmax(0.5 * jax.random.normal(k_w, (N_MODES,)))

    def sample(k: Array, m: int) -> Array:
        kc, kn = jax.random.split(k)
        comp = jax.random.choice(kc, N_MODES, (m,), p=weights)
        return means[comp] + MODE_STD * jax.random.normal(kn, (m, OBS_DIM))

    train, test = sample(k_train, N_TRAIN), sample(k_test, N_TEST)

    # Ceiling: mean held-out log-likelihood under the true generator.
    d2 = jnp.sum((test[:, None, :] - means[None, :, :]) ** 2, axis=-1)
    log_comp = (
        jnp.log(weights)[None]
        - 0.5 * d2 / MODE_STD**2
        - 0.5 * OBS_DIM * jnp.log(2 * jnp.pi * MODE_STD**2)
    )
    ceiling = float(jnp.mean(logsumexp(log_comp, axis=1)))
    return train, test, ceiling


def grid_edges(side: int) -> list[tuple[int, int]]:
    """4-neighbour edges on a ``side x side`` grid."""
    edges: list[tuple[int, int]] = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                edges.append((i, r * side + c + 1))
            if r + 1 < side:
                edges.append((i, (r + 1) * side + c))
    return edges


def build_model(kind: str) -> VariationalHierarchical:
    # Fixed-covariance Gaussian observable: Theta_XY drives only the mean, so the
    # covariance is a global parameter and cannot collapse per-sample.
    obs_man = Normal(OBS_DIM, Diagonal())
    common = dict(mlp_hidden=(64,), obs_location_only=True)
    if kind == "chordal":
        return build_boltzmann_gaussian_hierarchy(
            obs_man, N_MID, TOP_DIM, mid_kind="chordal",
            mid_edges=grid_edges(GRID_SIDE), **common,
        )
    return build_boltzmann_gaussian_hierarchy(
        obs_man, N_MID, TOP_DIM, mid_kind=kind, **common
    )


def reconstruct_mse(
    model: VariationalHierarchical, params: Array, xs: Array, key: Array
) -> float:
    """Posterior-mean reconstruction: E_q[y] through the likelihood mean."""
    keys = jax.random.split(key, xs.shape[0])

    def one(x: Array, k: Array) -> Array:
        ys, _ = model.sample_posterior(k, params, x, 16)
        _, lower_lkl, _ = model.split_coords(params)
        s_y = jax.vmap(model.mid_man.sufficient_statistic)(ys)
        x_nat = jax.vmap(lambda s: model.lower_hrm.lkl_fun_man(lower_lkl, s))(s_y)
        means = jax.vmap(model.obs_man.to_mean)(x_nat)[:, : model.obs_man.data_dim]
        return jnp.mean(means, axis=0)

    recon = jax.vmap(one)(xs, keys)
    return float(jnp.mean((xs - recon) ** 2))


def fit(kind: str, train: Array, test: Array, key: Array) -> dict[str, Any]:
    model = build_model(kind)
    k_init, k_train, k_eval, k_rec = jax.random.split(key, 4)
    params = model.initialize_from_sample(k_init, train, location=0.0, shape=0.3)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR, warmup_steps=LR_WARMUP,
        decay_steps=STEPS, end_value=0.0,
    )
    # clip_by_global_norm + apply_if_finite: bound each step and skip any batch
    # whose gradient is non-finite, so a single blown-up Normal likelihood can
    # never corrupt the parameters (the diagonal/chordal NaN failure mode).
    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(schedule)),
        max_consecutive_errors=100,
    )
    opt_state = optimizer.init(params)

    def loss_fn(p: Array, k: Array, batch: Array, gen_beta: Array) -> Array:
        ke, kc, ki = jax.random.split(k, 3)
        elbo = model.mean_elbo(ke, p, batch, MC_SAMPLES)
        gen = model.prior_conjugation_loss(kc, p, CONJ_SAMPLES)
        inner = model.mean_recognition_inner_loss(ki, p, batch, MC_SAMPLES)
        return -elbo + gen_beta * LAMBDA_GEN * gen + LAMBDA_INNER * inner

    def step(carry: tuple[Any, Any, Array], g: Array) -> tuple[Any, None]:
        p, opt_state, k = carry
        gen_beta = jnp.minimum(1.0, g / WARMUP)
        k, kb, kl = jax.random.split(k, 3)
        batch = train[jax.random.choice(kb, train.shape[0], (BATCH,))]
        _, grads = jax.value_and_grad(loss_fn)(p, kl, batch, gen_beta)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return (optax.apply_updates(p, updates), opt_state, k), None

    k_etr, k_ete, k_gtr = jax.random.split(k_eval, 3)
    carry = (params, opt_state, k_train)
    steps_log: list[int] = []
    etr_log: list[float] = []
    ete_log: list[float] = []
    gvar_log: list[float] = []
    for c in range(STEPS // LOG_EVERY):
        gs = jnp.arange(c * LOG_EVERY, (c + 1) * LOG_EVERY)
        carry, _ = jax.lax.scan(step, carry, gs)
        params = carry[0]
        etr = float(model.mean_elbo(k_etr, params, train[:512], EVAL_SAMPLES))
        ete = float(model.mean_elbo(k_ete, params, test[:512], EVAL_SAMPLES))
        gvar = float(model.prior_conjugation_loss(k_gtr, params, 256))
        steps_log.append((c + 1) * LOG_EVERY)
        etr_log.append(etr)
        ete_log.append(ete)
        gvar_log.append(gvar)
        print(f"  {kind:8s} step {steps_log[-1]:5d}  ELBO train {etr:8.3f}  "
              f"test {ete:8.3f}  Var[r_gen] {gvar:.4f}")

    # Final diagnostics
    kg, kin, krec = jax.random.split(k_rec, 3)
    gen_var = float(model.prior_conjugation_loss(kg, params, 512))
    inner_var = float(model.mean_recognition_inner_loss(kin, params, test[:256], 8))
    mse = reconstruct_mse(model, params, test[:256], krec)
    return {
        "kind": kind,
        "elbo_test": ete_log[-1],
        "elbo_train": etr_log[-1],
        "gen_var": gen_var,
        "inner_var": inner_var,
        "recon_mse": mse,
        "steps": steps_log,
        "elbo_train_traj": etr_log,
        "elbo_test_traj": ete_log,
        "gen_var_traj": gvar_log,
    }


def main() -> None:
    key = jax.random.PRNGKey(0)
    k_data, k_diag, k_chain, k_chord = jax.random.split(key, 4)
    train, test, ceiling = make_data(k_data)
    print(f"Data: {N_MODES}-mode MoG in R^{OBS_DIM}; ceiling (max mean log p) = {ceiling:.3f}")
    print(f"Hierarchy: X(Normal-{OBS_DIM}) <- Y(Boltzmann-{N_MID}) <- Z(Gaussian-{TOP_DIM})\n")

    results = []
    for kind, k in [("diagonal", k_diag), ("chain", k_chain), ("chordal", k_chord)]:
        print(f"--- middle = {kind} ---")
        results.append(fit(kind, train, test, k))
        print()

    print("=" * 78)
    print(f"{'middle':10s} {'ELBO test':>11s} {'gap-to-ceil':>12s} "
          f"{'Var[r_gen]':>11s} {'Var[r_inner]':>13s} {'recon MSE':>11s}")
    print("-" * 78)
    for r in results:
        print(f"{r['kind']:10s} {r['elbo_test']:11.3f} {ceiling - r['elbo_test']:12.3f} "
              f"{r['gen_var']:11.4f} {r['inner_var']:13.4f} {r['recon_mse']:11.4f}")
    print("=" * 78)

    results_dir = example_paths(__file__).results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "hierarchical_experiment_results.json"
    out.write_text(json.dumps({"ceiling": ceiling, "results": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
