"""Clustering test for the mixture-top hierarchy on labelled synthetic data.

Generates a mixture of Gaussians in ``R^D`` with *known* mode labels, fits the
four-level model ``p(x, y, z, k)`` (X continuous, Y Boltzmann spikes, Z Gaussian,
K cluster), and measures how well the inferred clusters ``k`` recover the true
modes -- unsupervised, via NMI and purity against the held-out labels.

The point: the mixture top should discover the generative modes *through* the
spike bottleneck, not just fit density. We compare the three middle-layer
connectivities (diagonal / chain / chordal).

Run::

    uv run python -m examples.variational_mnist.hierarchical_mixture_experiment
"""

import json
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

jax.config.update("jax_enable_x64", True)

from goal.geometry import Diagonal  # noqa: E402
from goal.models import Normal  # noqa: E402

from ..shared import example_paths  # noqa: E402
from .hierarchical_mixture import (  # noqa: E402
    VariationalHierarchicalMixture,
    build_boltzmann_gaussian_mixture_hierarchy,
)

# Geometry
OBS_DIM = 10
N_MID = 16
TOP_DIM = 4
GRID_SIDE = 4

# Data
N_MODES = 6  # true clusters == mixture components
MODE_SPREAD = 3.0
MODE_STD = 0.6
N_TRAIN = 4096
N_TEST = 1024

# Training
STEPS = 4000
BATCH = 128
LR = 2e-3
LR_WARMUP = 600
GRAD_CLIP = 1.0
MC_SAMPLES = 4
CONJ_SAMPLES = 32
LAMBDA_GEN = 5.0
LAMBDA_INNER = 1.0
# Delayed conjugation ramp: hold lambda_gen=0 for CONJ_HOLD steps (let the ELBO
# shape the latent into a mode-reflecting spread first), then ramp over CONJ_RAMP.
CONJ_HOLD = 1500
CONJ_RAMP = 1000
LOG_EVERY = 500
RESP_SAMPLES = 32  # z-samples per x for responsibilities

# Mixture-prior stabilization (ported from goal-apps hmog trainer). Gradient
# training of the top mixture drifts component precisions out of the PD cone
# (-> NaN) and lets components die (-> cluster collapse); these bound/regularize
# the mixture the way the HMoG EM trainer bounds its posterior statistics.
MIN_PROB = 1e-4  # floor on mixing weights (keep components alive)
# z here is unit-scale (not whitened as in the HMoG trainer), so the component
# variance floor must be O(1), not the HMoG default 1e-6: too small a floor lets
# a component sharpen to a near-spike, blowing up log p_mix(z) and the
# score-function ELBO term (the NaN at step ~1500).
LAT_MIN_VAR = 5e-2  # floor on component variances
LAT_JITTER_VAR = 1e-4  # diagonal jitter added to component covariances (keep PD)
ENT_REG = 0.3  # weight on mixture-weight entropy (push toward uniform; anti-collapse)


def make_data(key: Array) -> tuple[Array, Array, Array, Array]:
    """MoG in ``R^OBS_DIM`` with labels; returns train_x, train_y, test_x, test_y."""
    k_mean, k_w, k_train, k_test = jax.random.split(key, 4)
    means = MODE_SPREAD * jax.random.normal(k_mean, (N_MODES, OBS_DIM))
    weights = jax.nn.softmax(0.5 * jax.random.normal(k_w, (N_MODES,)))

    def sample(k: Array, m: int) -> tuple[Array, Array]:
        kc, kn = jax.random.split(k)
        comp = jax.random.choice(kc, N_MODES, (m,), p=weights)
        x = means[comp] + MODE_STD * jax.random.normal(kn, (m, OBS_DIM))
        return x, comp

    train_x, train_y = sample(k_train, N_TRAIN)
    test_x, test_y = sample(k_test, N_TEST)
    return train_x, train_y, test_x, test_y


def grid_edges(side: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                edges.append((i, r * side + c + 1))
            if r + 1 < side:
                edges.append((i, (r + 1) * side + c))
    return edges


def build_model(kind: str) -> VariationalHierarchicalMixture:
    obs_man = Normal(OBS_DIM, Diagonal())
    common = dict(mlp_hidden=(64,), obs_location_only=True)
    if kind == "chordal":
        return build_boltzmann_gaussian_mixture_hierarchy(
            obs_man, N_MID, TOP_DIM, N_MODES, mid_kind="chordal",
            mid_edges=grid_edges(GRID_SIDE), **common,
        )
    return build_boltzmann_gaussian_mixture_hierarchy(
        obs_man, N_MID, TOP_DIM, N_MODES, mid_kind=kind, **common
    )


def bound_mixture(model: VariationalHierarchicalMixture, params: Array) -> Array:
    """Project the top mixture prior back to a safe region (post-step, mirrors HMoG).

    Works in mean coordinates: floor+jitter each component covariance (keeps the
    Gaussians PD) and clip the mixing weights to ``[MIN_PROB, 1]`` (keeps
    components alive), then map back to natural parameters and write the mixture
    block back into ``params``. Everything else in ``params`` is untouched.
    """
    uh = model.top_prior
    mix_means = uh.to_mean(model.split_mixture(params))
    comp_means, prob_means = uh.split_mean_mixture(mix_means)
    comp_means = uh.cmp_man.map(
        lambda c: uh.obs_man.regularize_covariance(c, LAT_JITTER_VAR, LAT_MIN_VAR),
        comp_means,
        flatten=True,
    )
    probs = uh.lat_man.to_probs(prob_means)
    probs = jnp.clip(probs, MIN_PROB, 1.0)
    probs = probs / jnp.sum(probs)
    bounded = uh.join_mean_mixture(comp_means, uh.lat_man.from_probs(probs))
    new_mix = uh.to_natural(bounded)

    top, lower_lkl, third = model.split_coords(params)
    recog_params, _ = model.trd_man.split_coords(third)
    new_third = model.trd_man.join_coords(recog_params, new_mix)
    return model.join_coords(top, lower_lkl, new_third)


def mixture_entropy_penalty(model: VariationalHierarchicalMixture, params: Array) -> Array:
    """``-H(pi)`` of the mixture weights via ``dual_potential`` (no ``log 0`` on dying components).

    Added to the loss with weight ``ENT_REG``; minimizing it maximizes the mixing
    entropy, pushing the weights toward uniform so clusters do not collapse.
    """
    _, cat_nat = model.top_prior.split_natural_mixture(model.split_mixture(params))
    return model.top_prior.lat_man.dual_potential(cat_nat)


def cluster_metrics(pred: np.ndarray, true: np.ndarray, n_clusters: int) -> tuple[float, float]:
    """(NMI, purity) between predicted clusters and true labels."""
    classes = np.unique(true)
    cont = np.zeros((n_clusters, classes.size))
    for p, t in zip(pred, true):
        cont[p, np.searchsorted(classes, t)] += 1
    n = pred.size
    purity = cont.max(axis=1).sum() / n

    p_k = cont.sum(1) / n
    p_c = cont.sum(0) / n
    p_kc = cont / n
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(p_kc * np.log(p_kc / (p_k[:, None] * p_c[None, :] + 1e-12) + 1e-12))
        h_k = -np.nansum(p_k * np.log(p_k + 1e-12))
        h_c = -np.nansum(p_c * np.log(p_c + 1e-12))
    # If the assignment collapses to one cluster, H(clusters)=0 and NMI is 0
    # (no information), not undefined.
    denom = np.sqrt(max(h_k, 0.0) * max(h_c, 0.0))
    nmi = float(mi / denom) if denom > 1e-9 else 0.0
    return nmi, float(purity)


def evaluate_clusters(
    model: VariationalHierarchicalMixture, params: Array, xs: Array, ys: Array, key: Array
) -> tuple[float, float]:
    pred = np.array(model.cluster_assignments(key, params, xs, RESP_SAMPLES))
    return cluster_metrics(pred, np.array(ys), model.n_clusters)


def fit(kind: str, train_x: Array, test_x: Array, test_y: Array, key: Array) -> dict[str, Any]:
    model = build_model(kind)
    k_init, k_train, k_eval = jax.random.split(key, 3)
    params = model.initialize_from_sample(k_init, train_x, location=0.0, shape=0.3)
    params = bound_mixture(model, params)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR, warmup_steps=LR_WARMUP,
        decay_steps=STEPS, end_value=0.0,
    )
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
        ent = mixture_entropy_penalty(model, p)
        return -elbo + gen_beta * LAMBDA_GEN * gen + LAMBDA_INNER * inner + ENT_REG * ent

    def step(carry: tuple[Any, Any, Array], g: Array) -> tuple[Any, None]:
        p, opt_state, k = carry
        gen_beta = jnp.clip((g - CONJ_HOLD) / CONJ_RAMP, 0.0, 1.0)
        k, kb, kl = jax.random.split(k, 3)
        batch = train_x[jax.random.choice(kb, train_x.shape[0], (BATCH,))]
        _, grads = jax.value_and_grad(loss_fn)(p, kl, batch, gen_beta)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = optax.apply_updates(p, updates)
        p = bound_mixture(model, p)  # keep the mixture PD and non-degenerate
        return (p, opt_state, k), None

    carry = (params, opt_state, k_train)
    steps_log, elbo_log, nmi_log, purity_log = [], [], [], []
    for c in range(STEPS // LOG_EVERY):
        gs = jnp.arange(c * LOG_EVERY, (c + 1) * LOG_EVERY)
        carry, _ = jax.lax.scan(step, carry, gs)
        params = carry[0]
        ke1, ke2 = jax.random.split(jax.random.fold_in(k_eval, c))
        elbo = float(model.mean_elbo(ke1, params, test_x[:512], 8))
        nmi, purity = evaluate_clusters(model, params, test_x[:512], test_y[:512], ke2)
        steps_log.append((c + 1) * LOG_EVERY)
        elbo_log.append(elbo)
        nmi_log.append(nmi)
        purity_log.append(purity)
        print(f"  {kind:8s} step {steps_log[-1]:5d}  ELBO {elbo:8.3f}  "
              f"NMI {nmi:.3f}  purity {purity:.3f}")

    # Persist final params so downstream scripts (generative sampling) use the
    # exact trained model rather than re-deriving it.
    results_dir = example_paths(__file__).results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / f"mixture_params_{kind}.npy", np.array(params))
    pk = model.top_prior.lat_man.to_probs(
        model.top_prior.lat_man.to_mean(
            model.top_prior.split_natural_mixture(model.split_mixture(params))[1]
        )
    )
    print(f"  {kind:8s} final prior weights p(k): {np.array(pk)}")

    return {
        "kind": kind,
        "elbo_test": elbo_log[-1],
        "nmi": nmi_log[-1],
        "purity": purity_log[-1],
        "steps": steps_log,
        "elbo_traj": elbo_log,
        "nmi_traj": nmi_log,
        "purity_traj": purity_log,
    }


def main() -> None:
    key = jax.random.PRNGKey(0)
    k_data, k_diag, k_chain, k_chord = jax.random.split(key, 4)
    train_x, _, test_x, test_y = make_data(k_data)
    print(f"Data: {N_MODES}-mode MoG in R^{OBS_DIM} (labels known)")
    print(f"Model: X(Normal-{OBS_DIM}) <- Y(Boltzmann-{N_MID}) <- Z(Gaussian-{TOP_DIM}) "
          f"<- K(Categorical-{N_MODES})\n")

    results = []
    for kind, k in [("diagonal", k_diag), ("chain", k_chain), ("chordal", k_chord)]:
        print(f"--- middle = {kind} ---")
        results.append(fit(kind, train_x, test_x, test_y, k))
        print()

    print("=" * 60)
    print(f"{'middle':10s} {'ELBO test':>11s} {'NMI':>8s} {'purity':>8s}")
    print("-" * 60)
    for r in results:
        print(f"{r['kind']:10s} {r['elbo_test']:11.3f} {r['nmi']:8.3f} {r['purity']:8.3f}")
    print("=" * 60)

    results_dir = example_paths(__file__).results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "hierarchical_mixture_results.json"
    out.write_text(json.dumps({"n_modes": N_MODES, "results": results}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
