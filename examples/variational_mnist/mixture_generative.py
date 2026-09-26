"""Ancestral generative samples from the trained mixture-top hierarchy.

Answers "is the NMI real?" by drawing *generative* samples (NOT reconstructions)
from the model: ``k ~ p(k)``, ``z ~ p(z|k)``, ``y ~ p(y|z)``, ``x ~ p(x|y)`` -- no
data touches the sampler. We train the diagonal-middle model (the NMI~0.95 one),
generate, and overlay on the true data in the data's top-2 PCA plane. Model
samples are coloured by the generating component ``k``; if the model learned the
modes, its samples form the same blobs and each ``k`` maps to one true mode.

Run::

    uv run python -m examples.variational_mnist.mixture_generative --kind diagonal
"""

import argparse
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import Array

jax.config.update("jax_enable_x64", True)

from ..shared import example_paths  # noqa: E402
from . import hierarchical_mixture_experiment as E  # noqa: E402,N812
from .hierarchical_mixture import VariationalHierarchicalMixture  # noqa: E402

N_GEN = 2000


def train_params(kind: str, train_x: Array, key: Array) -> tuple[VariationalHierarchicalMixture, Array]:
    """Train a mixture-top model, returning (model, params).

    Splits ``key`` and runs the loop exactly as ``E.fit`` does, so passing the
    per-kind key from ``E.main``'s seeding reproduces the experiment's params
    (and its reported NMI) deterministically.
    """
    model = E.build_model(kind)
    k_init, k_train, _ = jax.random.split(key, 3)
    params = model.initialize_from_sample(k_init, train_x, location=0.0, shape=0.3)
    params = E.bound_mixture(model, params)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=E.LR, warmup_steps=E.LR_WARMUP,
        decay_steps=E.STEPS, end_value=0.0,
    )
    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(E.GRAD_CLIP), optax.adam(schedule)), 100
    )
    opt_state = optimizer.init(params)

    def loss_fn(p: Array, k: Array, batch: Array, gen_beta: Array) -> Array:
        ke, kc, ki = jax.random.split(k, 3)
        elbo = model.mean_elbo(ke, p, batch, E.MC_SAMPLES)
        gen = model.prior_conjugation_loss(kc, p, E.CONJ_SAMPLES)
        inner = model.mean_recognition_inner_loss(ki, p, batch, E.MC_SAMPLES)
        ent = E.mixture_entropy_penalty(model, p)
        return -elbo + gen_beta * E.LAMBDA_GEN * gen + E.LAMBDA_INNER * inner + E.ENT_REG * ent

    def step(
        carry: tuple[Array, Any, Array, Array], g: Array
    ) -> tuple[tuple[Array, Any, Array, Array], None]:
        p, opt_state, k, p_safe = carry
        gen_beta = jnp.clip((g - E.CONJ_HOLD) / E.CONJ_RAMP, 0.0, 1.0)
        k, kb, kl = jax.random.split(k, 3)
        batch = train_x[jax.random.choice(kb, train_x.shape[0], (E.BATCH,))]
        _, grads = jax.value_and_grad(loss_fn)(p, kl, batch, gen_beta)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = E.bound_mixture(model, optax.apply_updates(p, updates))
        # Snapshot the last all-finite params: the model clusters (NMI ~0.88)
        # before the late numerical blowup, so this yields a usable, PD model to
        # sample from even when training later goes NaN.
        p_safe = jnp.where(jnp.all(jnp.isfinite(p)), p, p_safe)
        return (p, opt_state, k, p_safe), None

    (_, _, _, p_safe), _ = jax.lax.scan(
        step, (params, opt_state, k_train, params), jnp.arange(E.STEPS)
    )
    return model, p_safe


def _z_to_x(model: VariationalHierarchicalMixture, params: Array, z: Array, key: Array) -> Array:
    """Sample x through the lower stack for each z: y ~ p(y|z), x ~ p(x|y)."""
    ky, kx = jax.random.split(key)
    n = z.shape[0]
    _, top_lkl, _ = model.split_top(params)
    _, lower_lkl, _ = model.split_coords(params)

    def y_of_z(subkey: Array, zi: Array) -> Array:
        s_z = model.top_man.sufficient_statistic(zi)
        y_params = model.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)
        return model.top_var.obs_man.sample(subkey, y_params, 1)[0]

    ys = jax.vmap(y_of_z)(jax.random.split(ky, n), z)

    def x_of_y(subkey: Array, yi: Array) -> Array:
        s_y = model.mid_man.sufficient_statistic(yi)
        x_params = model.lower_hrm.lkl_fun_man(lower_lkl, s_y)
        return model.obs_man.sample(subkey, x_params, 1)[0]

    return jax.vmap(x_of_y)(jax.random.split(kx, n), ys)


def generate(model: VariationalHierarchicalMixture, params: Array, key: Array, n: int) -> tuple[Array, Array]:
    """Ancestral generative samples using the prior p(k): returns (x (n, obs_dim), k (n,))."""
    kz, kx = jax.random.split(key)
    zk = model.top_prior.sample(kz, model.split_mixture(params), n)  # [z | k]
    d = model.top_man.data_dim
    z, k = zk[:, :d], zk[:, -1].astype(jnp.int32)
    return _z_to_x(model, params, z, kx), k


def generate_per_component(
    model: VariationalHierarchicalMixture, params: Array, key: Array, n_each: int
) -> tuple[Array, Array]:
    """Generate n_each x's from EACH component (uniform k), ignoring prior weights.

    Reveals whether the components are distinct in data space regardless of
    whether the generative weights p(k) collapsed.
    """
    mix = model.split_mixture(params)
    comp_nat, _ = model.top_prior.split_natural_mixture(mix)  # per-component Normal nats
    xs_all, k_all = [], []
    for m in range(model.n_clusters):
        km = jax.random.fold_in(key, m)
        comp_params = model.top_prior.cmp_man.get_replicate(comp_nat, m)
        z = model.top_man.sample(km, comp_params, n_each)
        xs_all.append(_z_to_x(model, params, z, jax.random.fold_in(km, 1)))
        k_all.append(jnp.full((n_each,), m, dtype=jnp.int32))
    return jnp.concatenate(xs_all), jnp.concatenate(k_all)


def prior_weights(model: VariationalHierarchicalMixture, params: Array) -> Array:
    """Learned generative mixing weights p(k)."""
    _, cat_nat = model.top_prior.split_natural_mixture(model.split_mixture(params))
    cat = model.top_prior.lat_man
    return cat.to_probs(cat.to_mean(cat_nat))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="diagonal", choices=["diagonal", "chain", "chordal"])
    ap.add_argument("--retrain", action="store_true", help="retrain even if cached params exist")
    args = ap.parse_args()

    # Reproduce E.main's exact seeding so cached params match the reported NMI.
    key = jax.random.PRNGKey(0)
    k_data, k_diag, k_chain, k_chord = jax.random.split(key, 4)
    perkey = {"diagonal": k_diag, "chain": k_chain, "chordal": k_chord}[args.kind]
    train_x, _, test_x, test_y = E.make_data(k_data)
    k_gen, k_pc = jax.random.split(jax.random.fold_in(key, 99))

    results_dir = example_paths(__file__).results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    param_path = results_dir / f"mixture_params_{args.kind}.npy"

    model = E.build_model(args.kind)
    if param_path.exists() and not args.retrain:
        params = jnp.asarray(np.load(param_path))
        print(f"loaded cached params from {param_path}")
    else:
        print(f"training middle={args.kind} ({E.STEPS} steps)...")
        model, params = train_params(args.kind, train_x, perkey)
        np.save(param_path, np.array(params))

    # Diagnostics: generative prior weights p(k) vs. posterior cluster usage.
    pk = np.array(prior_weights(model, params))
    resp_assign = np.array(model.cluster_assignments(k_pc, params, test_x[:512], E.RESP_SAMPLES))
    resp_hist = np.bincount(resp_assign, minlength=model.n_clusters) / resp_assign.size
    np.set_printoptions(precision=3, suppress=True)
    print(f"prior weights p(k)           : {pk}")
    print(f"posterior cluster usage r_k  : {resp_hist}")

    gen_x, gen_k = generate(model, params, k_gen, N_GEN)
    gen_x, gen_k = np.array(gen_x), np.array(gen_k)
    print(f"ancestral gen components used: {sorted(set(gen_k.tolist()))}")

    pc_x, pc_k = generate_per_component(model, params, jax.random.fold_in(k_gen, 1), 350)
    pc_x, pc_k = np.array(pc_x), np.array(pc_k)

    # PCA plane fitted on the DATA, applied to all.
    data = np.array(test_x)
    mu = data.mean(0)
    _, _, vt = np.linalg.svd(data - mu, full_matrices=False)
    pcs = vt[:2].T
    data2, gen2, pc2 = (data - mu) @ pcs, (gen_x - mu) @ pcs, (pc_x - mu) @ pcs

    labels = np.array(test_y)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True, sharex=True, sharey=True)
    axes[0].set_title(f"True data — {E.N_MODES} modes (by label)")
    for m in range(E.N_MODES):
        pts = data2[labels == m]
        axes[0].scatter(pts[:, 0], pts[:, 1], s=8, color=cmap(m), alpha=0.5)

    axes[1].set_title(f"Ancestral gen  k~p(k)  (by k) — {args.kind}")
    for m in range(model.n_clusters):
        pts = gen2[gen_k == m]
        if pts.size:
            axes[1].scatter(pts[:, 0], pts[:, 1], s=8, color=cmap(m), alpha=0.5)

    axes[2].set_title("Per-component gen  (uniform k, by k)")
    for m in range(model.n_clusters):
        pts = pc2[pc_k == m]
        if pts.size:
            axes[2].scatter(pts[:, 0], pts[:, 1], s=8, color=cmap(m), alpha=0.5)

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(f"Generative samples vs data (data PCA plane) — middle={args.kind}")

    out = results_dir / f"mixture_generative_{args.kind}.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
