"""Mixture-top hierarchy on MNIST: real digits vs. per-cluster ancestral samples.

Trains ``p(x, y, z, k)`` (X=784 pixels, Y=Boltzmann spikes, Z=Gaussian,
K=cluster) on MNIST and shows what the *generative* model produces: for each
learned cluster ``k`` it draws ancestral samples ``z ~ p(z|k)``, ``y ~ p(y|z)``,
``x = E[x|y]`` and renders them as digits, alongside a row of real test digits.
If clusters correspond to digit classes, each row should look like one digit.
Digit labels are used only to *evaluate* the unsupervised clustering (NMI /
purity), never to train.

Supports the full set of middle layers from :mod:`.mnist_hierarchical`,
including the **conv/chordal lower edge** (exact ``r_Y == 0`` conjugation), and
the validated CW2 training recipe: reparameterized z-gradient, norm-preserving
conjugation gradient on Theta_ZN, learned bounded observation noise, plus the
mixture stabilization from :mod:`.hierarchical_mixture_experiment`
(``bound_mixture`` projection, mixture-weight entropy).

**Layerwise warm start** (the same saddle-breaking principle that produced the
CW2 champion): ``--resume`` a trained single-Gaussian checkpoint -- the top
edge, lower likelihood, and recognition blocks transfer verbatim -- and seed the
mixture components by k-means + within-cluster moment matching on the
aggregate-posterior z's of the training data.

Run::

    uv run python -m examples.variational_mnist.mnist_mixture --middle conv \\
        --conv-prior chordal --conv-kernel 6 4 --reparam-z --lambda-z 1.0 \\
        --norm-preserve --resume CKPT.npz --steps 10000
"""

import argparse
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import Array

jax.config.update("jax_enable_x64", True)

from goal.geometry import Diagonal  # noqa: E402
from goal.models import Normal  # noqa: E402

from ..shared import example_paths  # noqa: E402
from . import hierarchical_mixture_experiment as E  # noqa: E402,N812
from . import mnist_hierarchical as MH  # noqa: E402,N812
from .hierarchical import VariationalHierarchical  # noqa: E402
from .hierarchical_mixture import (  # noqa: E402
    VariationalHierarchicalMixture,
    build_boltzmann_gaussian_mixture_hierarchy,
    build_conv_boltzmann_gaussian_mixture_hierarchy,
)

IMG = MH.IMG
N_OBS = MH.N_OBS
N_CLUSTERS = 10  # one per digit class (unsupervised)

# CW2-recipe constants shared with mnist_hierarchical
LAMBDA_INNER = 1.0
LR_WARMUP = 200
WEIGHT_DECAY = 1e-3
WARMUP = 200  # ramp for the conjugation penalty
MC_SAMPLES = 3
CONJ_SAMPLES = 16
EVAL_N = 64  # ELBO/conjugation monitoring batch (see mnist_hierarchical.EVAL_N)
CLUSTER_EVAL_N = 256  # test images for NMI/purity readout
RESP_SAMPLES = 16  # z-samples per image for responsibilities


def build_model(middle: str, n_mid: int, top_dim: int, n_clusters: int,
                chordal_width: int = MH.CHORDAL_WIDTH,
                conv_in: tuple[int, int] = MH.CONV_IN,
                conv_stride: tuple[int, int] = MH.CONV_STRIDE,
                conv_kernel: tuple[int, int] = MH.CONV_KERNEL,
                conv_channels: int = 1,
                conv_prior: str = "chordal",
                mlp_hidden: tuple[int, ...] = (128,)) -> VariationalHierarchicalMixture:
    obs_man = Normal(N_OBS, Diagonal())
    if middle == "conv":
        if conv_in[0] * conv_stride[0] != IMG or conv_in[1] * conv_stride[1] != IMG:
            raise ValueError(f"conv_in*stride must equal ({IMG},{IMG})")
        return build_conv_boltzmann_gaussian_mixture_hierarchy(
            obs_man, conv_in, conv_stride, conv_kernel, top_dim, n_clusters,
            in_channels=conv_channels, prior_graph=conv_prior,
            max_treewidth=2 * max(conv_in) * conv_channels, mlp_hidden=mlp_hidden,
        )
    common = dict(mlp_hidden=mlp_hidden, obs_location_only=True)
    if middle == "chordal":
        w = chordal_width
        h = n_mid // w
        return build_boltzmann_gaussian_mixture_hierarchy(
            obs_man, h * w, top_dim, n_clusters, mid_kind="chordal",
            mid_edges=MH.grid_edges(h, w), max_treewidth=w + 2, **common,
        )
    return build_boltzmann_gaussian_mixture_hierarchy(
        obs_man, n_mid, top_dim, n_clusters, mid_kind=middle, **common
    )


# --- Layerwise warm start ----------------------------------------------------


def _posterior_top_moments(
    base_model: VariationalHierarchical, base_params: Array, xs: Array
) -> tuple[Array, Array]:
    """Per-image mean and covariance of q(z | x) under the trained base model."""
    d = base_model.top_man.data_dim

    def one(x: Array) -> tuple[Array, Array]:
        q = base_model.approximate_posterior_top(base_params, x)
        loc, prec_params = base_model.top_man.split_location_precision(q)
        prec = base_model.top_man.cov_man.rep.to_matrix((d, d), prec_params)
        cov = jnp.linalg.inv(0.5 * (prec + prec.T))
        return cov @ loc, cov

    return jax.vmap(one)(xs)


def seed_mixture_kmeans(
    mix_model: VariationalHierarchicalMixture,
    base_model: VariationalHierarchical,
    base_params: Array,
    xs: Array,
    seed: int = 0,
    n_codes: int = 4000,
) -> Array:
    """Mixture natural parameters from k-means on the aggregate posterior q(z | x).

    The layerwise seeding principle applied to the top prior: the base model's
    aggregate posterior is multimodal (digit-class clusters) while its
    single-Gaussian prior is not -- the dominant generation defect. Run k-means
    (K = n_clusters) on the posterior means, then moment-match each component to
    its cluster's aggregate-posterior moments: mean = centroid, covariance =
    within-cluster covariance of posterior means + mean posterior covariance
    (the exact within-cluster second moment), weights = cluster proportions.
    Covariances are eigenvalue-floored the same way ``bound_mixture`` bounds
    them during training.
    """
    mus, covs = _posterior_top_moments(base_model, base_params, xs[:n_codes])
    z = np.asarray(mus)
    cvs = np.asarray(covs)
    k = mix_model.n_clusters
    dd = mix_model.top_man.data_dim
    rng = np.random.default_rng(seed)

    cent = z[rng.choice(z.shape[0], k, replace=False)]
    lab = np.zeros(z.shape[0], dtype=int)
    for _ in range(100):
        lab = ((z[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            members = z[lab == j]
            cent[j] = members.mean(0) if members.shape[0] else z[rng.integers(z.shape[0])]

    top_man = mix_model.top_man
    comp_nats = []
    counts = []
    for j in range(k):
        m = lab == j
        counts.append(int(m.sum()))
        if not m.any():  # dead cluster: fall back to global moments
            m = np.ones_like(m)
        mu_j = z[m].mean(0)
        sigma_j = np.cov(z[m].T, bias=True).reshape(dd, dd) + cvs[m].mean(0)
        evals, evecs = np.linalg.eigh(sigma_j + E.LAT_JITTER_VAR * np.eye(dd))
        sigma_j = (evecs * np.maximum(evals, E.LAT_MIN_VAR)) @ evecs.T
        prec = np.linalg.inv(sigma_j)
        comp_nats.append(top_man.join_location_precision(
            jnp.asarray(prec @ mu_j), top_man.cov_man.rep.from_matrix(jnp.asarray(prec))
        ))

    probs = np.maximum(np.asarray(counts, dtype=float) / sum(counts), E.MIN_PROB)
    probs = probs / probs.sum()
    cat = mix_model.top_prior.lat_man
    cat_nat = cat.to_natural(cat.from_probs(jnp.asarray(probs)))
    print(f"  [seed-mixture] k-means cluster sizes: {counts}")
    return mix_model.top_prior.join_natural_mixture(jnp.concatenate(comp_nats), cat_nat)


def warm_start_from_base(
    mix_model: VariationalHierarchicalMixture,
    base_model: VariationalHierarchical,
    base_params: Array,
    mixture_nat: Array,
) -> Array:
    """Transfer a single-Gaussian checkpoint into the mixture layout verbatim.

    The top edge (including the now-inert stored theta*_Z), lower likelihood,
    and recognition blocks are identical manifolds in both models; only the
    mixture prior is new.
    """
    top, lower_lkl, recog = base_model.split_coords(base_params)
    third = mix_model.trd_man.join_coords(recog, mixture_nat)
    return mix_model.join_coords(top, lower_lkl, third)


# --- Training (CW2 recipe + mixture stabilization) ---------------------------


def train(model: VariationalHierarchicalMixture, train_data: Array, test_data: Array,
          steps: int, key: Array, lambda_y: float = 0.0, lambda_z: float = 0.0,
          lr: float = MH.LR, grad_clip: float = MH.GRAD_CLIP,
          max_var: float = MH.OBS_MAX_VAR,
          batch: int = MH.BATCH, mc_samples: int = MC_SAMPLES,
          ent_reg: float = E.ENT_REG,
          reparam_z: bool = False, norm_preserve: bool = False,
          marginal_y: bool = False,
          init_params: Array | None = None,
          test_labels: np.ndarray | None = None) -> Array:
    k_init, k_train, k_eval = jax.random.split(key, 3)
    if init_params is None:
        params = model.initialize_from_sample(k_init, train_data, location=0.0, shape=0.3)
        params = MH.init_observation_noise(
            model, params, jnp.mean(train_data, axis=0), jnp.var(train_data, axis=0),
            max_var=max_var,
        )
    else:
        params = init_params  # layerwise warm start
    params = MH.bound_observable_covariance(model, params, max_var=max_var)
    params = E.bound_mixture(model, params)

    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, LR_WARMUP, steps, end_value=0.0)
    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
        ),
        100,
    )
    opt_state = optimizer.init(params)

    # Theta_ZN block bounds for the norm-preserving projection (same layout as
    # the base model: the top block leads the parameter vector).
    zn_s = model.top_man.dim + model.mid_man.dim
    zn_e = model.top_man.dim + model.top_var.gen_hrm.lkl_fun_man.dim

    def loss_main(p: Array, k: Array, batch_xs: Array) -> Array:
        ke, ki = jax.random.split(k)
        if marginal_y:  # exact-N estimator: pathwise z-gradients only
            elbo = model.mean_marginal_elbo(ke, p, batch_xs, mc_samples)
        else:
            elbo = model.mean_elbo(ke, p, batch_xs, mc_samples, reparam_z=reparam_z)
        inner = model.mean_recognition_inner_loss(ki, p, batch_xs, mc_samples)
        ent = E.mixture_entropy_penalty(model, p)
        return -elbo + LAMBDA_INNER * inner + ent_reg * ent

    def loss_conj(p: Array, k: Array) -> Array:
        var_r_y, var_r_z = model.prior_conjugation_loss_components(k, p, CONJ_SAMPLES)
        return lambda_y * var_r_y + lambda_z * var_r_z

    use_conj = lambda_y > 0.0 or lambda_z > 0.0  # static at trace time

    @jax.jit
    def step(carry, g):
        p, opt_state, k, p_safe = carry
        gen_beta = jnp.minimum(1.0, g / WARMUP)
        k, kb, kl, kc = jax.random.split(k, 4)
        batch_xs = train_data[jax.random.choice(kb, train_data.shape[0], (batch,))]
        grads = jax.grad(loss_main)(p, kl, batch_xs)
        if use_conj:
            g_conj = jax.grad(loss_conj)(p, kc)
            if norm_preserve:
                blk, th = g_conj[zn_s:zn_e], p[zn_s:zn_e]
                that = th / (jnp.linalg.norm(th) + 1e-12)
                g_conj = g_conj.at[zn_s:zn_e].set(blk - jnp.dot(blk, that) * that)
            grads = grads + gen_beta * g_conj
        updates, opt_state = optimizer.update(grads, opt_state, p)
        p = MH.bound_observable_covariance(model, optax.apply_updates(p, updates), max_var=max_var)
        p = E.bound_mixture(model, p)  # keep the mixture PD and non-degenerate
        p_safe = jnp.where(jnp.all(jnp.isfinite(p)), p, p_safe)  # last all-finite params
        return (p, opt_state, k, p_safe), None

    def report(tag: str, params: Array, t0: float, live: str = "") -> None:
        k_etr, k_ete = jax.random.split(k_eval)
        ev = model.mean_marginal_elbo if marginal_y else model.mean_elbo
        etr = float(ev(k_etr, params, train_data[:EVAL_N], 8))
        ete = float(ev(k_ete, params, test_data[:EVAL_N], 8))
        vry, vrz = model.prior_conjugation_loss_components(k_etr, params, EVAL_N)
        pk = np.array(_prior_weights(model, params))
        _, top_lkl, _ = model.split_top(params)
        _, theta_zn = model.top_var.gen_hrm.lkl_fun_man.split_coords(top_lkl)
        cl = ""
        if test_labels is not None:
            pred = np.array(model.cluster_assignments(
                k_ete, params, test_data[:CLUSTER_EVAL_N], RESP_SAMPLES))
            nmi, purity = E.cluster_metrics(
                pred, test_labels[:CLUSTER_EVAL_N], model.n_clusters)
            cl = f"  NMI {nmi:.3f}  purity {purity:.3f}"
        print(f"  {tag}  ELBO train {etr:8.2f}  test {ete:8.2f}  "
              f"Var[rY] {float(vry):6.2f}  Var[rZ] {float(vrz):6.3f}  "
              f"|Theta_ZN| {float(jnp.linalg.norm(theta_zn)):6.3f}  "
              f"p(k)[{pk.min():.3f},{pk.max():.2f}]{cl}  ({time.time()-t0:.0f}s){live}")

    t0 = time.time()
    report("step     0", params, t0)  # transfer quality before any mixture training
    log_every = max(1, steps // 20)
    carry = (params, opt_state, k_train, params)
    for c in range(steps // log_every):
        gs = jnp.arange(c * log_every, (c + 1) * log_every)
        carry, _ = jax.lax.scan(step, carry, gs)
        params = carry[3]  # evaluate the last-finite snapshot
        live = "" if bool(jnp.all(jnp.isfinite(carry[0]))) else "  [live=NaN, using snapshot]"
        report(f"step {(c+1)*log_every:5d}", params, t0, live)
    return carry[3]


def _prior_weights(model: VariationalHierarchicalMixture, params: Array) -> Array:
    _, cat_nat = model.top_prior.split_natural_mixture(model.split_mixture(params))
    cat = model.top_prior.lat_man
    return cat.to_probs(cat.to_mean(cat_nat))


def per_cluster_gen_means(
    model: VariationalHierarchicalMixture, params: Array, key: Array, n_each: int
) -> Array:
    """Generative likelihood means E[x|y] for n_each ancestral draws from EACH component.

    Returns shape ``(n_clusters, n_each, N_OBS)``.
    """
    comp_nat, _ = model.top_prior.split_natural_mixture(model.split_mixture(params))
    _, lower_lkl, _ = model.split_coords(params)
    _, top_lkl, _ = model.split_top(params)

    def gen_one(subkey: Array, z: Array) -> Array:
        s_z = model.top_man.sufficient_statistic(z)
        y_nat = model.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)
        y = model.mid_man.sample(subkey, y_nat, 1)[0]
        s_y = model.mid_man.sufficient_statistic(y)
        x_nat = model.lower_hrm.lkl_fun_man(lower_lkl, s_y)
        return model.obs_man.to_mean(x_nat)[: model.obs_man.data_dim]

    def cluster_row(kk: int) -> Array:
        comp = model.top_prior.cmp_man.get_replicate(comp_nat, kk)
        zc = model.top_man.sample(jax.random.fold_in(key, kk), comp, n_each)
        keys = jax.random.split(jax.random.fold_in(key, 1000 + kk), n_each)
        return jax.vmap(gen_one)(keys, zc)

    return jnp.stack([cluster_row(kk) for kk in range(model.n_clusters)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--middle", default="chordal",
                    choices=["diagonal", "chain", "chordal", "conv"])
    ap.add_argument("--n-mid", type=int, default=64)
    ap.add_argument("--top-dim", type=int, default=16)
    ap.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    ap.add_argument("--chordal-width", type=int, default=MH.CHORDAL_WIDTH)
    ap.add_argument("--conv-in", type=int, nargs=2, default=list(MH.CONV_IN))
    ap.add_argument("--conv-stride", type=int, nargs=2, default=list(MH.CONV_STRIDE))
    ap.add_argument("--conv-kernel", type=int, nargs=2, default=list(MH.CONV_KERNEL))
    ap.add_argument("--conv-channels", type=int, default=1)
    ap.add_argument("--conv-prior", default="chordal", choices=["chordal", "diagonal"])
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[128])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=MH.BATCH)
    ap.add_argument("--mc", type=int, default=MC_SAMPLES)
    ap.add_argument("--n-train", type=int, default=MH.N_TRAIN)
    ap.add_argument("--lambda-y", type=float, default=0.0,
                    help="bottom-edge (Gaussian-Boltzmann) conjugation weight")
    ap.add_argument("--lambda-z", type=float, default=0.0,
                    help="top-edge (Boltzmann pop-code Y|Z) conjugation weight")
    ap.add_argument("--ent-reg", type=float, default=E.ENT_REG,
                    help="mixture-weight entropy regularizer (anti-collapse)")
    ap.add_argument("--lr", type=float, default=MH.LR)
    ap.add_argument("--grad-clip", type=float, default=MH.GRAD_CLIP)
    ap.add_argument("--max-var", type=float, default=MH.OBS_MAX_VAR)
    ap.add_argument("--marginal-y", action="store_true",
                    help="exact-N ELBO (conv middles only): spikes integrate out of the "
                    "residual (r_Y == 0), z is pathwise -- no score-function estimator")
    ap.add_argument("--reparam-z", action="store_true",
                    help="pathwise gradient for the Gaussian top latent")
    ap.add_argument("--norm-preserve", action="store_true",
                    help="shape-only conjugation gradient on Theta_ZN")
    ap.add_argument("--resume", default="",
                    help="single-Gaussian npz checkpoint to warm-start from "
                    "(top/lower/recog transfer verbatim; mixture seeded by k-means "
                    "on the aggregate-posterior z's)")
    ap.add_argument("--seed-codes", type=int, default=4000,
                    help="training images used for the k-means mixture seeding")
    ap.add_argument("--outdir", default="", help="subfolder under results/ for this run")
    ap.add_argument("--note", default="", help="one-line description for the INDEX")
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)
    _, k_train, k_gen = jax.random.split(key, 3)
    train_data, test_data, _, test_labels = MH.load_mnist(
        args.n_train, MH.N_TEST, with_labels=True)
    print(f"MNIST: train {train_data.shape}, test {test_data.shape}")

    model = build_model(args.middle, args.n_mid, args.top_dim, args.n_clusters,
                        args.chordal_width,
                        conv_in=tuple(args.conv_in), conv_stride=tuple(args.conv_stride),
                        conv_kernel=tuple(args.conv_kernel),
                        conv_channels=args.conv_channels,
                        conv_prior=args.conv_prior, mlp_hidden=tuple(args.mlp_hidden))
    if args.middle == "conv":
        args.n_mid = model.mid_man.data_dim
        print(f"conv decoder {tuple(args.conv_in)} -> ({IMG},{IMG}) (stride "
              f"{tuple(args.conv_stride)}, kernel {tuple(args.conv_kernel)}); "
              f"latent nodes={args.n_mid}, mid dim={model.mid_man.dim}")
    print(f"Model: X(Normal-{N_OBS}) <- Y(Boltzmann-{args.n_mid}, {args.middle}) "
          f"<- Z(Gaussian-{args.top_dim}) <- K(Categorical-{args.n_clusters})")

    init_params = None
    if args.resume:
        base_model = MH.build_model(
            args.middle, args.n_mid, args.top_dim, args.chordal_width,
            conv_in=tuple(args.conv_in), conv_stride=tuple(args.conv_stride),
            conv_kernel=tuple(args.conv_kernel), conv_channels=args.conv_channels,
            conv_prior=args.conv_prior, mlp_hidden=tuple(args.mlp_hidden))
        base_params = jnp.asarray(np.load(args.resume)["params"])
        print(f"warm start from {args.resume}")
        mixture_nat = seed_mixture_kmeans(
            model, base_model, base_params, train_data, n_codes=args.seed_codes)
        init_params = warm_start_from_base(model, base_model, base_params, mixture_nat)

    params = train(model, train_data, test_data, args.steps, k_train,
                   lambda_y=args.lambda_y, lambda_z=args.lambda_z,
                   lr=args.lr, grad_clip=args.grad_clip, max_var=args.max_var,
                   batch=args.batch, mc_samples=args.mc, ent_reg=args.ent_reg,
                   reparam_z=args.reparam_z, norm_preserve=args.norm_preserve,
                   marginal_y=args.marginal_y,
                   init_params=init_params, test_labels=test_labels)

    # Final metrics
    k_rec, k_cl = jax.random.split(k_gen)
    ev = model.mean_marginal_elbo if args.marginal_y else model.mean_elbo
    ete = float(ev(k_rec, params, test_data[:EVAL_N], 8))
    vry, vrz = model.prior_conjugation_loss_components(k_rec, params, EVAL_N)
    vry, vrz = float(vry), float(vrz)
    pred = np.array(model.cluster_assignments(
        k_cl, params, test_data[:CLUSTER_EVAL_N], RESP_SAMPLES))
    nmi, purity = E.cluster_metrics(pred, test_labels[:CLUSTER_EVAL_N], model.n_clusters)
    print(f"ELBO test {ete:.2f}  Var[rY] {vry:.2f}  Var[rZ] {vrz:.3f}  "
          f"NMI {nmi:.3f}  purity {purity:.3f}")

    n_each = 8
    gens = np.array(per_cluster_gen_means(model, params, k_gen, n_each))
    pk = np.array(_prior_weights(model, params))
    print(f"final p(k): {pk}")

    # Figure: row 0 = real digits; rows 1..K = per-cluster generative means.
    rows = model.n_clusters + 1
    fig, axes = plt.subplots(rows, n_each, figsize=(1.1 * n_each, 1.1 * rows))
    for j in range(n_each):
        axes[0, j].imshow(np.array(test_data[j]).reshape(IMG, IMG), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_ylabel("real", rotation=0, ha="right", va="center", fontsize=9)
    for kk in range(model.n_clusters):
        for j in range(n_each):
            ax = axes[kk + 1, j]
            ax.imshow(np.clip(gens[kk, j].reshape(IMG, IMG), 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[kk + 1, 0].set_ylabel(f"k={kk}\np={pk[kk]:.2f}", rotation=0, ha="right",
                                   va="center", fontsize=8)
    for r in range(rows):
        for j in range(n_each):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    fig.suptitle(f"MNIST per-cluster ancestral samples (middle={args.middle}, "
                 f"NMI {nmi:.2f}, purity {purity:.2f})")
    fig.tight_layout()

    results_dir = example_paths(__file__).results_dir
    if args.outdir:
        results_dir = results_dir / args.outdir
    results_dir.mkdir(parents=True, exist_ok=True)
    wtag = f"_w{args.chordal_width}" if args.middle == "chordal" else ""
    if args.middle == "conv":
        ci, cs, ck = tuple(args.conv_in), tuple(args.conv_stride), tuple(args.conv_kernel)
        wtag = f"_{ci[0]}x{ci[1]}s{cs[0]}k{ck[0]}x{ck[1]}_C{args.conv_channels}_{args.conv_prior}"
    etag = (("_mg" if args.marginal_y else "")
            + ("_rp" if args.reparam_z else "") + ("_np" if args.norm_preserve else ""))
    if args.resume:
        etag += "_wsKM"
    tag = (f"{args.middle}{wtag}_n{args.n_mid}_td{args.top_dim}_K{args.n_clusters}"
           f"_ly{args.lambda_y:g}_lz{args.lambda_z:g}_lr{args.lr:g}{etag}_st{args.steps}")
    out = results_dir / f"mnist_mixture_{tag}.png"
    fig.savefig(out, dpi=130)
    np.savez(out.with_suffix(".npz"), params=np.asarray(params))
    _append_index(results_dir, out.name, args, ete, vry, vrz, nmi, purity)
    print(f"saved {out}")


def _append_index(results_dir, fname, args, ete, vry, vrz, nmi, purity) -> None:
    """Append a row to INDEX.md in ``results_dir`` describing this run."""
    from datetime import datetime

    index = results_dir / "INDEX.md"
    header = ("| when | middle | n_mid | K | lam_y | lam_z | lr "
              "| ELBO test | Var[rY] | Var[rZ] | NMI | purity | figure | note |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    if not index.exists():
        index.write_text("# MNIST mixture-top runs\n\n" + header)
    when = datetime.now().strftime("%m-%d %H:%M")
    row = (f"| {when} | {args.middle} | {args.n_mid} | {args.n_clusters} "
           f"| {args.lambda_y:g} | {args.lambda_z:g} | {args.lr:g} "
           f"| {ete:.1f} | {vry:.1f} | {vrz:.3f} | {nmi:.3f} | {purity:.3f} "
           f"| {fname} | {args.note} |\n")
    with index.open("a") as f:
        f.write(row)


if __name__ == "__main__":
    main()
