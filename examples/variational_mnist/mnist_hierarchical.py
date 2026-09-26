"""Hierarchical variational conjugation on MNIST (WP2, continuous observable).

Trains the 3-level ``p(x, y, z) = p(x|y) p(y|z) p(z)`` model from
:mod:`.hierarchical` on MNIST as a *continuous* observable:

- ``x`` -- 784 pixels rescaled to ``[0, 1]``, a fixed-covariance DiagonalNormal
  (the interaction drives only the mean; covariance is a global parameter),
- ``y`` -- a Boltzmann spike population (chain or chordal grid),
- ``z`` -- a Gaussian top latent.

Outputs ``mnist_hierarchical.png`` (data / reconstruction / generative samples)
and prints ELBO, reconstruction MSE, and conjugation-residual variance.

Run::

    uv run python -m examples.variational_mnist.mnist_hierarchical
    uv run python -m examples.variational_mnist.mnist_hierarchical --middle chordal --steps 2000
"""

import argparse
import time
from pathlib import Path

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
from .hierarchical import (  # noqa: E402
    VariationalHierarchical,
    build_boltzmann_gaussian_hierarchy,
    build_conv_boltzmann_gaussian_hierarchy,
)

IMG = 28
N_OBS = IMG * IMG  # 784

# Defaults (CPU-friendly; override on the CLI)
N_MID = 64
TOP_DIM = 16
N_TRAIN = 8000
N_TEST = 2000
BATCH = 100
MC_SAMPLES = 3
CONJ_SAMPLES = 16
# Diagnostic (ELBO/conjugation) eval batch. Kept modest because high-treewidth
# middles (e.g. the 7x7 conv king-graph, treewidth ~11) materialize a clique
# tensor per example x MC sample; a 256-wide vmap there exhausts memory. 64 is a
# fine monitoring estimate and keeps the peak bounded for every middle.
EVAL_N = 64
# Streamlined ("back to theory") defaults. The bottom edge p(x|y) = N(mu + W s_Y,
# Psi) is exactly a factor-analysis likelihood (y-independent diagonal noise) with
# a Boltzmann code, so we *learn* the specific variances Psi rather than freezing
# them, projecting into [OBS_MIN_VAR, OBS_MAX_VAR] after each step. The Gaussian ->
# Boltzmann edge is not analytically conjugate, and unregularized (lambda_gen = 0)
# amortized VI generates the sharpest samples, so conjugation regularization is off
# by default; weight decay + grad clip + the variance cap keep Theta_XY bounded.
LAMBDA_GEN = 0.0
LAMBDA_INNER = 1.0
LR = 1e-3
LR_WARMUP = 200
GRAD_CLIP = 0.5
WEIGHT_DECAY = 1e-3
WARMUP = 200  # ramp for lambda_gen when it is turned on
OBS_MIN_VAR = 1e-3  # variance floor: caps precision, tames zero-variance bg pixels
OBS_MAX_VAR = 0.15  # variance cap: floors precision, keeps the quadratic psi_X bounded


def load_mnist(n_train: int, n_test: int, with_labels: bool = False):
    """MNIST pixels rescaled to [0, 1]; cached under project .data/.

    With ``with_labels`` also returns the integer digit labels (train, test) --
    used only for *evaluation* of unsupervised clustering (NMI/purity), never
    for training.
    """
    from sklearn.datasets import fetch_openml

    data_home = str(Path(__file__).parents[2] / ".data")
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto", data_home=data_home
    )
    data = (mnist.data / 255.0).astype(np.float32)
    train = jnp.array(data[:n_train])
    test = jnp.array(data[60000 : 60000 + n_test])
    if with_labels:
        labels = mnist.target.astype(int)
        return train, test, np.asarray(labels[:n_train]), np.asarray(labels[60000 : 60000 + n_test])
    return train, test


def grid_edges(h: int, w: int) -> list[tuple[int, int]]:
    """4-neighbour edges on an ``h x w`` grid (row-major node indexing).

    A thin grid (small ``w``) triangulates to treewidth ~``w``, so the
    junction-tree log-partition stays cheap while still giving genuinely 2-D
    (chordal) connectivity, richer than a chain.
    """
    edges: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            if c + 1 < w:
                edges.append((i, r * w + c + 1))
            if r + 1 < h:
                edges.append((i, (r + 1) * w + c))
    return edges


OBS_VAR = 0.05  # fixed observation noise variance (Gaussian decoder)


def fix_observation_noise(
    model: VariationalHierarchical, params: Array, data_mean: Array, obs_var: float
) -> Array:
    """Set the observable to a fixed-noise Gaussian with mean ``data_mean``.

    MNIST has constant (zero-variance) background/corner pixels, so
    ``initialize_from_sample`` sets near-infinite precision there and the ELBO
    blows up. We set a standard Gaussian decoder instead: precision ``1/obs_var``
    uniformly, and the observable bias mean to the data mean (natural location
    ``theta_1 = Lambda . mu``). Only the mean is driven by ``y`` thereafter.

    (Uniform-noise initializer, kept for the mixture experiment. The streamlined
    non-mixture model uses :func:`init_observation_noise` for a per-pixel,
    FA-style variance init that is then learned.)
    """
    top, lower_lkl, recog = model.split_coords(params)
    _, theta_xy = model.lower_hrm.lkl_fun_man.split_coords(lower_lkl)
    prec = jnp.full(model.obs_man.data_dim, 1.0 / obs_var)
    theta_x = model.obs_man.join_location_precision(data_mean / obs_var, prec)
    lower_lkl = model.lower_hrm.lkl_fun_man.join_coords(theta_x, theta_xy)
    return model.join_coords(top, lower_lkl, recog)


def init_observation_noise(
    model: VariationalHierarchical, params: Array,
    data_mean: Array, data_var: Array,
    min_var: float = OBS_MIN_VAR, max_var: float = OBS_MAX_VAR,
) -> Array:
    """Initialize the observable as a diagonal Gaussian decoder with FA-style
    per-pixel specific variances.

    Sets ``Psi_ii = clip(Var(x_i), min_var, max_var)`` from the empirical
    per-pixel variance and the bias mean to ``data_mean``. Flooring finite-izes
    MNIST's zero-variance background pixels without freezing anything -- ``Psi`` is
    a learned parameter from here, refined each step and reprojected by
    :func:`bound_observable_covariance`. Background pixels thus start sharp
    (``var -> min_var``) and stroke pixels at their true variance, giving the
    precision gradient a correct per-pixel starting point instead of a flat one.
    """
    top, lower_lkl, recog = model.split_coords(params)
    _, theta_xy = model.lower_hrm.lkl_fun_man.split_coords(lower_lkl)
    prec = 1.0 / jnp.clip(data_var, min_var, max_var)
    theta_x = model.obs_man.join_location_precision(data_mean * prec, prec)
    lower_lkl = model.lower_hrm.lkl_fun_man.join_coords(theta_x, theta_xy)
    return model.join_coords(top, lower_lkl, recog)


def bound_observable_covariance(
    model: VariationalHierarchical, params: Array,
    min_var: float = OBS_MIN_VAR, max_var: float = OBS_MAX_VAR,
) -> Array:
    """Project the observable's diagonal noise variance into ``[min_var, max_var]``.

    The principled replacement for freezing the precision: the specific variances
    ``Psi`` are learned parameters, but clamped after each step so a constant
    (zero-variance) background pixel cannot drive its precision to infinity and the
    fitted stroke-pixel variance stays small enough to keep the quadratic ``psi_X``
    bounded. Done in natural (precision) space to avoid a ``to_mean`` through a
    possibly non-PD covariance.
    """
    top, lower_lkl, recog = model.split_coords(params)
    theta_x, theta_xy = model.lower_hrm.lkl_fun_man.split_coords(lower_lkl)
    loc, prec = model.obs_man.split_location_precision(theta_x)
    prec = model.obs_man.cov_man.map_diagonal(
        prec, lambda p: jnp.clip(p, 1.0 / max_var, 1.0 / min_var)
    )
    theta_x = model.obs_man.join_location_precision(loc, prec)
    lower_lkl = model.lower_hrm.lkl_fun_man.join_coords(theta_x, theta_xy)
    return model.join_coords(top, lower_lkl, recog)


def bound_top_prior(
    model: VariationalHierarchical, params: Array,
    min_eig: float = 1e-2, max_eig: float = 1e2,
) -> Array:
    """Project the top prior's precision eigenvalues into ``[min_eig, max_eig]``.

    The generative Gaussian prior theta*_Z has no structural guard: training can
    push its precision out of the PD cone, at which point sampling p(z), the
    baseline psi_Z, and the conjugation penalty all go NaN *while every
    coordinate stays finite* -- so the last-all-finite snapshot doesn't protect
    against it (finite is not valid). The observed C1a/A1/A2b blowups share this
    signature. Same stability-shim pattern as :func:`bound_observable_covariance`,
    applied to the 16x16 (or 8x8) top precision via an eigenvalue clamp.
    """
    top, lower_lkl, recog = model.split_coords(params)
    theta_z, top_lkl, rho0 = model.top_var.split_coords(top)
    loc, prec_params = model.top_man.split_location_precision(theta_z)
    d = model.top_man.data_dim
    prec = model.top_man.cov_man.rep.to_matrix((d, d), prec_params)
    prec = 0.5 * (prec + prec.T)
    evals, evecs = jnp.linalg.eigh(prec)
    prec = (evecs * jnp.clip(evals, min_eig, max_eig)) @ evecs.T
    theta_z = model.top_man.join_location_precision(
        loc, model.top_man.cov_man.rep.from_matrix(prec)
    )
    top = model.top_var.join_coords(theta_z, top_lkl, rho0)
    return model.join_coords(top, lower_lkl, recog)


def _posterior_codes(
    model: VariationalHierarchical, params: Array, xs: Array, chunk: int = 128
) -> Array:
    """Posterior mean spike codes E[N | x] (probabilities) for a batch of images.

    Chunked: ``to_mean`` is a junction-tree pass, and vmapping thousands of them
    at high treewidth materializes tens of GiB of clique tensors.
    """
    f = jax.jit(jax.vmap(
        lambda x: model.mid_man.to_mean(model.posterior_mid_bias(params, x))
    ))
    return jnp.concatenate([f(xs[i:i + chunk]) for i in range(0, xs.shape[0], chunk)])


def seed_top_interaction_pca(
    model: VariationalHierarchical, params: Array, xs: Array,
    target_std: float = 1.5, n_codes: int = 2000,
) -> Array:
    """Seed Theta_ZN's location block from PCA of the posterior spike codes.

    Breaks the symmetric collapse fixed point (uninformative q(z|x) <->
    uninformative Theta_ZN, each holding the other at zero gradient): after
    seeding, p(N | z) spans the top principal directions of *real* data
    covariation between patches -- exactly the structure a diagonal N-prior
    cannot express -- so the inner tractability penalty pulls the recognition
    toward an informative q(z|x), and the EM-style interaction gradient becomes
    nonzero. Greedy layerwise pretraining, in variational-conjugation clothes.

    Rows are rescaled so the median per-node logit std under z ~ N(0, I) is
    ``target_std`` (modulation that matters without saturating the sigmoids).
    """
    n_nodes = model.mid_man.data_dim
    codes = _posterior_codes(model, params, xs[:n_codes])[:, :n_nodes]
    # Node marginals only: for a chordal middle the code vector also carries edge
    # moments, but seeding Theta_ZN's edge rows lets z drive the COUPLINGS --
    # far more potent than node biases, and empirically destabilizing (the CW
    # failure: ||Theta_ZN|| ballooned to 61, decoder degraded). Seed nodes only.
    codes = codes - jnp.mean(codes, axis=0)
    _, _, vt = jnp.linalg.svd(codes, full_matrices=False)
    d = model.top_man.data_dim
    comps = vt[:d]  # (d, n_nodes), orthonormal rows

    im = model.top_var.gen_hrm.int_man
    top, lower_lkl, recog = model.split_coords(params)
    theta_z, top_lkl, rho0 = model.top_var.split_coords(top)
    theta_y, theta_zn = model.top_var.gen_hrm.lkl_fun_man.split_coords(top_lkl)
    mat = im.rep.to_matrix(im.matrix_shape, theta_zn)  # (mid_dim, top_nat_dim)

    loc_block = comps.T  # node i modulated by z along its PCA loadings
    node_std = jnp.linalg.norm(loc_block, axis=1)
    scale = target_std / (jnp.median(node_std) + 1e-12)
    mat = mat.at[:n_nodes, :d].set(scale * loc_block)
    mat = mat.at[n_nodes:, :d].set(0.0)  # edge rows: z must not drive couplings

    theta_zn = im.rep.from_matrix(mat)
    top_lkl = model.top_var.gen_hrm.lkl_fun_man.join_coords(theta_y, theta_zn)
    top = model.top_var.join_coords(theta_z, top_lkl, rho0)
    print(f"  [seed-top-pca] location block set from top-{d} PCA of spike codes "
          f"(median node logit-std {target_std:g})")
    return model.join_coords(top, lower_lkl, recog)


def seed_recognition_regression(
    model: VariationalHierarchical, params: Array, xs: Array, key: Array,
    steps: int = 800, prec: float = 1.0, n_codes: int = 2000,
) -> Array:
    """Pretrain the recognition MLP so q(z | x) tracks the PCA scores of the codes.

    The bilateral half of the symmetry break: regress the MLP's conjugation
    correction so the posterior mean of q(z | x) matches each image's
    (standardized) projection onto the same principal directions used by
    :func:`seed_top_interaction_pca`. Recognition and generative edge then start
    *aligned*, so neither has to discover the other from a zero-gradient state.
    """
    n_nodes = model.mid_man.data_dim
    codes = _posterior_codes(model, params, xs[:n_codes])[:, :n_nodes]  # node marginals
    mean_c = jnp.mean(codes, axis=0)
    cc = codes - mean_c
    _, _, vt = jnp.linalg.svd(cc, full_matrices=False)
    d = model.top_man.data_dim
    scores = cc @ vt[:d].T
    scores = scores / (jnp.std(scores, axis=0) + 1e-6)

    prec_params = model.top_man.cov_man.rep.from_matrix(prec * jnp.eye(d))
    theta_z, _, rho0 = model.split_top(params)
    base = theta_z - rho0

    def target(sc: Array) -> Array:
        return model.top_man.join_location_precision(prec * sc, prec_params) - base

    targets = jax.vmap(target)(scores)
    biases = jax.vmap(lambda x: model.posterior_mid_bias(params, x))(xs[:n_codes])

    top, lower_lkl, recog = model.split_coords(params)
    rho_y, phi = model.recog_man.split_coords(recog)

    def loss(ph: Array, idx: Array) -> Array:
        out = jax.vmap(lambda b: model.mlp_man(ph, b))(biases[idx])
        return jnp.mean((out - targets[idx]) ** 2)

    opt = optax.adam(1e-3)

    @jax.jit
    def step(carry, k):
        ph, st = carry
        idx = jax.random.choice(k, biases.shape[0], (256,))
        val, g = jax.value_and_grad(loss)(ph, idx)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(ph, upd), st), val

    (phi, _), ls = jax.lax.scan(step, (phi, opt.init(phi)), jax.random.split(key, steps))
    print(f"  [seed-recog] MLP regression loss {float(ls[0]):.4f} -> {float(ls[-1]):.4f}")
    recog = model.recog_man.join_coords(rho_y, phi)
    return model.join_coords(top, lower_lkl, recog)


def lift_conv_checkpoint(
    base_model: VariationalHierarchical, model: VariationalHierarchical,
    base_params: Array, key: Array,
) -> Array:
    """Lift a trained conv checkpoint into a larger-kernel geometry.

    The decoder transfers *exactly*: a smaller kernel embeds in a larger one at
    the tap offset ``center_target - center_base`` (taps are ndindex-ordered with
    ``center = kernel_shape // 2``), so the lifted model starts with the
    identical likelihood ``p(x | N)`` -- and since the conjugation parameters are
    closed-form in ``W``, the lower edge stays exact from step 0. This is the
    principled escape from the cold-start decoder saddle: with the exact-N
    estimator there is no gradient noise to kick ``Theta_XY`` off zero, so a
    cold large-kernel model never recruits (K66); initialization must do the
    work, per the layerwise principle.

    Transfers: theta_X, conv kernel (zero-padded taps), theta_Y node biases +
    common chordal edge couplings, Theta_ZN node rows AND common-edge rows
    (fill-in edges/rows start 0), theta*_Z, rho0_Z. Mapping the *trained* edge
    rows is essential: dropping them changes p(y | z) and invalidates the
    transferred conjugation slope rho0_Z (first attempt: Var[rZ] 0.2 -> 227).
    With them, the lifted generative joint IS the base model's, exactly. (The
    node-only principle constrains fresh *seeding*, not preservation of trained
    structure.) The recognition MLP cannot transfer (spike-coordinate dim
    differs) -- glorot init here, re-align with ``--seed-recog``.
    """
    from .lattice_convolution import LatticeConvolution

    conv_b = base_model.lower_hrm.int_man.inner
    conv_t = model.lower_hrm.int_man.inner
    assert isinstance(conv_b, LatticeConvolution) and isinstance(conv_t, LatticeConvolution)
    assert conv_b.in_channels == conv_t.in_channels == 1
    assert conv_b.out_channels == conv_t.out_channels == 1
    kb, kt = conv_b.kernel_shape, conv_t.kernel_shape

    top_b, lower_b, _ = base_model.split_coords(base_params)
    theta_x, theta_xy_b = base_model.lower_hrm.lkl_fun_man.split_coords(lower_b)

    # Kernel: place base taps at the center-aligned offset, zeros elsewhere.
    off = tuple((t // 2) - (b // 2) for t, b in zip(kt, kb))
    kmat = jnp.zeros(kt)
    kmat = kmat.at[
        off[0]:off[0] + kb[0], off[1]:off[1] + kb[1]
    ].set(theta_xy_b.reshape(kb))
    lower_lkl = model.lower_hrm.lkl_fun_man.join_coords(theta_x, kmat.ravel())

    # Top edge: node-level structure transfers verbatim; edges map by identity.
    theta_z, top_lkl_b, rho0 = base_model.top_var.split_coords(top_b)
    theta_y_b, theta_zn_b = base_model.top_var.gen_hrm.lkl_fun_man.split_coords(top_lkl_b)
    n = model.mid_man.data_dim
    diag_b, off_b = base_model.mid_man.split_couplings(theta_y_b)
    eb = base_model.mid_man.junction_tree.chordal_edges_arr
    et = model.mid_man.junction_tree.chordal_edges_arr
    pos_t = {(int(i), int(j)): p for p, (i, j) in enumerate(et)}
    off_t = jnp.zeros(et.shape[0])
    matched = 0
    for p_b, (i, j) in enumerate(eb):
        p = pos_t.get((int(i), int(j))) or pos_t.get((int(j), int(i)))
        if p is not None:
            off_t = off_t.at[p].set(off_b[p_b])
            matched += 1
    theta_y = model.mid_man.join_couplings(diag_b, off_t)

    im_b, im_t = base_model.top_var.gen_hrm.int_man, model.top_var.gen_hrm.int_man
    mat_b = im_b.rep.to_matrix(im_b.matrix_shape, theta_zn_b)
    mat_t = jnp.zeros(im_t.matrix_shape)
    mat_t = mat_t.at[:n, :].set(mat_b[:n, :])  # node rows
    for p_b, (i, j) in enumerate(eb):  # trained edge rows, mapped by edge identity
        p = pos_t.get((int(i), int(j))) or pos_t.get((int(j), int(i)))
        if p is not None:
            mat_t = mat_t.at[n + p, :].set(mat_b[n + p_b, :])
    theta_zn = im_t.rep.from_matrix(mat_t)

    top_lkl = model.top_var.gen_hrm.lkl_fun_man.join_coords(theta_y, theta_zn)
    top = model.top_var.join_coords(theta_z, top_lkl, rho0)

    rho_y = jnp.zeros(model.mid_man.dim)
    phi = model.mlp_man.glorot_initialize(key)
    recog = model.recog_man.join_coords(rho_y, phi)
    print(f"  [lift] kernel {kb} -> {kt} at offset {off}; "
          f"{matched}/{eb.shape[0]} base edges mapped, "
          f"{et.shape[0] - matched} new edges start at 0")
    return model.join_coords(top, lower_lkl, recog)


CHORDAL_WIDTH = 4  # grid width for the chordal middle (treewidth ~= width)


# Conv decoder: coarse latent lattice -> 28x28 image. (7x7, stride 4, kernel 6)
# gives nearest-neighbor induced couplings on the 7x7 latent grid (treewidth ~7).
CONV_IN = (7, 7)
CONV_STRIDE = (4, 4)
CONV_KERNEL = (6, 6)


def build_model(middle: str, n_mid: int, top_dim: int,
                chordal_width: int = CHORDAL_WIDTH,
                couple_edges: bool = False,
                conv_in: tuple[int, int] = CONV_IN,
                conv_stride: tuple[int, int] = CONV_STRIDE,
                conv_kernel: tuple[int, int] = CONV_KERNEL,
                conv_channels: int = 1,
                conv_prior: str = "chordal",
                mlp_hidden: tuple[int, ...] = (128,)) -> VariationalHierarchical:
    obs_man = Normal(N_OBS, Diagonal())
    if middle == "conv":
        # in_lattice * stride must tile the 28x28 image.
        if conv_in[0] * conv_stride[0] != IMG or conv_in[1] * conv_stride[1] != IMG:
            raise ValueError(f"conv_in*stride must equal ({IMG},{IMG})")
        # chordal prior: exact conjugation but treewidth ~x channels (caps C~2-3).
        # diagonal prior: free channels, soft conjugation (see factory docstring).
        return build_conv_boltzmann_gaussian_hierarchy(
            obs_man, conv_in, conv_stride, conv_kernel, top_dim,
            in_channels=conv_channels, prior_graph=conv_prior,
            max_treewidth=2 * max(conv_in) * conv_channels, mlp_hidden=mlp_hidden,
        )
    common = dict(mlp_hidden=(128,), obs_location_only=True, couple_edges=couple_edges)
    if middle == "chordal":
        w = chordal_width
        h = n_mid // w
        return build_boltzmann_gaussian_hierarchy(
            obs_man, h * w, top_dim, mid_kind="chordal",
            mid_edges=grid_edges(h, w), max_treewidth=w + 2, **common,
        )
    return build_boltzmann_gaussian_hierarchy(
        obs_man, n_mid, top_dim, mid_kind=middle, **common
    )


def reconstruct(model: VariationalHierarchical, params: Array, xs: Array, key: Array) -> Array:
    """Posterior-mean reconstruction E_q[y] -> observable mean, in [0,1] pixels."""
    keys = jax.random.split(key, xs.shape[0])
    _, lower_lkl, _ = model.split_coords(params)

    def one(x: Array, k: Array) -> Array:
        ys, _ = model.sample_posterior(k, params, x, 16)
        s_y = jax.vmap(model.mid_man.sufficient_statistic)(ys)
        x_nat = jax.vmap(lambda s: model.lower_hrm.lkl_fun_man(lower_lkl, s))(s_y)
        means = jax.vmap(model.obs_man.to_mean)(x_nat)[:, : model.obs_man.data_dim]
        return jnp.mean(means, axis=0)

    return jax.vmap(one)(xs, keys)


def generative_means(model: VariationalHierarchical, params: Array, key: Array, n: int) -> Array:
    """Generative samples shown as likelihood means: z~p(z), y~p(y|z), E[x|y]."""
    _, lower_lkl, _ = model.split_coords(params)
    theta_z, top_lkl, _ = model.split_top(params)
    kz, ky = jax.random.split(key)
    zs = model.top_man.sample(kz, theta_z, n)

    def one(k: Array, z: Array) -> Array:
        s_z = model.top_man.sufficient_statistic(z)
        y_nat = model.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)
        y = model.mid_man.sample(k, y_nat, 1)[0]
        s_y = model.mid_man.sufficient_statistic(y)
        x_nat = model.lower_hrm.lkl_fun_man(lower_lkl, s_y)
        return model.obs_man.to_mean(x_nat)[: model.obs_man.data_dim]

    return jax.vmap(one)(jax.random.split(ky, n), zs)


def train(model: VariationalHierarchical, train_data: Array, test_data: Array,
          steps: int, key: Array, lambda_y: float = 0.0, lambda_z: float = 0.0,
          lr: float = LR, grad_clip: float = GRAD_CLIP,
          max_var: float = OBS_MAX_VAR,
          batch: int = BATCH, mc_samples: int = MC_SAMPLES,
          reparam_z: bool = False, norm_preserve: bool = False,
          marginal_y: bool = False, eval_n: int = EVAL_N,
          init_params: Array | None = None) -> Array:
    k_init, k_train, k_eval = jax.random.split(key, 3)
    if init_params is None:
        params = model.initialize_from_sample(k_init, train_data, location=0.0, shape=0.3)
        params = init_observation_noise(
            model, params, jnp.mean(train_data, axis=0), jnp.var(train_data, axis=0),
            max_var=max_var,
        )
    else:
        params = init_params  # warm start (e.g. layerwise seeding)
    params = bound_observable_covariance(model, params, max_var=max_var)

    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, LR_WARMUP, steps, end_value=0.0)
    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(grad_clip),
            optax.adamw(schedule, weight_decay=WEIGHT_DECAY),
        ),
        100,
    )
    opt_state = optimizer.init(params)

    # Theta_ZN block bounds, for the norm-preserving projection (A2): the
    # conjugation gradient may reshape the top interaction (drive Var[rZ] down by
    # changing its direction) but not shrink it -- removing the trivial
    # Theta_ZN -> 0 exit from the conjugation objective.
    zn_s = model.top_man.dim + model.mid_man.dim
    zn_e = model.top_man.dim + model.top_var.gen_hrm.lkl_fun_man.dim

    def loss_main(p: Array, k: Array, batch_xs: Array) -> Array:
        ke, ki = jax.random.split(k)
        if marginal_y:  # exact-N estimator: pathwise z-gradients only
            elbo = model.mean_marginal_elbo(ke, p, batch_xs, mc_samples)
        else:
            elbo = model.mean_elbo(ke, p, batch_xs, mc_samples, reparam_z=reparam_z)
        inner = model.mean_recognition_inner_loss(ki, p, batch_xs, mc_samples)
        return -elbo + LAMBDA_INNER * inner

    def loss_conj(p: Array, k: Array) -> Array:
        var_r_y, var_r_z = model.prior_conjugation_loss_components(k, p, CONJ_SAMPLES)
        return lambda_y * var_r_y + lambda_z * var_r_z  # bottom / top edge, weighted

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
        p = bound_observable_covariance(model, optax.apply_updates(p, updates), max_var=max_var)
        p = bound_top_prior(model, p)  # keep theta*_Z in the PD cone (finite != valid)
        p_safe = jnp.where(jnp.all(jnp.isfinite(p)), p, p_safe)  # last all-finite params
        return (p, opt_state, k, p_safe), None

    log_every = max(1, steps // 20)
    carry = (params, opt_state, k_train, params)
    k_etr, k_ete = jax.random.split(k_eval)
    t0 = time.time()
    for c in range(steps // log_every):
        gs = jnp.arange(c * log_every, (c + 1) * log_every)
        carry, _ = jax.lax.scan(step, carry, gs)
        params = carry[3]  # evaluate the last-finite snapshot
        if marginal_y:
            etr = float(model.mean_marginal_elbo(k_etr, params, train_data[:eval_n], 8))
            ete = float(model.mean_marginal_elbo(k_ete, params, test_data[:eval_n], 8))
        else:
            etr = float(model.mean_elbo(k_etr, params, train_data[:eval_n], 8))
            ete = float(model.mean_elbo(k_ete, params, test_data[:eval_n], 8))
        vry, vrz = model.prior_conjugation_loss_components(k_etr, params, eval_n)
        theta_x, _ = model.split_lower(params)
        _, prec = model.obs_man.split_location_precision(theta_x)
        var = 1.0 / jnp.asarray(prec)
        _, txy = model.split_lower(params)  # bottom decoder liveness
        _, top_lkl, _ = model.split_top(params)  # top edge liveness (collapse check)
        _, theta_zn = model.top_var.gen_hrm.lkl_fun_man.split_coords(top_lkl)
        txn = float(jnp.linalg.norm(txy))
        tzn = float(jnp.linalg.norm(theta_zn))
        live = "" if bool(jnp.all(jnp.isfinite(carry[0]))) else "  [live=NaN, using snapshot]"
        print(f"  step {(c+1)*log_every:5d}  ELBO train {etr:8.2f}  test {ete:8.2f}  "
              f"Var[rY] {float(vry):6.2f}  Var[rZ] {float(vrz):6.3f}  "
              f"|Theta_XY| {txn:6.2f}  |Theta_ZN| {tzn:6.3f}  "
              f"Psi[{float(var.min()):.3f},{float(var.max()):.3f}]  ({time.time()-t0:.0f}s){live}")
    return carry[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--middle", default="chain", choices=["diagonal", "chain", "chordal", "conv"])
    ap.add_argument("--n-mid", type=int, default=N_MID)
    ap.add_argument("--top-dim", type=int, default=TOP_DIM)
    ap.add_argument("--chordal-width", type=int, default=CHORDAL_WIDTH,
                    help="grid width for the chordal middle (treewidth ~= width; denser absorbs more)")
    ap.add_argument("--conv-in", type=int, nargs=2, default=list(CONV_IN),
                    help="(conv middle) coarse latent lattice HxW; conv_in*stride must tile 28x28")
    ap.add_argument("--conv-stride", type=int, nargs=2, default=list(CONV_STRIDE),
                    help="(conv middle) upsampling stride per axis")
    ap.add_argument("--conv-kernel", type=int, nargs=2, default=list(CONV_KERNEL),
                    help="(conv middle) transposed-conv kernel extent per axis")
    ap.add_argument("--conv-channels", type=int, default=1,
                    help="(conv middle) latent channels per cell = number of shared kernels "
                    "(decoder capacity); scales treewidth by this factor")
    ap.add_argument("--conv-prior", default="chordal", choices=["chordal", "diagonal"],
                    help="(conv middle) 'chordal' = exact conjugation (treewidth ~x channels); "
                    "'diagonal' = free channels + soft conjugation")
    ap.add_argument("--mlp-hidden", type=int, nargs="+", default=[128],
                    help="hidden widths of the amortized recognition MLP")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="minibatch size (shrink for high-treewidth middles to bound memory)")
    ap.add_argument("--mc", type=int, default=MC_SAMPLES, help="ELBO Monte Carlo samples")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--lambda-gen", type=float, default=LAMBDA_GEN,
                    help="shorthand: sets both --lambda-y and --lambda-z unless overridden")
    ap.add_argument("--lambda-y", type=float, default=None,
                    help="bottom-edge (Gaussian-Boltzmann) conjugation weight")
    ap.add_argument("--lambda-z", type=float, default=None,
                    help="top-edge (Boltzmann pop-code Y|Z) conjugation weight")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    ap.add_argument("--max-var", type=float, default=OBS_MAX_VAR,
                    help="observable variance cap (smaller = tamer quadratic psi_X)")
    ap.add_argument("--marginal-y", action="store_true",
                    help="exact-N ELBO (conv middles only): spikes integrate out of the "
                    "residual (r_Y == 0), z is pathwise -- no score-function estimator")
    ap.add_argument("--eval-n", type=int, default=EVAL_N,
                    help="diagnostic eval batch (shrink for high-treewidth middles: "
                    "eval vmaps eval_n x 8 junction-tree passes)")
    ap.add_argument("--reparam-z", action="store_true",
                    help="pathwise (reparameterized) gradient for the Gaussian top latent")
    ap.add_argument("--norm-preserve", action="store_true",
                    help="project the conjugation gradient off Theta_ZN's radial direction "
                    "(shape-only; forbids the trivial Theta_ZN -> 0 exit)")
    ap.add_argument("--resume", default="",
                    help="npz checkpoint to warm-start from (model config must match)")
    ap.add_argument("--lift-from", default="",
                    help="npz checkpoint of a SMALLER-kernel conv model to lift into this "
                    "geometry (decoder transfers exactly; combine with --seed-recog)")
    ap.add_argument("--lift-kernel", type=int, nargs=2, default=None,
                    help="kernel extent of the --lift-from checkpoint's model")
    ap.add_argument("--seed-top-pca", action="store_true",
                    help="with --resume: re-seed Theta_ZN location block from PCA of spike codes")
    ap.add_argument("--seed-std", type=float, default=1.5,
                    help="median per-node logit std of the PCA seeding (lower for many nodes)")
    ap.add_argument("--seed-recog", action="store_true",
                    help="with --resume: pretrain the recognition MLP to track the PCA scores")
    ap.add_argument("--outdir", default="", help="subfolder under results/ for this run")
    ap.add_argument("--note", default="", help="one-line description for the INDEX")
    ap.add_argument("--couple-edges", action="store_true",
                    help="couple X to the full Boltzmann stat (nodes+edges); default is node-only")
    args = ap.parse_args()
    lambda_y = args.lambda_gen if args.lambda_y is None else args.lambda_y
    lambda_z = args.lambda_gen if args.lambda_z is None else args.lambda_z

    key = jax.random.PRNGKey(0)
    _, k_train, k_rec, k_gen = jax.random.split(key, 4)
    train_data, test_data = load_mnist(args.n_train, N_TEST)
    print(f"MNIST: train {train_data.shape}, test {test_data.shape}")
    print(f"Model: X(Normal-{N_OBS}) <- Y(Boltzmann-{args.n_mid}, {args.middle}) "
          f"<- Z(Gaussian-{args.top_dim})")

    print(f"conjugation: lambda_y={lambda_y} lambda_z={lambda_z}  "
          f"lr={args.lr} grad_clip={args.grad_clip} max_var={args.max_var}")
    model = build_model(args.middle, args.n_mid, args.top_dim, args.chordal_width,
                        couple_edges=args.couple_edges,
                        conv_in=tuple(args.conv_in), conv_stride=tuple(args.conv_stride),
                        conv_kernel=tuple(args.conv_kernel), conv_channels=args.conv_channels,
                        conv_prior=args.conv_prior, mlp_hidden=tuple(args.mlp_hidden))
    if args.middle == "conv":
        args.n_mid = model.mid_man.data_dim  # determined by the conv lattice
        print(f"conv decoder {tuple(args.conv_in)} -> ({IMG},{IMG}) (stride "
              f"{tuple(args.conv_stride)}, kernel {tuple(args.conv_kernel)}); "
              f"latent nodes={args.n_mid}, mid dim={model.mid_man.dim}")
    init_params = None
    if args.lift_from:
        assert args.lift_kernel is not None, "--lift-from requires --lift-kernel"
        base_model = build_model(
            "conv", args.n_mid, args.top_dim, conv_in=tuple(args.conv_in),
            conv_stride=tuple(args.conv_stride), conv_kernel=tuple(args.lift_kernel),
            conv_channels=args.conv_channels, conv_prior=args.conv_prior,
            mlp_hidden=tuple(args.mlp_hidden))
        base_params = jnp.asarray(np.load(args.lift_from)["params"])
        print(f"lift from {args.lift_from}")
        init_params = lift_conv_checkpoint(
            base_model, model, base_params, jax.random.PRNGKey(7))
    elif args.resume:
        init_params = jnp.asarray(np.load(args.resume)["params"])
        print(f"warm start from {args.resume}")
    if init_params is not None:
        if args.seed_top_pca:
            init_params = seed_top_interaction_pca(
                model, init_params, train_data, target_std=args.seed_std)
        if args.seed_recog:
            init_params = seed_recognition_regression(
                model, init_params, train_data, jax.random.PRNGKey(42))
    params = train(model, train_data, test_data, args.steps, k_train, lambda_y, lambda_z,
                   lr=args.lr, grad_clip=args.grad_clip, max_var=args.max_var,
                   batch=args.batch, mc_samples=args.mc,
                   reparam_z=args.reparam_z, norm_preserve=args.norm_preserve,
                   marginal_y=args.marginal_y, eval_n=args.eval_n,
                   init_params=init_params)

    # Final metrics
    recons = reconstruct(model, params, test_data[:8], k_rec)
    mse = float(jnp.mean((test_data[:8] - recons) ** 2))
    gens = generative_means(model, params, k_gen, 8)
    if args.marginal_y:
        ete = float(model.mean_marginal_elbo(k_rec, params, test_data[:args.eval_n], 8))
    else:
        ete = float(model.mean_elbo(k_rec, params, test_data[:args.eval_n], 8))
    vry, vrz = model.prior_conjugation_loss_components(k_gen, params, args.eval_n)
    vry, vrz = float(vry), float(vrz)
    print(f"reconstruction MSE (8 test digits): {mse:.4f}  ELBO test {ete:.2f}  "
          f"Var[rY] {vry:.2f}  Var[rZ] {vrz:.3f}")

    # Figure: originals / reconstructions / generative samples
    fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
    for j in range(8):
        axes[0, j].imshow(np.array(test_data[j]).reshape(IMG, IMG), cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(np.array(recons[j]).reshape(IMG, IMG), cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(np.clip(np.array(gens[j]).reshape(IMG, IMG), 0, 1), cmap="gray", vmin=0, vmax=1)
        for i in range(3):
            axes[i, j].axis("off")
    axes[0, 0].set_title("data", loc="left")
    axes[1, 0].set_title("reconstruction", loc="left")
    axes[2, 0].set_title("generative samples", loc="left")
    fig.suptitle(f"MNIST {args.middle} n_mid={args.n_mid}  lam_y={lambda_y} lam_z={lambda_z}  "
                 f"MSE {mse:.3f}  Var[rY]={vry:.1f} Var[rZ]={vrz:.2f}")
    fig.tight_layout()

    results_dir = example_paths(__file__).results_dir
    if args.outdir:
        results_dir = results_dir / args.outdir
    results_dir.mkdir(parents=True, exist_ok=True)
    wtag = f"_w{args.chordal_width}" if args.middle == "chordal" else ""
    if args.middle == "conv":
        ci, cs, ck = tuple(args.conv_in), tuple(args.conv_stride), tuple(args.conv_kernel)
        wtag = (f"_{ci[0]}x{ci[1]}s{cs[0]}k{ck[0]}_C{args.conv_channels}_{args.conv_prior}")
    ctag = "" if args.middle == "conv" else ("_edges" if args.couple_edges else "_nodes")
    mtag = "x".join(str(h) for h in args.mlp_hidden)
    etag = (("_mg" if args.marginal_y else "")
            + ("_rp" if args.reparam_z else "") + ("_np" if args.norm_preserve else ""))
    if args.resume:
        etag += "_ws" + ("P" if args.seed_top_pca else "") + ("R" if args.seed_recog else "")
    if args.lift_from:
        etag += "_lift" + ("R" if args.seed_recog else "")
    tag = (f"{args.middle}{wtag}{ctag}_n{args.n_mid}_td{args.top_dim}_ly{lambda_y:g}_lz{lambda_z:g}"
           f"_lr{args.lr:g}_gc{args.grad_clip:g}_mv{args.max_var:g}_mlp{mtag}{etag}_st{args.steps}")
    out = results_dir / f"mnist_hierarchical_{tag}.png"
    fig.savefig(out, dpi=130)
    np.savez(out.with_suffix(".npz"), params=np.asarray(params))  # for post-hoc analysis
    _append_index(results_dir, tag, out.name, args, lambda_y, lambda_z, mse, ete, vry, vrz)
    print(f"saved {out}")


def _append_index(results_dir, tag, fname, args, lambda_y, lambda_z, mse, ete, vry, vrz) -> None:
    """Append a row to INDEX.md in ``results_dir`` describing this run.

    Keeps a human-readable manifest so runs are identifiable by config + metrics
    rather than by file mtime.
    """
    from datetime import datetime

    index = results_dir / "INDEX.md"
    header = ("| when | middle | n_mid | lam_y | lam_z | lr | clip | max_var "
              "| ELBO test | MSE | Var[rY] | Var[rZ] | figure | note |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    if not index.exists():
        index.write_text("# MNIST hierarchical runs\n\n" + header)
    when = datetime.now().strftime("%m-%d %H:%M")
    row = (f"| {when} | {args.middle} | {args.n_mid} | {lambda_y:g} | {lambda_z:g} "
           f"| {args.lr:g} | {args.grad_clip:g} | {args.max_var:g} "
           f"| {ete:.1f} | {mse:.4f} | {vry:.1f} | {vrz:.3f} | {fname} | {args.note} |\n")
    with index.open("a") as f:
        f.write(row)


if __name__ == "__main__":
    main()
