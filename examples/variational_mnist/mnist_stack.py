"""Stacked hierarchy on MNIST: a second Gaussian->Boltzmann unit on layer-1 codes.

The grant-story architecture, made concrete by the validated layerwise protocol:

    unit 1 (trained, frozen):  X (784 px) <- N1 (conv Boltzmann) <- Z1 (Gauss 16)
    unit 2 (this script):      Z1 codes   <- N2 (full Boltzmann)  <- Z2 (Gauss)

Unit 2 is the *same model class* as unit 1 -- a :class:`VariationalHierarchical`
whose observable is the 16-dim continuous code space of unit 1's recognition
posterior means q(z1 | x). Its dense 16x16 decoder induces the complete coupling
graph on N2, whose "junction tree" is a single clique -- exact enumeration over
2^n2 states -- so the lower edge is *exactly conjugate* by the same closed form
as the conv edges (:class:`ConvChordalBoltzmannHarmonium` is generic in the
interaction map), and the exact-N marginal ELBO applies verbatim.

Layerwise ingredients (all previously validated):

- codes from the frozen unit-1 checkpoint (posterior means of q(z1 | x)),
- PCA seeding of the unit-2 *decoder* (cold decoders are dead under the exact
  estimator -- initialization must do the work),
- ``MH.train`` reused unchanged (marginal estimator + lambda_z, no surgery).

Outputs the full-stack ancestral sampling figure: z2 ~ p(z2), n2 ~ p(n2|z2),
z1 ~ p(z1|n2), n1 ~ p(n1|z1), E[x|n1] -- against unit 1's own prior samples and
real digits. If the stack works, the top-of-stack samples should be *more*
digit-like than unit 1's, because z2/N2 impose global structure on z1 that
unit 1's unimodal prior cannot.

Run::

    uv run python -m examples.variational_mnist.mnist_stack \\
        --layer1 CKPT.npz --marginal-y --lambda-z 1.0 --steps 10000
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

jax.config.update("jax_enable_x64", True)

from goal.geometry import Diagonal, EmbeddedMap, Rectangular  # noqa: E402
from goal.models import ChordalBoltzmann, Normal, full_normal  # noqa: E402
from goal.models.harmonium.lgm import (  # noqa: E402
    GeneralizedGaussianLocationEmbedding,
)
from goal.models.harmonium.population_codes import (  # noqa: E402
    BoltzmannNormalHarmonium,
    BoltzmannPopulationCode,
)
from goal.geometry.manifold.map import MultilayerPerceptron  # noqa: E402

from ..shared import example_paths  # noqa: E402
from . import mnist_hierarchical as MH  # noqa: E402,N812
from .hierarchical import (  # noqa: E402
    BoltzmannNodeEmbedding,
    ConvChordalBoltzmannHarmonium,
    HierarchicalRecognition,
    VariationalHierarchical,
)

IMG = MH.IMG

# Unit-2 geometry: 16-dim codes, complete graph on n2 nodes (single-clique JT,
# exact enumeration over 2^n2 states -- keep n2 <= ~14 for memory).
N2_NODES = 12
Z2_DIM = 8
# Codes live on a z-scale (per-dim variance up to a few), not a pixel scale:
CODE_MAX_VAR = 2.0
CODE_MIN_VAR = 1e-3


def build_unit2(
    code_dim: int, n_nodes: int, top_dim: int, mlp_hidden: tuple[int, ...] = (64,)
) -> VariationalHierarchical:
    """Unit 2: Normal(code_dim) <- complete-graph Boltzmann(n_nodes) <- Normal(top_dim).

    The dense decoder couples every node to every code coordinate, so the
    conjugation parameter ``W^T Sigma W`` is dense -- the complete graph is its
    exact support, and the closed-form conjugation applies (r_Y == 0).
    """
    obs_man = Normal(code_dim, Diagonal())
    edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    mid_man = ChordalBoltzmann.from_edges(n_nodes, edges, max_treewidth=n_nodes)
    top_var = BoltzmannPopulationCode(BoltzmannNormalHarmonium(mid_man, top_dim))

    lower_int = EmbeddedMap(
        Rectangular(),
        BoltzmannNodeEmbedding(mid_man),
        GeneralizedGaussianLocationEmbedding(obs_man),
    )
    lower_hrm = ConvChordalBoltzmannHarmonium(lower_int, mid_man)

    mlp = MultilayerPerceptron(full_normal(top_dim), mid_man, mlp_hidden, jax.nn.gelu)
    recog = HierarchicalRecognition(mid_man, mlp)
    return VariationalHierarchical(
        top_var=top_var, lower_hrm=lower_hrm, recog_man=recog
    )


def layer1_codes(
    model1: VariationalHierarchical, params1: Array, xs: Array, chunk: int = 256
) -> Array:
    """Recognition-posterior means of q(z1 | x) -- unit 2's observable data."""
    d = model1.top_man.data_dim

    def one(x: Array) -> Array:
        q = model1.approximate_posterior_top(params1, x)
        loc, prec_params = model1.top_man.split_location_precision(q)
        prec = model1.top_man.cov_man.rep.to_matrix((d, d), prec_params)
        return jnp.linalg.solve(0.5 * (prec + prec.T), loc)

    f = jax.jit(jax.vmap(one))
    return jnp.concatenate([f(xs[i : i + chunk]) for i in range(0, xs.shape[0], chunk)])


def seed_decoder_pca(
    model: VariationalHierarchical, params: Array, codes: Array
) -> Array:
    """Seed unit 2's dense decoder from the PCA of the codes (FA-style).

    Column i of W (node i's loading) = sqrt(eigval_i) * v_i, so each binary node
    toggles one principal direction of the code distribution at its natural
    scale; the observable bias is re-centered so that E[z1] at p(n2_i = 1/2)
    matches the code mean. This is the cold-start saddle breaker for the lower
    edge -- without it the exact estimator leaves the decoder at zero gradient.
    """
    n = model.mid_man.data_dim
    d = model.obs_man.data_dim
    mu = jnp.mean(codes, axis=0)
    cc = codes - mu
    _, s, vt = jnp.linalg.svd(cc, full_matrices=False)
    scales = s[:n] / jnp.sqrt(codes.shape[0])  # component stds
    w = vt[:n].T * scales  # (d, n), zero-padded implicitly if n > d
    if n > d:
        w = jnp.concatenate([w, jnp.zeros((d, n - d))], axis=1)

    top, lower_lkl, recog = model.split_coords(params)
    theta_x, _ = model.lower_hrm.lkl_fun_man.split_coords(lower_lkl)
    _, prec = model.obs_man.split_location_precision(theta_x)
    # Bias location: with y in {0,1}, E[z1] = mu requires loc_mean = mu - W E[y];
    # take E[y] = 1/2 at the zero-bias start.
    loc_mean = mu - 0.5 * jnp.sum(w, axis=1)
    theta_x = model.obs_man.join_location_precision(jnp.asarray(prec) * loc_mean, prec)
    lower_lkl = model.lower_hrm.lkl_fun_man.join_coords(theta_x, w.ravel())
    print(
        f"  [seed-decoder-pca] W from top-{n} PCA of codes "
        f"(component stds {float(scales.min()):.2f}..{float(scales.max()):.2f})"
    )
    return model.join_coords(top, lower_lkl, recog)


def stack_generate(
    model1: VariationalHierarchical,
    params1: Array,
    model2: VariationalHierarchical,
    params2: Array,
    key: Array,
    n: int,
) -> Array:
    """Full-stack ancestral samples: z2 -> n2 -> z1 -> n1 -> E[x | n1]."""
    k2, k1y, k1x = jax.random.split(key, 3)
    joint2 = model2.sample(k2, params2, n)  # [z1_code, n2, z2]
    z1s = joint2[:, : model2.obs_man.data_dim]

    _, top_lkl1, _ = model1.split_top(params1)
    _, lower_lkl1, _ = model1.split_coords(params1)

    def decode(k: Array, z1: Array) -> Array:
        s_z = model1.top_man.sufficient_statistic(z1)
        y = model1.mid_man.sample(
            k, model1.top_var.gen_hrm.lkl_fun_man(top_lkl1, s_z), 1
        )[0]
        s_y = model1.mid_man.sufficient_statistic(y)
        x_nat = model1.lower_hrm.lkl_fun_man(lower_lkl1, s_y)
        return model1.obs_man.to_mean(x_nat)[: model1.obs_man.data_dim]

    return jax.vmap(decode)(jax.random.split(k1y, n), z1s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layer1",
        required=True,
        help="unit-1 npz checkpoint (conv model; config via the conv args)",
    )
    ap.add_argument("--conv-in", type=int, nargs=2, default=list(MH.CONV_IN))
    ap.add_argument("--conv-stride", type=int, nargs=2, default=list(MH.CONV_STRIDE))
    ap.add_argument("--conv-kernel", type=int, nargs=2, default=[6, 4])
    ap.add_argument("--conv-prior", default="chordal", choices=["chordal", "diagonal"])
    ap.add_argument("--top-dim", type=int, default=16, help="unit-1 top dim = code dim")
    ap.add_argument("--n2", type=int, default=N2_NODES)
    ap.add_argument("--z2-dim", type=int, default=Z2_DIM)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--n-train", type=int, default=MH.N_TRAIN)
    ap.add_argument("--lambda-z", type=float, default=1.0)
    ap.add_argument("--marginal-y", action="store_true")
    ap.add_argument("--lr", type=float, default=MH.LR)
    ap.add_argument("--max-var", type=float, default=CODE_MAX_VAR)
    ap.add_argument("--outdir", default="stack")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)
    k_train, k_gen = jax.random.split(key)
    train_x, test_x = MH.load_mnist(args.n_train, MH.N_TEST)

    model1 = MH.build_model(
        "conv",
        0,
        args.top_dim,
        conv_in=tuple(args.conv_in),
        conv_stride=tuple(args.conv_stride),
        conv_kernel=tuple(args.conv_kernel),
        conv_prior=args.conv_prior,
    )
    params1 = jnp.asarray(np.load(args.layer1)["params"])
    print(f"unit 1: {args.layer1}")

    train_codes = layer1_codes(model1, params1, train_x)
    test_codes = layer1_codes(model1, params1, test_x)
    print(
        f"codes: train {train_codes.shape}, per-dim std "
        f"{float(jnp.std(train_codes, axis=0).min()):.2f}.."
        f"{float(jnp.std(train_codes, axis=0).max()):.2f}"
    )

    model2 = build_unit2(args.top_dim, args.n2, args.z2_dim)
    print(
        f"unit 2: Z1({args.top_dim}) <- N2(Boltzmann-{args.n2}, complete) "
        f"<- Z2(Gaussian-{args.z2_dim}); mid dim={model2.mid_man.dim}"
    )

    # Exactness self-check: the dense lower edge must be exactly conjugate.
    p0 = model2.initialize(jax.random.PRNGKey(1), 0.0, 0.3)
    ys = jax.random.bernoulli(jax.random.PRNGKey(2), 0.5, (32, args.n2)).astype(float)
    max_ry = float(
        jnp.max(jnp.abs(jax.vmap(lambda y: model2.residual_lower(p0, y))(ys)))
    )
    print(f"unit-2 lower edge max|r_Y| = {max_ry:.2e}")
    assert max_ry < 1e-10, "dense lower edge not exactly conjugate"

    # Init: from-code observable stats + PCA decoder seeding (saddle breaker).
    params2 = model2.initialize_from_sample(
        jax.random.PRNGKey(3), train_codes, location=0.0, shape=0.3
    )
    params2 = MH.init_observation_noise(
        model2,
        params2,
        jnp.mean(train_codes, axis=0),
        jnp.var(train_codes, axis=0),
        min_var=CODE_MIN_VAR,
        max_var=args.max_var,
    )
    params2 = seed_decoder_pca(model2, params2, train_codes)

    params2 = MH.train(
        model2,
        train_codes,
        test_codes,
        args.steps,
        k_train,
        lambda_y=0.0,
        lambda_z=args.lambda_z,
        lr=args.lr,
        max_var=args.max_var,
        marginal_y=args.marginal_y,
        init_params=params2,
    )

    # Metrics: code reconstruction + Z2 liveness through the FULL stack.
    ev = model2.mean_marginal_elbo if args.marginal_y else model2.mean_elbo
    e2 = float(ev(k_gen, params2, test_codes[:64], 8))
    _, vrz2 = model2.prior_conjugation_loss_components(k_gen, params2, 64)
    print(f"unit-2 ELBO test {e2:.2f}  Var[rZ2] {float(vrz2):.3f}")

    # Figure: data / unit-1 prior samples / full-stack samples.
    n_show = 10
    gens1 = MH.generative_means(model1, params1, jax.random.fold_in(k_gen, 1), n_show)
    gens_stack = stack_generate(
        model1, params1, model2, params2, jax.random.fold_in(k_gen, 2), n_show
    )
    fig, axes = plt.subplots(3, n_show, figsize=(1.2 * n_show, 3.8))
    rows = [np.array(test_x[:n_show]), np.array(gens1), np.array(gens_stack)]
    labels = ["data", "unit-1 prior", "stack z2->x"]
    for r in range(3):
        for j in range(n_show):
            axes[r, j].imshow(
                np.clip(rows[r][j].reshape(IMG, IMG), 0, 1), cmap="gray", vmin=0, vmax=1
            )
            axes[r, j].axis("off")
        axes[r, 0].set_title(labels[r], loc="left", fontsize=9)
    fig.suptitle(
        f"Stacked hierarchy: n2={args.n2} z2={args.z2_dim}  "
        f"unit-2 ELBO {e2:.1f}  Var[rZ2] {float(vrz2):.2f}"
    )
    fig.tight_layout()

    results_dir = example_paths(__file__).results_dir / args.outdir
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"n2{args.n2}_z2{args.z2_dim}_lz{args.lambda_z:g}"
        + ("_mg" if args.marginal_y else "")
        + f"_st{args.steps}"
    )
    out = results_dir / f"mnist_stack_{tag}.png"
    fig.savefig(out, dpi=130)
    np.savez(out.with_suffix(".npz"), params=np.asarray(params2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
