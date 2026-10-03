"""K -> N -> X on MNIST: a categorical directly over a conjugate spike code.

The architecture that avoids posterior collapse by construction:

    p(x, N, k) = p(x | N) . p(N | k) . p(k),

    K -- categorical cluster (integrated out exactly, no amortized KL),
    N -- diagonal Boltzmann spike code (Bernoulli per unit, per cluster),
    X -- diagonal-Gaussian pixels, decoded by a NON-overlapping lattice conv.

Because the conv is non-overlapping (kernel == stride), the columns of the
decoder have disjoint support, so ``1/2 W^T Sigma W`` is diagonal and the diagonal
spike prior is *exactly* conjugate to the Gaussian likelihood (validated to
machine precision). Each component's marginal ``log p_k(x)`` is therefore the
inherited analytic ``log_observable_density``, and the model marginal is just

    log p(x) = logsumexp_k [ log pi_k + log p_k(x) ].

No variational bound, no amortized recognition, no continuous latent above the
bottleneck -- so the collapse that killed K -> Z -> N -> X isn't in the model.
The categorical is recruited iff the spike-code distribution is multimodal
(which, for MNIST digits, it is).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import Array

from goal.geometry import Diagonal
from goal.models import Bernoullis, DiagonalBoltzmann, Normal
from goal.models.harmonium.lgm import GeneralizedGaussianLocationEmbedding

from .hierarchical import BoltzmannNodeEmbedding, ConvBoltzmannHarmonium
from .lattice_convolution import EmbeddedLinearMap, LatticeConvolution
from .mnist_hierarchical import IMG, N_OBS, example_paths, load_mnist


def build_edge(
    in_lat, stride, kernel
) -> tuple[ConvBoltzmannHarmonium, Normal[Diagonal]]:
    n_nodes = int(np.prod(in_lat))
    obs_man = Normal(N_OBS, Diagonal())
    conv = LatticeConvolution.create(
        Bernoullis(n_nodes), obs_man.loc_man, in_lat, stride, kernel, 1, 1
    )
    lat = DiagonalBoltzmann(n_neurons=n_nodes)
    int_man = EmbeddedLinearMap(
        conv, BoltzmannNodeEmbedding(lat), GeneralizedGaussianLocationEmbedding(obs_man)
    )
    return ConvBoltzmannHarmonium(int_man, lat), obs_man


# --- K -> N -> X mixture (hand-rolled: shared decoder, per-cluster spike prior) ---


@dataclass(frozen=True)
class SpikeMixture:
    """Parameters: shared ``(obs_bias, int)`` decoder + K spike priors + K logits."""

    edge: ConvBoltzmannHarmonium
    n_clusters: int

    @property
    def obs_man(self) -> Normal[Diagonal]:
        return self.edge.obs_man

    @property
    def lat_man(self) -> DiagonalBoltzmann:
        return self.edge.lat_man

    def split(self, params: Array) -> tuple[Array, Array, Array]:
        """(lkl_params, lat_biases[K, n], logits[K])."""
        lkl_dim = self.edge.lkl_fun_man.dim
        n = self.lat_man.dim
        lkl = params[:lkl_dim]
        lat = params[lkl_dim : lkl_dim + self.n_clusters * n].reshape(
            self.n_clusters, n
        )
        logits = params[lkl_dim + self.n_clusters * n :]
        return lkl, lat, logits

    def log_prob(self, params: Array, x: Array) -> Array:
        lkl, lat, logits = self.split(params)
        obs_bias, int_p = self.edge.lkl_fun_man.split_coords(lkl)
        log_pi = jax.nn.log_softmax(logits)

        def comp(theta_n: Array) -> Array:
            hp = self.edge.join_coords(obs_bias, int_p, theta_n)
            return self.edge.log_observable_density(hp, x)

        return jax.scipy.special.logsumexp(log_pi + jax.vmap(comp)(lat))

    def cluster_mean_images(self, params: Array) -> Array:
        """E[x | N = E[N|k]] per cluster (mean-spike decode, noise-free)."""
        lkl, lat, _ = self.split(params)

        def one(theta_n: Array) -> Array:
            s_n = self.lat_man.to_mean(theta_n)  # spike probabilities
            x_nat = self.edge.lkl_fun_man(lkl, s_n)
            return self.obs_man.to_mean(x_nat)[:N_OBS]

        return jax.vmap(one)(lat)

    def sample_cluster(self, key: Array, params: Array, k: int, n: int) -> Array:
        lkl, lat, _ = self.split(params)
        theta_n = lat[k]
        ns = self.lat_man.sample(key, theta_n, n)

        def dec(nn: Array) -> Array:
            s_n = self.lat_man.sufficient_statistic(nn)
            return self.obs_man.to_mean(self.edge.lkl_fun_man(lkl, s_n))[:N_OBS]

        return jax.vmap(dec)(ns)

    def initialize(
        self, key: Array, data: Array, min_var: float, max_var: float
    ) -> Array:
        k_int, k_lat = jax.random.split(key)
        obs_man, lat_man = self.obs_man, self.lat_man
        # observable bias from data statistics (fixed diagonal noise)
        var = jnp.clip(jnp.var(data, axis=0), min_var, max_var)
        prec = 1.0 / var
        obs_bias = obs_man.join_location_precision(jnp.mean(data, axis=0) * prec, prec)
        int_p = 0.01 * jax.random.normal(
            k_int, (self.edge.lkl_fun_man.dim - obs_man.dim,)
        )
        lkl = self.edge.lkl_fun_man.join_coords(obs_bias, int_p)
        # break symmetry across clusters
        lat = 0.3 * jax.random.normal(k_lat, (self.n_clusters, lat_man.dim))
        logits = jnp.zeros(self.n_clusters)
        return jnp.concatenate([lkl, lat.reshape(-1), logits])


def train(
    model: SpikeMixture,
    data: Array,
    steps: int,
    key: Array,
    lr: float,
    min_var: float,
    max_var: float,
) -> Array:
    params = model.initialize(key, data, min_var, max_var)
    warmup = min(200, max(1, steps // 10))
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, end_value=0.0)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4)
    )
    opt_state = opt.init(params)

    lkl_dim = model.edge.lkl_fun_man.dim
    obs_man = model.obs_man

    def bound_var(p: Array) -> Array:
        lkl, lat, logits = model.split(p)
        obs_bias, int_p = model.edge.lkl_fun_man.split_coords(lkl)
        loc, prec = obs_man.split_location_precision(obs_bias)
        prec = jnp.clip(jnp.asarray(prec), 1.0 / max_var, 1.0 / min_var)
        obs_bias = obs_man.join_location_precision(loc, prec)
        lkl = model.edge.lkl_fun_man.join_coords(obs_bias, int_p)
        return jnp.concatenate([lkl, lat.reshape(-1), logits])

    def loss_fn(p: Array, xs: Array) -> Array:
        return -jnp.mean(jax.vmap(lambda x: model.log_prob(p, x))(xs))

    @jax.jit
    def step(carry, _):
        p, os, k = carry
        k, kb = jax.random.split(k)
        xs = data[jax.random.choice(kb, data.shape[0], (128,))]
        loss, g = jax.value_and_grad(loss_fn)(p, xs)
        upd, os = opt.update(g, os, p)
        p = bound_var(optax.apply_updates(p, upd))
        return (p, os, k), loss

    log_every = max(1, steps // 15)
    carry = (params, opt_state, key)
    t0 = time.time()
    for c in range(steps // log_every):
        carry, losses = jax.lax.scan(step, carry, None, length=log_every)
        p = carry[0]
        _, _, logits = model.split(p)
        pk = np.asarray(jax.nn.softmax(logits))
        print(
            f"  step {(c + 1) * log_every:5d}  -logp {float(losses[-1]):8.2f}  "
            f"p(k) max {pk.max():.3f} min {pk.min():.3f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return carry[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-lat", type=int, nargs=2, default=[14, 14])
    ap.add_argument("--stride", type=int, nargs=2, default=[2, 2])
    ap.add_argument(
        "--kernel", type=int, nargs=2, default=[2, 2]
    )  # == stride: non-overlap
    ap.add_argument("--clusters", type=int, default=20)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--min-var", type=float, default=1e-3)
    ap.add_argument("--max-var", type=float, default=0.15)
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)
    train_data, test_data = load_mnist(args.n_train, 2000)
    edge, _ = build_edge(tuple(args.in_lat), tuple(args.stride), tuple(args.kernel))
    model = SpikeMixture(edge, args.clusters)
    n_nodes = model.lat_man.data_dim
    print(
        f"K->N->X: X(Normal-{N_OBS}) <- N(DiagBoltzmann-{n_nodes}) <- K(Cat-{args.clusters})"
    )
    print(
        f"conv {tuple(args.in_lat)} s{tuple(args.stride)} k{tuple(args.kernel)} "
        f"(non-overlap), {n_nodes} spike units"
    )

    params = train(
        model, train_data, args.steps, key, args.lr, args.min_var, args.max_var
    )

    # Figure: real digits + per-cluster mean image + per-cluster ancestral samples
    means = np.asarray(model.cluster_mean_images(params))
    _, _, logits = model.split(params)
    pk = np.asarray(jax.nn.softmax(logits))
    order = np.argsort(-pk)
    ncols = min(args.clusters, 12)
    fig, ax = plt.subplots(3, ncols, figsize=(1.3 * ncols, 4.2))
    for j in range(ncols):
        k = order[j]
        ax[0, j].imshow(
            np.array(test_data[j]).reshape(IMG, IMG), cmap="gray", vmin=0, vmax=1
        )
        ax[1, j].imshow(
            np.clip(means[k].reshape(IMG, IMG), 0, 1), cmap="gray", vmin=0, vmax=1
        )
        s = np.asarray(
            model.sample_cluster(jax.random.fold_in(key, k), params, int(k), 1)
        )[0]
        ax[2, j].imshow(np.clip(s.reshape(IMG, IMG), 0, 1), cmap="gray", vmin=0, vmax=1)
        ax[1, j].set_title(f"p={pk[k]:.2f}", fontsize=8)
        for i in range(3):
            ax[i, j].axis("off")
    ax[0, 0].set_title("real", loc="left")
    ax[1, 0].set_title("cluster mean", loc="left")
    ax[2, 0].set_title("sample", loc="left")
    fig.suptitle(f"K->N->X  {n_nodes} spikes, {args.clusters} clusters")
    fig.tight_layout()
    out = (
        example_paths(__file__).results_dir
        / f"spike_mixture_{n_nodes}n_{args.clusters}k.png"
    )
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
