"""MNIST proof-of-principle: diagonal vs chain vs chordal Boltzmann middle layer.

Trains the 3-level hierarchy ``X(Normal-784) <- Y(Boltzmann) <- Z(Gaussian-16)``
for each middle-layer connectivity and produces one figure:

    row 0        : original test digits
    rows 1..3    : reconstructions (diagonal / chain / chordal)
    rows 4..6    : generative samples (diagonal / chain / chordal)

plus a printed ELBO / reconstruction-MSE / conjugation-residual table.

Run (GPU by default)::

    uv run python -m examples.variational_mnist.mnist_compare --steps 4000
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from ..shared import example_paths  # noqa: E402
from . import mnist_hierarchical as MH  # noqa: E402,N812

IMG = MH.IMG
MIDDLES = ["diagonal", "chain", "chordal"]
COLORS = {"diagonal": "#888888", "chain": "#1f77b4", "chordal": "#d62728"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-mid", type=int, default=MH.N_MID)
    ap.add_argument("--top-dim", type=int, default=MH.TOP_DIM)
    ap.add_argument("--n-train", type=int, default=MH.N_TRAIN)
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)
    _, k_rest = jax.random.split(key)
    train_data, test_data = MH.load_mnist(args.n_train, MH.N_TEST)
    show = test_data[:8]
    print(f"MNIST train {train_data.shape}, test {test_data.shape}\n")

    results = []
    recon_rows, gen_rows = {}, {}
    for kind in MIDDLES:
        print(f"=== middle = {kind} ===")
        try:
            k_tr, k_rec, k_gen, k_res = jax.random.split(
                jax.random.fold_in(k_rest, hash(kind) % 1000), 4
            )
            model = MH.build_model(kind, args.n_mid, args.top_dim)
            params = MH.train(model, train_data, test_data, args.steps, k_tr)

            recons = MH.reconstruct(model, params, show, k_rec)
            mse = float(jnp.mean((show - recons) ** 2))
            gens = MH.generative_means(model, params, k_gen, 8)
            elbo = float(model.mean_elbo(k_res, params, test_data[:512], 16))
            gvar = float(model.prior_conjugation_loss(k_res, params, 256))
            recon_rows[kind] = np.array(recons)
            gen_rows[kind] = np.clip(np.array(gens), 0, 1)
            results.append({"kind": kind, "elbo": elbo, "mse": mse, "gvar": gvar})
            print(
                f"  -> test ELBO {elbo:.2f}  recon MSE {mse:.4f}  Var[r] {gvar:.4f}\n"
            )
        except Exception as e:
            print(f"  !! {kind} failed: {type(e).__name__}: {e}\n")
            recon_rows[kind] = np.zeros((8, MH.N_OBS))
            gen_rows[kind] = np.zeros((8, MH.N_OBS))
            results.append(
                {
                    "kind": kind,
                    "elbo": float("nan"),
                    "mse": float("nan"),
                    "gvar": float("nan"),
                }
            )

    print("=" * 60)
    print(f"{'middle':10s} {'test ELBO':>10s} {'recon MSE':>10s} {'Var[r]':>9s}")
    for r in results:
        print(f"{r['kind']:10s} {r['elbo']:10.2f} {r['mse']:10.4f} {r['gvar']:9.4f}")
    print("=" * 60)

    # Figure
    rows = ["data"] + [f"recon:{m}" for m in MIDDLES] + [f"gen:{m}" for m in MIDDLES]
    fig, axes = plt.subplots(len(rows), 8, figsize=(11, 1.3 * len(rows)))
    for j in range(8):
        axes[0, j].imshow(
            np.array(show[j]).reshape(IMG, IMG), cmap="gray", vmin=0, vmax=1
        )
        for i, m in enumerate(MIDDLES):
            axes[1 + i, j].imshow(
                recon_rows[m][j].reshape(IMG, IMG), cmap="gray", vmin=0, vmax=1
            )
            axes[4 + i, j].imshow(
                gen_rows[m][j].reshape(IMG, IMG), cmap="gray", vmin=0, vmax=1
            )
    for i, label in enumerate(rows):
        axes[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        for j in range(8):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle(
        "Hierarchical variational conjugation on MNIST: middle-layer connectivity"
    )
    fig.tight_layout()
    results_dir = example_paths(__file__).results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "mnist_compare.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
