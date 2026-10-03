"""Post-hoc analysis of a trained Z -> N -> X model: does the top latent matter?

Loads params saved by ``mnist_hierarchical`` (the ``.npz`` beside the panel) and
produces, under results/variational_mnist/analysis/:
  <tag>_traverse.png : vary each of the first Z dims, decode E[x | E[N|z]]
and prints:
  ||Theta_XY||, ||Theta_ZN||          -- edge liveness
  generation diversity                -- mean per-pixel std across sampled z
  Z-sensitivity                        -- mean |dx| as z sweeps (0 == z is inert)

Usage:
  JAX_PLATFORMS=cpu PYTHONPATH=. uv run python -m examples.variational_mnist.analyze_znx \
    IN_H IN_W ST_H ST_W K_H K_W CHANNELS TOP_DIM NPZ_PATH
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt  # noqa: E402
from examples.variational_mnist import mnist_hierarchical as M  # noqa: E402

IMG = 28
OUT = Path("/home/alex404/code/goal-jax/results/variational_mnist/analysis")


def main() -> None:
    a = sys.argv[1:]
    ih, iw, sh, sw, kh, kw, ch, td = (int(a[i]) for i in range(8))
    npz = a[8]
    prior = a[9] if len(a) > 9 else "diagonal"
    tag = Path(npz).stem
    OUT.mkdir(parents=True, exist_ok=True)

    model = M.build_model(
        "conv",
        0,
        td,
        conv_in=(ih, iw),
        conv_stride=(sh, sw),
        conv_kernel=(kh, kw),
        conv_channels=ch,
        conv_prior=prior,
    )
    params = jnp.asarray(np.load(npz)["params"])

    _, theta_xy = model.split_lower(params)
    _, top_lkl, _ = model.split_top(params)
    _, theta_zn = model.top_var.gen_hrm.lkl_fun_man.split_coords(top_lkl)
    _, lower_lkl, _ = model.split_coords(params)
    theta_z, _, _ = model.split_top(params)
    print(
        f"||Theta_XY|| = {float(jnp.linalg.norm(theta_xy)):.3f}   "
        f"||Theta_ZN|| = {float(jnp.linalg.norm(theta_zn)):.3f}"
    )

    def decode_mean_from_z(z: jax.Array) -> jax.Array:
        s_z = model.top_man.sufficient_statistic(z)
        n_nat = model.top_var.gen_hrm.lkl_fun_man(top_lkl, s_z)
        n_mean = model.mid_man.to_mean(n_nat)  # E[N | z] = spike probabilities
        x_nat = model.lower_hrm.lkl_fun_man(lower_lkl, n_mean)
        return model.obs_man.to_mean(x_nat)[: IMG * IMG]

    # generation diversity: per-pixel std over decoded means for 64 prior draws
    zs = model.top_man.sample(jax.random.PRNGKey(1), theta_z, 64)
    gens = jax.vmap(decode_mean_from_z)(zs)
    diversity = float(jnp.mean(jnp.std(gens, axis=0)))

    # Z-sensitivity: how much the output moves as each of first dims sweeps [-2,2]
    z0 = model.top_man.to_mean(theta_z)[: model.top_man.data_dim]
    ndim = min(6, model.top_man.data_dim)
    grid = np.linspace(-2.5, 2.5, 9)

    def sweep_dim(r: int) -> jax.Array:
        zr = jax.vmap(lambda g: jnp.asarray(z0).at[r].set(g))(jnp.asarray(grid))
        return jax.vmap(decode_mean_from_z)(zr)

    sweeps = [np.asarray(sweep_dim(r)) for r in range(ndim)]
    sens = float(np.mean([np.mean(np.abs(s - s.mean(0))) for s in sweeps]))
    print(f"generation diversity (per-pixel std) = {diversity:.4f}")
    print(f"Z-sensitivity (mean |dx| over sweep)  = {sens:.4f}   (0 == z inert)")

    fig, ax = plt.subplots(ndim, 9, figsize=(1.1 * 9, 1.1 * ndim))
    for r in range(ndim):
        for c in range(9):
            ax[r][c].imshow(
                np.clip(sweeps[r][c].reshape(IMG, IMG), 0, 1),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            ax[r][c].axis("off")
    fig.suptitle(f"{tag}: Z traversals  (div={diversity:.3f} sens={sens:.3f})")
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_traverse.png", dpi=130)
    print(f"saved {OUT / f'{tag}_traverse.png'}")


if __name__ == "__main__":
    main()
