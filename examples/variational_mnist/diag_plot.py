"""Plot the diag_stability.py trajectory.

Produces a single multi-panel figure with the per-step probe data, sized for
quick visual inspection of where the precision crosses zero (if ever).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _series(history: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([h["step"] for h in history if key in h])
    ys = np.array([h[key] for h in history if key in h])
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.path) as f:
        blob = json.load(f)
    history = blob["history"]
    cfg = blob["config"]
    title = (
        f"{cfg['observable']} | {cfg['latent']} | n_lat={cfg['n_latent']} "
        f"K={cfg['n_clusters']} | lr={cfg['learning_rate']} conj={cfg['conj_weight']} "
        f"clip={cfg.get('grad_clip_norm', 0)} ceil={cfg.get('theta2_ceiling', 0)} "
        f"| nan_at={blob['nan_at']}"
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    xs, ys = _series(history, "elbo")
    ax.plot(xs, ys, color="C0")
    ax.set_ylabel("ELBO")
    ax.set_xlabel("step")
    ax.set_title("ELBO")

    ax = axes[0, 1]
    xs, ys = _series(history, "rho_norm")
    ax.plot(xs, ys, color="C2")
    ax.set_ylabel("||rho||")
    ax.set_xlabel("step")
    ax.set_title("||rho||")

    ax = axes[1, 0]
    xs, bp_min = _series(history, "bias_prec_min")
    _, bp_max = _series(history, "bias_prec_max")
    ax.plot(xs, bp_min, label="bias prec min", color="C3")
    ax.plot(xs, bp_max, label="bias prec max", color="C4", ls="--")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylabel("bias precision")
    ax.set_xlabel("step")
    ax.set_title("bias precision diag (min/max)")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1, 1]
    xs, batch_min = _series(history, "batch_prec_min")
    _, batch_q01 = _series(history, "batch_prec_q01")
    _, batch_q99 = _series(history, "batch_prec_q99")
    _, batch_max = _series(history, "batch_prec_max")
    ax.plot(xs, batch_min, label="min", color="C3")
    ax.plot(xs, batch_q01, label="q01", color="C1", ls=":")
    ax.plot(xs, batch_q99, label="q99", color="C4", ls=":")
    ax.plot(xs, batch_max, label="max", color="C4", ls="--")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.set_ylabel("batch precision (per-sample)")
    ax.set_xlabel("step")
    ax.set_title("batch p(x|z) precision (z~prior)")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[2, 0]
    for key, color in [
        ("xy_abs_max", "C0"),
        ("xyk_abs_max", "C1"),
        ("xk_abs_max", "C2"),
        ("obs_bias_abs_max", "C3"),
    ]:
        xs, ys = _series(history, key)
        ax.plot(xs, ys, label=key, color=color)
    ax.set_yscale("log")
    ax.set_ylabel("abs max")
    ax.set_xlabel("step")
    ax.set_title("interaction & bias L_inf")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    xs, grad = _series(history, "grad_norm")
    _, conj_var = _series(history, "conj_var")
    ax.plot(xs, grad, label="grad_norm", color="C5")
    ax2 = ax.twinx()
    ax2.plot(xs, conj_var, label="conj_var", color="C6", ls="--")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_title("grad_norm & conj_var")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(title)

    out = args.out or str(Path(args.path).with_suffix(".png"))
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
