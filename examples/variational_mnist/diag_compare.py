"""Compare multiple diag_stability runs side-by-side.

Loads several JSON output files and prints a compact summary table plus
optionally a plot of the matched per-step trajectories.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _summary(blob: dict[str, Any]) -> dict[str, Any]:
    history: list[dict[str, Any]] = blob["history"]
    nan_at = blob.get("nan_at", -1)
    last = history[-1] if history else {}
    bp_mins = [h["batch_prec_min"] for h in history]
    rho_norms = [h["rho_norm"] for h in history]
    return {
        "label": blob["config"].get("label", "?"),
        "obs": blob["config"]["observable"],
        "latent": blob["config"]["latent"],
        "lr": blob["config"]["learning_rate"],
        "conj_w": blob["config"]["conj_weight"],
        "clip": blob["config"].get("grad_clip_norm", 0),
        "ceil": blob["config"].get("theta2_ceiling", 0),
        "steps": blob["n_steps_run"],
        "nan_at": nan_at,
        "min_batch_prec": min(bp_mins) if bp_mins else float("nan"),
        "max_rho": max(rho_norms) if rho_norms else float("nan"),
        "final_elbo": last.get("elbo", float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--plot", default=None, help="If set, write a comparison plot here")
    args = parser.parse_args()

    blobs = []
    for i, p in enumerate(args.paths):
        with open(p) as f:
            b = json.load(f)
        label = args.labels[i] if args.labels and i < len(args.labels) else Path(p).stem
        b.setdefault("config", {})["label"] = label
        blobs.append(b)

    print(f"{'label':<24} {'obs':<8} {'latent':<18} {'lr':>7} {'conj':>5} "
          f"{'clip':>5} {'ceil':>5} {'steps':>6} {'nan_at':>7} "
          f"{'minBP':>9} {'maxRho':>8} {'lastELBO':>9}")
    print("-" * 130)
    summaries = [_summary(b) for b in blobs]
    for s in summaries:
        nan_str = "" if s["nan_at"] < 0 else str(s["nan_at"])
        print(
            f"{s['label']:<24} {s['obs']:<8} {s['latent']:<18} "
            f"{s['lr']:>7.0e} {s['conj_w']:>5.1f} "
            f"{s['clip']:>5.1f} {s['ceil']:>5.2f} "
            f"{s['steps']:>6d} {nan_str:>7} "
            f"{s['min_batch_prec']:>+9.2e} {s['max_rho']:>8.2f} "
            f"{s['final_elbo']:>9.1f}"
        )

    if args.plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for blob, s in zip(blobs, summaries):
            history = blob["history"]
            xs = np.array([h["step"] for h in history])
            axes[0, 0].plot(
                xs, [h["batch_prec_min"] for h in history], label=s["label"]
            )
            axes[0, 1].plot(xs, [h["rho_norm"] for h in history], label=s["label"])
            axes[1, 0].plot(xs, [h["elbo"] for h in history], label=s["label"])
            axes[1, 1].plot(xs, [h["conj_var"] for h in history], label=s["label"])
        axes[0, 0].axhline(0, color="k", lw=0.5, ls=":")
        axes[0, 0].set_yscale("symlog", linthresh=1e-2)
        axes[0, 0].set_ylabel("batch precision min")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_xlabel("step")
        axes[0, 1].set_ylabel("||rho||")
        axes[0, 1].set_xlabel("step")
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].set_ylabel("ELBO")
        axes[1, 0].set_xlabel("step")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_ylabel("conj Var[r]")
        axes[1, 1].set_xlabel("step")
        axes[1, 1].legend(fontsize=8)
        fig.savefig(args.plot, dpi=110)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
