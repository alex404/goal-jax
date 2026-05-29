"""Analyze diag_stability.py output.

Loads the JSON history and prints a summary table of the final pre-NaN steps,
plus the index at which each precision-bound first crossed a critical threshold.
"""

import argparse
import json
import math
from typing import Any


def _first_bad(history: list[dict[str, Any]], key: str, predicate) -> int | None:
    for snap in history:
        v = snap.get(key)
        if v is None or not math.isfinite(v):
            return snap["step"]
        if predicate(v):
            return snap["step"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--tail", type=int, default=20)
    args = parser.parse_args()

    with open(args.path) as f:
        blob = json.load(f)

    history: list[dict[str, Any]] = blob["history"]
    nan_at = blob.get("nan_at", -1)
    print(f"# config: {blob['config']}")
    print(f"# n_steps_run={blob['n_steps_run']} nan_at={nan_at} elapsed={blob['elapsed_seconds']:.1f}s")
    print(f"# total probes: {len(history)}")

    # First crossings
    first_bias_neg = _first_bad(history, "bias_prec_min", lambda v: v <= 0.0)
    first_batch_neg = _first_bad(history, "batch_prec_min", lambda v: v <= 0.0)
    first_batch_q01 = _first_bad(history, "batch_prec_q01", lambda v: v <= 0.0)
    first_rho_5 = _first_bad(history, "rho_norm", lambda v: v >= 5.0)
    first_rho_10 = _first_bad(history, "rho_norm", lambda v: v >= 10.0)
    print("# first crossings:")
    print(f"#   bias_prec <= 0: {first_bias_neg}")
    print(f"#   batch_prec_min <= 0: {first_batch_neg}")
    print(f"#   batch_prec_q01 <= 0: {first_batch_q01}")
    print(f"#   rho_norm >= 5: {first_rho_5}")
    print(f"#   rho_norm >= 10: {first_rho_10}")

    print()
    print(
        "step elbo conj_var rho_norm bias_min bias_max batch_min batch_q01 batch_q99 batch_max xy xyk xk grad_norm"
    )
    print("-" * 130)
    tail = history[-args.tail :] if args.tail > 0 else history
    for snap in tail:
        print(
            f"{snap['step']:>5d} {snap['elbo']:>9.1f} {snap['conj_var']:>9.2e} "
            f"{snap['rho_norm']:>8.2f} "
            f"{snap['bias_prec_min']:>+9.2e} {snap['bias_prec_max']:>+9.2e} "
            f"{snap['batch_prec_min']:>+9.2e} {snap['batch_prec_q01']:>+9.2e} "
            f"{snap['batch_prec_q99']:>+9.2e} {snap['batch_prec_max']:>+9.2e} "
            f"{snap['xy_abs_max']:>6.2f} {snap['xyk_abs_max']:>6.2f} {snap['xk_abs_max']:>6.2f} "
            f"{snap['grad_norm']:>9.2e}"
        )


if __name__ == "__main__":
    main()
