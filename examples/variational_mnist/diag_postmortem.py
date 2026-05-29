"""Post-mortem analyzer for a diag_stability.py run that NaN'd.

Loads the saved JSON history + the last-finite params .npz, then re-creates the
model and:

1. Reports the trajectory leading up to NaN (last K probes).
2. For each pixel, finds the dimension where precision dropped most.
3. Stresses the last-finite params with a large batch of posterior samples to
   try to find the specific (x, z) pair that produces NaN.
4. Reports counts of NaN-producing posterior samples per x in the test data.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .model import create_model
from .train import load_mnist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="diag_stability JSON output")
    parser.add_argument(
        "--params-path",
        default=None,
        help="path to *_params.npz (defaults to JSON_PATH with _params.npz)",
    )
    parser.add_argument(
        "--n-posterior-samples",
        type=int,
        default=64,
        help="Z samples per data point to test for NaN log_density.",
    )
    parser.add_argument(
        "--n-test-x",
        type=int,
        default=512,
        help="Number of x's from train data to probe.",
    )
    args = parser.parse_args()
    jax.config.update("jax_platform_name", "gpu")

    with open(args.json_path) as f:
        blob = json.load(f)
    cfg = blob["config"]
    print(f"# Config: obs={cfg['observable']} latent={cfg['latent']} "
          f"n_lat={cfg['n_latent']} lr={cfg['learning_rate']} "
          f"conj_weight={cfg['conj_weight']}")
    print(f"# Steps run: {blob['n_steps_run']}; NaN at: {blob['nan_at']}")
    history = blob["history"]
    print(f"# Probes: {len(history)}")

    # Print last 8 probes for trajectory context.
    print()
    print("Last 8 probes:")
    print(
        f"{'step':>5} {'rho':>8} {'b_pmin':>9} {'b_pmax':>9} "
        f"{'p_pmin':>9} {'p_pmax':>9} {'xy':>6} {'xyk':>6} {'xk':>6}"
    )
    for h in history[-8:]:
        print(
            f"{h['step']:>5d} {h['rho_norm']:>8.3f} "
            f"{h['bias_prec_min']:>+9.2e} {h['bias_prec_max']:>+9.2e} "
            f"{h['batch_prec_min']:>+9.2e} {h['batch_prec_max']:>+9.2e} "
            f"{h['xy_abs_max']:>6.3f} {h['xyk_abs_max']:>6.3f} "
            f"{h['xk_abs_max']:>6.3f}"
        )

    # Recreate model and load params.
    params_path = args.params_path or args.json_path.replace(".json", "_params.npz")
    d = np.load(params_path)
    params = jnp.asarray(d["params"])

    model = create_model(
        n_latent=cfg["n_latent"],
        n_clusters=cfg["n_clusters"],
        observable_type=cfg["observable"],
        interaction="full",
        latent=cfg["latent"],
        latent_graph="chain",
    )
    print(f"\n# Model dim: {model.dim}; loaded params: {params.shape}")
    print(f"# Params: n_nan={int(jnp.sum(jnp.isnan(params)))}")

    # Bias precision diagonals.
    _, lkl_p, _ = model.split_coords(params)
    obs_b, _ = model.gen_hrm.lkl_fun_man.split_coords(lkl_p)
    _, bias_prec = model.obs_man.split_location_precision(obs_b)
    print(
        f"# bias precision: min={float(jnp.min(bias_prec)):.3e} "
        f"max={float(jnp.max(bias_prec)):.3e} median={float(jnp.median(bias_prec)):.3e}"
    )

    # Load training data and probe posterior samples for NaN.
    train_data, _, _, _ = load_mnist(observable_type=cfg["observable"])
    xs = train_data[: args.n_test_x]

    key = jax.random.PRNGKey(0xDEADBEEF)
    n_pairs = args.n_test_x * args.n_posterior_samples
    print(
        f"\n# Probing {args.n_test_x} x's, {args.n_posterior_samples} z's each "
        f"({n_pairs} (x,z) pairs)..."
    )

    @jax.jit
    def per_x_nan_count(
        k: Array, x: Array
    ) -> tuple[Array, Array, Array]:
        q_params = model.approximate_posterior_at(params, x)
        zs = model.pst_man.sample(k, q_params, args.n_posterior_samples)
        lkls = jax.vmap(lambda z: model.likelihood_at(params, z))(zs)
        precs = jax.vmap(
            lambda lkl: model.obs_man.split_location_precision(lkl)[1]
        )(lkls)
        ld = jax.vmap(lambda lkl: model.obs_man.log_density(lkl, x))(lkls)
        prec_min = jnp.min(precs)
        n_bad_prec = jnp.sum(precs <= 0)
        n_nan_ld = jnp.sum(jnp.isnan(ld))
        return prec_min, n_bad_prec.astype(jnp.float32), n_nan_ld.astype(jnp.float32)

    keys = jax.random.split(key, args.n_test_x)
    per_x = jax.vmap(per_x_nan_count)(keys, xs)
    prec_mins, n_bad_per_x, n_nan_per_x = per_x

    print(f"# overall min precision across all (x,z): {float(jnp.min(prec_mins)):.4e}")
    print(f"# total bad precision (<=0): {int(jnp.sum(n_bad_per_x))}")
    print(f"# total NaN log_density: {int(jnp.sum(n_nan_per_x))}")
    print(f"# x's with ≥1 bad precision: {int(jnp.sum(n_bad_per_x > 0))}/{args.n_test_x}")
    print(f"# x's with ≥1 NaN log_density: {int(jnp.sum(n_nan_per_x > 0))}/{args.n_test_x}")

    # Index of the worst x
    worst_x_idx = int(jnp.argmin(prec_mins))
    print(
        f"\n# Worst x: index {worst_x_idx} prec_min={float(prec_mins[worst_x_idx]):.4e} "
        f"n_bad_prec={int(n_bad_per_x[worst_x_idx])} "
        f"n_nan_ld={int(n_nan_per_x[worst_x_idx])}"
    )


if __name__ == "__main__":
    main()
