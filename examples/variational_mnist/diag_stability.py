"""Stability diagnostic for the variational MNIST chordal-normal NaN.

Runs a normal-observable training loop and tracks per-step parameter ranges,
focusing on the precision (theta_2) component of the Gaussian observable. The
hypothesis under test is that the affine likelihood ``b + L(s(z))`` pushes
``theta_2`` toward zero (i.e. the precision toward singular) until
``log_partition_function`` produces NaN.

Outputs a per-step CSV-like log to stdout suitable for tee-ing to a file, plus
a JSON snapshot of the final 50 steps right before any NaN.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

import argparse
import json
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from goal.geometry.exponential_family.variational import conjugation_metrics

from .model import MixtureModel, create_model
from .train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_MC_SAMPLES,
    N_CONJ_PENALTY_SAMPLES,
    N_TRAIN,
    compute_batch_entropy,
    load_mnist,
)


def _theta2_diag(obs_man: Any, lkl_nat: Array) -> Array:
    """Extract the precision diagonal from a diagonal-normal natural-param vector.

    Returns the *precision diagonal* (must be positive). For non-Gaussian
    observables (Binomial, Poisson) there is no precision parameter; we return
    a vector of ones so the downstream stats stay finite and positive (precision
    failure mode is not applicable).
    """
    if not hasattr(obs_man, "split_location_precision"):
        return jnp.ones((obs_man.data_dim,))
    _, precision = obs_man.split_location_precision(lkl_nat)
    return precision


def _bias_and_int_blocks(model: MixtureModel, params: Array) -> dict[str, Array]:
    """Pull out the obs bias and the three interaction blocks for diagnostics."""
    _, lkl_params, _ = model.split_coords(params)
    obs_bias, int_params = model.gen_hrm.lkl_fun_man.split_coords(lkl_params)
    xy, xyk, xk = model.gen_hrm.int_man.coord_blocks(int_params)
    return {"obs_bias": obs_bias, "xy": xy, "xyk": xyk, "xk": xk}


def _batch_lkl_natural(
    model: MixtureModel, params: Array, key: Array, n_samples: int
) -> Array:
    """Sample z from the latent prior and return p(x|z) naturals for each."""
    prior_mixture_params = model.prior_params(params)
    z_samples = model.mix_man.sample(key, prior_mixture_params, n_samples)
    return jax.vmap(lambda z: model.likelihood_at(params, z))(z_samples)


def _posterior_lkl_natural(
    model: MixtureModel,
    params: Array,
    key: Array,
    data: Array,
    n_samples_per: int,
) -> Array:
    """Sample z from posterior q(z|x) for many x, return p(x|z) naturals.

    Uses ``approximate_posterior_at`` for each ``x`` in ``data``, draws
    ``n_samples_per`` z's from each posterior, and returns the resulting
    likelihood naturals stacked. The MC sampler that the ELBO uses is exactly
    this same path, so any precision-crossing here is exactly what would NaN
    the ELBO.
    """
    n_x = data.shape[0]
    keys = jax.random.split(key, n_x)

    def per_x(k: Array, x: Array) -> Array:
        q_params = model.approximate_posterior_at(params, x)
        zs = model.pst_man.sample(k, q_params, n_samples_per)
        # likelihood_at expects raw z; pst_man samples return raw z's.
        return jax.vmap(lambda z: model.likelihood_at(params, z))(zs)

    out = jax.vmap(per_x)(keys, data)
    return out.reshape((-1, *out.shape[2:]))


def _diag_snapshot(
    model: MixtureModel,
    params: Array,
    key: Array,
    n_samples: int,
    post_data: Array | None = None,
    n_post_x: int = 64,
    n_post_per: int = 4,
) -> dict[str, float]:
    """Compute a snapshot of stability-relevant statistics at this step."""
    blocks = _bias_and_int_blocks(model, params)
    obs_bias = blocks["obs_bias"]
    xy = blocks["xy"]
    xyk = blocks["xyk"]
    xk = blocks["xk"]

    # obs-bias precision diagonal (must be > 0)
    bias_prec = _theta2_diag(model.obs_man, obs_bias)

    # batch of z samples from the prior: precision diagonal must be > 0 for every sample
    lkl_batch = _batch_lkl_natural(model, params, key, n_samples)
    batch_prec = jax.vmap(lambda lkl: _theta2_diag(model.obs_man, lkl))(lkl_batch)
    # shape [n_samples, data_dim]
    batch_prec_min = float(jnp.min(batch_prec))
    batch_prec_max = float(jnp.max(batch_prec))
    batch_prec_q01 = float(jnp.percentile(batch_prec, 1.0))
    batch_prec_q99 = float(jnp.percentile(batch_prec, 99.0))

    # rho parameters: use conjugation_parameters() to match train.py — it returns
    # the FULL rho (stored rho_y + computed rho_z from cluster offsets), not the
    # stored slot alone.
    lat_params, _, _ = model.split_coords(params)
    rho_full = model.conjugation_parameters(params)
    rho_norm = float(jnp.linalg.norm(rho_full))

    # Latent prior parameters: mix manifold has (mix_obs_bias, mix_int, cat_params).
    # For chordal/bernoullis, mix_obs_bias is the base-latent natural params (biases + couplings).
    try:
        mix_obs_bias, mix_int, cat_params = model.mix_man.split_coords(lat_params)
        prr_abs_max = float(jnp.max(jnp.abs(mix_obs_bias)))
        cat_abs_max = float(jnp.max(jnp.abs(cat_params)))
        mix_int_abs_max = float(jnp.max(jnp.abs(mix_int)))
    except Exception:
        prr_abs_max = float("nan")
        cat_abs_max = float("nan")
        mix_int_abs_max = float("nan")

    # Direct probe of psi values across batch samples — if any explode we want
    # to know about it. NaN-tolerant: replace NaN with +inf before max so a
    # single bad sample is visible.
    psi_vals = jax.vmap(lambda lkl: model.obs_man.log_partition_function(lkl))(lkl_batch)
    psi_max = float(jnp.max(jnp.where(jnp.isnan(psi_vals), jnp.inf, psi_vals)))
    psi_min = float(jnp.min(jnp.where(jnp.isnan(psi_vals), -jnp.inf, psi_vals)))
    psi_any_nan = bool(jnp.any(jnp.isnan(psi_vals)))

    # Posterior-sampled probe: this matches the ELBO's z-sampling path. A NaN
    # here means the next loss/grad step will NaN.
    if post_data is not None:
        post_key, _ = jax.random.split(key)
        x_subset = post_data[:n_post_x]
        lkl_post = _posterior_lkl_natural(
            model, params, post_key, x_subset, n_post_per
        )
        post_prec = jax.vmap(
            lambda lkl: _theta2_diag(model.obs_man, lkl)
        )(lkl_post)
        psi_post = jax.vmap(
            lambda lkl: model.obs_man.log_partition_function(lkl)
        )(lkl_post)
        post_prec_min = float(
            jnp.min(jnp.where(jnp.isnan(post_prec), -jnp.inf, post_prec))
        )
        post_prec_max = float(
            jnp.max(jnp.where(jnp.isnan(post_prec), jnp.inf, post_prec))
        )
        psi_post_min = float(
            jnp.min(jnp.where(jnp.isnan(psi_post), -jnp.inf, psi_post))
        )
        psi_post_max = float(
            jnp.max(jnp.where(jnp.isnan(psi_post), jnp.inf, psi_post))
        )
        post_any_nan = bool(
            jnp.any(jnp.isnan(post_prec)) | jnp.any(jnp.isnan(psi_post))
        )
    else:
        post_prec_min = float("nan")
        post_prec_max = float("nan")
        psi_post_min = float("nan")
        psi_post_max = float("nan")
        post_any_nan = False

    return {
        "bias_prec_min": float(jnp.min(bias_prec)),
        "bias_prec_max": float(jnp.max(bias_prec)),
        "bias_prec_mean": float(jnp.mean(bias_prec)),
        "batch_prec_min": batch_prec_min,
        "batch_prec_q01": batch_prec_q01,
        "batch_prec_q99": batch_prec_q99,
        "batch_prec_max": batch_prec_max,
        "post_prec_min": post_prec_min,
        "post_prec_max": post_prec_max,
        "psi_post_min": psi_post_min,
        "psi_post_max": psi_post_max,
        "post_any_nan": post_any_nan,
        "xy_abs_max": float(jnp.max(jnp.abs(xy))),
        "xyk_abs_max": float(jnp.max(jnp.abs(xyk))),
        "xk_abs_max": float(jnp.max(jnp.abs(xk))),
        "obs_bias_abs_max": float(jnp.max(jnp.abs(obs_bias))),
        "prr_abs_max": prr_abs_max,
        "cat_abs_max": cat_abs_max,
        "mix_int_abs_max": mix_int_abs_max,
        "psi_max": psi_max,
        "psi_min": psi_min,
        "psi_any_nan": psi_any_nan,
        "rho_norm": rho_norm,
    }


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=8000)
    parser.add_argument("--n-latent", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--conj-weight", type=float, default=1.0)
    parser.add_argument("--conj-warmup-steps", type=int, default=2000)
    parser.add_argument("--kl-warmup-steps", type=int, default=1000)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument(
        "--latent", choices=["bernoullis", "chordal_boltzmann"], default="chordal_boltzmann"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-every", type=int, default=25)
    parser.add_argument("--n-probe-samples", type=int, default=512)
    parser.add_argument("--out", default="/tmp/varmnist_stability/diag.json")
    parser.add_argument(
        "--entropy-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--posterior-entropy-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU (default).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (overrides --gpu).",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="If > 0, clip global gradient norm to this value via optax.clip_by_global_norm.",
    )
    parser.add_argument(
        "--theta2-ceiling",
        type=float,
        default=0.0,
        help=(
            "If > 0, after each update project the observable bias's theta_2 "
            "(precision component) to stay <= -ceiling, ensuring precision >= "
            "2*ceiling. Set 0 to disable. Only the bias is projected — the "
            "interaction can still drive sample-level precision below 0."
        ),
    )
    parser.add_argument(
        "--observable",
        choices=["normal", "binomial", "poisson"],
        default="normal",
        help="Which observable to run with.",
    )
    parser.add_argument(
        "--posterior-probe",
        action="store_true",
        help=(
            "Also probe likelihood naturals at posterior-sampled z's. Slower to "
            "compile (chordal_boltzmann posterior sampler is a JT scan), but "
            "exactly matches the ELBO's z-sampling path."
        ),
    )
    parser.add_argument(
        "--clamp-precision",
        type=float,
        default=0.0,
        help=(
            "If > 0, monkey-patch ``Normal.log_partition_function`` to clamp "
            "the precision diagonal at this floor before ``inverse`` / "
            "``logdet``. Diagnostic-only — patches the class, so anything else "
            "that uses Normal is also affected. The bound is the smallest "
            "precision the model can express; any 'tighter' direction is "
            "absorbed by the clamp."
        ),
    )
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")
    else:
        jax.config.update("jax_platform_name", "gpu")

    # Optional precision clamp on Normal.log_partition_function. Patched at the
    # class level so it propagates to model.obs_man (and any other Normal that
    # gets created). This is invasive — we use it only inside the diagnostic.
    if args.clamp_precision > 0 and args.observable == "normal":
        from goal.models.base.gaussian.normal import Normal

        floor = float(args.clamp_precision)
        _orig_log_partition = Normal.log_partition_function

        def _safe_log_partition(self, params: Array) -> Array:
            loc, precision = self.split_location_precision(params)
            precision = jnp.maximum(precision, floor)
            new_params = self.join_location_precision(loc, precision)
            return _orig_log_partition(self, new_params)

        Normal.log_partition_function = _safe_log_partition  # type: ignore[method-assign]
        print(f"# patched Normal.log_partition_function with clamp >= {floor:g}")

    print(f"# diag_stability config: {vars(args)}", flush=True)

    # Match train.py's RNG flow exactly: split outer key into (outer, train_key)
    # then pass train_key down as the working key. Without this the init key
    # diverges from the production runs and reproductions of NaN trajectories
    # become impossible.
    outer_key = jax.random.PRNGKey(args.seed)
    _, key = jax.random.split(outer_key)

    train_data, _, _, _ = load_mnist(
        observable_type=args.observable
    )

    model = create_model(
        n_latent=args.n_latent,
        n_clusters=args.n_clusters,
        observable_type=args.observable,
        interaction="full",
        latent=args.latent,  # type: ignore[arg-type]
        latent_graph="chain",
    )
    print(
        f"# model dim={model.dim} obs_dim={model.obs_man.dim} obs_data_dim={model.obs_man.data_dim}",
        flush=True,
    )

    key, init_key = jax.random.split(key)
    params = model.initialize_from_sample(init_key, train_data, location=0.0, shape=0.1)

    key, cluster_key = jax.random.split(key)
    lat_params, lkl_params, rho = model.split_coords(params)
    obs_bias, int_params = model.gen_hrm.lkl_fun_man.split_coords(lkl_params)
    mix_obs_bias, _, cat_params = model.mix_man.split_coords(lat_params)

    mix_int_dim = model.bas_lat_man.dim * (model.n_categories - 1)
    k1, k2, k3 = jax.random.split(cluster_key, 3)
    new_mix_int_mat = jax.random.normal(k1, (mix_int_dim,)) * 0.5
    new_lat_params = model.mix_man.join_coords(mix_obs_bias, new_mix_int_mat, cat_params)

    block_int = model.gen_hrm.int_man
    xy_p, xyk_p, xk_p = block_int.coord_blocks(int_params)
    obs_dim = model.obs_man.dim
    lat_dim = model.bas_lat_man.dim
    xyk_scale = 0.1 / math.sqrt(obs_dim * lat_dim)
    xk_scale = 0.1 / math.sqrt(obs_dim)
    new_xyk = jax.random.normal(k2, xyk_p.shape) * xyk_scale
    new_xk = jax.random.normal(k3, xk_p.shape) * xk_scale
    int_params = jnp.concatenate([xy_p, new_xyk, new_xk])
    new_lkl_params = model.gen_hrm.lkl_fun_man.join_coords(obs_bias, int_params)
    params = model.join_coords(new_lat_params, new_lkl_params, rho)

    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0,
                end_value=args.learning_rate,
                transition_steps=args.lr_warmup_steps,
            ),
            optax.constant_schedule(args.learning_rate),
        ],
        boundaries=[args.lr_warmup_steps],
    )
    if args.grad_clip_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip_norm),
            optax.adam(lr_schedule),
        )
    else:
        optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    def loss_fn(
        params: Array, key: Array, batch: Array, beta: Array, conj_beta: Array
    ) -> tuple[Array, tuple[Array, Array]]:
        # Match train.py's loss_fn_gradient RNG flow: it splits the incoming
        # key into (key, elbo_key, conj_key) and discards the first piece.
        # Mirroring that here keeps the gradient trajectory comparable to
        # production runs.
        _, elbo_key, conj_key = jax.random.split(key, 3)
        elbo_1 = model.mean_elbo(elbo_key, params, batch, DEFAULT_N_MC_SAMPLES)
        mean_kl = jnp.mean(
            jax.vmap(lambda x: model.elbo_divergence(params, x))(batch)
        )
        elbo = elbo_1 + (1.0 - beta) * mean_kl

        prior_entropy = model.prior_entropy(params)

        def cprobs(x: Array) -> Array:
            q = model.approximate_posterior_at(params, x)
            return model.get_cluster_probs(q)

        all_probs = jax.vmap(cprobs)(batch)
        post_entropy = compute_batch_entropy(all_probs)

        conj_var, _, _ = conjugation_metrics(
            model, conj_key, params, N_CONJ_PENALTY_SAMPLES
        )
        conj_loss = conj_beta * args.conj_weight * conj_var
        loss = (
            -elbo
            - args.entropy_weight * prior_entropy
            - args.posterior_entropy_weight * post_entropy
            + conj_loss
        )
        return loss, (elbo, conj_var)

    project_bias = args.theta2_ceiling > 0 and args.observable == "normal"

    def _project_params(params: Array) -> Array:
        if not project_bias:
            return params
        lat_p, lkl_p, rho_p = model.split_coords(params)
        obs_b, int_p = model.gen_hrm.lkl_fun_man.split_coords(lkl_p)
        loc, theta2 = model.obs_man.split_coords(obs_b)
        theta2 = jnp.minimum(theta2, -args.theta2_ceiling)
        new_bias = model.obs_man.join_coords(loc, theta2)
        new_lkl = model.gen_hrm.lkl_fun_man.join_coords(new_bias, int_p)
        return model.join_coords(lat_p, new_lkl, rho_p)

    # Mirror train.py: the jitted step takes (params, opt_state, step_key, data)
    # and internally derives next_key + batch_key. This preserves the RNG
    # trajectory between the production train.py and this diagnostic.
    @jax.jit
    def train_step(
        params: Array,
        opt_state: Any,
        step_key: Array,
        data: Array,
        beta: Array,
        conj_beta: Array,
    ):
        step_key, next_key, batch_key = jax.random.split(step_key, 3)
        batch_idx = jax.random.choice(batch_key, N_TRAIN, shape=(DEFAULT_BATCH_SIZE,))
        batch = data[batch_idx]
        (_, (elbo, conj_var)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, next_key, batch, beta, conj_beta
        )
        grad_norm = jnp.linalg.norm(grads)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        update_norm = jnp.linalg.norm(updates)
        new_params = optax.apply_updates(params, updates)
        new_params = _project_params(new_params)
        return (
            new_params,
            new_opt_state,
            step_key,
            elbo,
            conj_var,
            grad_norm,
            update_norm,
        )

    history: list[dict[str, Any]] = []
    last_params = params
    nan_at = -1
    step = 0
    key, train_key = jax.random.split(key)

    print(
        "# step,elbo,conj_var,rho_norm,bias_prec_min,bias_prec_max,batch_prec_min,batch_prec_max,post_prec_min,post_prec_max,xy_abs_max,xyk_abs_max,xk_abs_max,prr_abs_max,psi_post_max,nan?,grad_norm,update_norm",
        flush=True,
    )

    t_start = time.perf_counter()
    for step in range(args.n_steps):
        beta = jnp.array(min(1.0, step / max(args.kl_warmup_steps, 1)))
        conj_beta = jnp.array(min(1.0, step / max(args.conj_warmup_steps, 1)))
        # Probe key derived deterministically from step number so it does NOT
        # consume from train_key; train_key flows through train_step exactly as
        # in train.py.
        probe_key = jax.random.fold_in(jax.random.PRNGKey(0xDEADBEEF), step)

        (
            params,
            opt_state,
            train_key,
            elbo,
            conj_var,
            grad_norm,
            update_norm,
        ) = train_step(
            params, opt_state, train_key, train_data, beta, conj_beta
        )
        jax.block_until_ready(elbo)
        elbo_f = float(elbo)
        if not math.isfinite(elbo_f):
            nan_at = step
            print(f"# NaN detected at step {step}", flush=True)
            break

        last_params = params

        if step % args.probe_every == 0:
            snap = _diag_snapshot(
                model,
                params,
                probe_key,
                args.n_probe_samples,
                post_data=train_data if args.posterior_probe else None,
            )
            snap["step"] = step
            snap["elbo"] = elbo_f
            snap["conj_var"] = float(conj_var)
            snap["grad_norm"] = float(grad_norm)
            snap["update_norm"] = float(update_norm)
            snap["beta"] = float(beta)
            snap["conj_beta"] = float(conj_beta)
            history.append(snap)
            nan_flag = (
                "1"
                if (snap["psi_any_nan"] or snap.get("post_any_nan", False))
                else "0"
            )
            print(
                f"{step},{elbo_f:.3f},{snap['conj_var']:.3e},{snap['rho_norm']:.5f},"
                f"{snap['bias_prec_min']:.3e},{snap['bias_prec_max']:.3e},"
                f"{snap['batch_prec_min']:.3e},{snap['batch_prec_max']:.3e},"
                f"{snap['post_prec_min']:.3e},{snap['post_prec_max']:.3e},"
                f"{snap['xy_abs_max']:.3f},{snap['xyk_abs_max']:.3f},"
                f"{snap['xk_abs_max']:.3f},{snap['prr_abs_max']:.3f},"
                f"{snap['psi_post_max']:.3e},{nan_flag},"
                f"{snap['grad_norm']:.3e},{snap['update_norm']:.3e}",
                flush=True,
            )

    elapsed = time.perf_counter() - t_start

    out = {
        "config": vars(args),
        "n_steps_run": step + 1,
        "nan_at": nan_at,
        "elapsed_seconds": elapsed,
        "history": history,
    }
    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"# wrote {args.out} (history={len(history)} snapshots, elapsed={elapsed:.1f}s)")
    # Also save last params for offline inspection
    npz_path = args.out.replace(".json", "_params.npz")
    np.savez(npz_path, params=np.array(last_params))
    print(f"# wrote {npz_path}")


if __name__ == "__main__":
    main()
