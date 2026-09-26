"""Posterior-fidelity validation of conjugation on the toy hierarchy.

R^2 of the conjugation residual only checks whether the conjugation *equation*
holds pointwise. What conjugation actually buys is that the conjugate-form
recognition ``q(.|x)`` equals the true posterior ``p(.|x)`` of the generative
model. This script certifies that directly, using ``q`` as an importance
proposal (so the gold standard is independent of the residual):

1. **Inference gap (IWAE ladder).** ``IWAE(K)`` climbs from the ELBO toward
   ``log p(x)``; the plateau minus the ELBO is ``KL(q || p(z|x))`` in nats.
2. **Effective sample size ``ESS/K``.** With weights ``p(x,y,z)/q(y,z|x)``,
   ``ESS/K -> 1`` iff ``q = p(.|x)``. The single most direct conjugation number.
3. **Posterior-moment match + Gaussianity.** Self-normalized importance weights
   give a gold-standard estimate of ``E_p[z|x]`` (compared to ``E_q[z|x]``) and
   of the posterior excess kurtosis (is the true posterior even Gaussian?).

Run::

    uv run python -m examples.variational_mnist.validate_conjugation
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jax import Array
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from ..shared import example_paths  # noqa: E402
from . import hierarchical_experiment as H  # noqa: E402,N812
from .hierarchical import VariationalHierarchical  # noqa: E402

N_POINTS = 48  # test x's to average over
M_SAMPLES = 4000  # importance samples per x
LADDER = [1, 5, 25, 100, 500, 2000, M_SAMPLES]
COLORS = {"diagonal": "#888888", "chain": "#1f77b4", "chordal": "#d62728"}


def train_params(model: VariationalHierarchical, train_data: Array, steps: int, key: Array) -> Array:
    """Short training reusing the experiment's loss, returning final params."""
    k_init, k_train = jax.random.split(key)
    params = model.initialize_from_sample(k_init, train_data, 0.0, 0.3)
    schedule = optax.warmup_cosine_decay_schedule(0.0, H.LR, H.LR_WARMUP, steps, end_value=0.0)
    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(H.GRAD_CLIP), optax.adam(schedule)), 100
    )
    opt_state = optimizer.init(params)

    def loss_fn(p: Array, k: Array, batch: Array, beta: Array) -> Array:
        ke, kc, ki = jax.random.split(k, 3)
        return (-model.mean_elbo(ke, p, batch, H.MC_SAMPLES)
                + beta * H.LAMBDA_GEN * model.prior_conjugation_loss(kc, p, H.CONJ_SAMPLES)
                + H.LAMBDA_INNER * model.mean_recognition_inner_loss(ki, p, batch, H.MC_SAMPLES))

    @jax.jit
    def step(carry, g):
        p, opt_state, k = carry
        beta = jnp.minimum(1.0, g / H.WARMUP)
        k, kb, kl = jax.random.split(k, 3)
        batch = train_data[jax.random.choice(kb, train_data.shape[0], (H.BATCH,))]
        _, grads = jax.value_and_grad(loss_fn)(p, kl, batch, beta)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return (optax.apply_updates(p, updates), opt_state, k), None

    (params, _, _), _ = jax.lax.scan(step, (params, opt_state, k_train), jnp.arange(steps))
    return params


def log_weights_for_x(model: VariationalHierarchical, params: Array, x: Array,
                      key: Array, m: int) -> tuple[Array, Array]:
    """Return (log_w over m samples, z-samples) with log_w = log p(x,y,z) - log q(y,z|x)."""
    ys, zs = model.sample_posterior(key, params, x, m)
    log_w = jax.vmap(lambda y, z: model.log_density_joint(params, x, y, z)
                     - model.log_q(params, x, y, z))(ys, zs)
    return log_w, zs


def analyze(model: VariationalHierarchical, params: Array, xs: Array, key: Array) -> dict:
    keys = jax.random.split(key, xs.shape[0])
    log_w, zs = jax.vmap(lambda x, k: log_weights_for_x(model, params, x, k, M_SAMPLES))(xs, keys)
    # log_w: (N, M); zs: (N, M, z_dim)

    # ELBO = E_q[log w]; IWAE(K) via bootstrap subsamples of the M-pool.
    def iwae_at(k: int, kb: Array) -> Array:
        idx = jax.random.randint(kb, (xs.shape[0], 8, k), 0, M_SAMPLES)  # 8 bootstraps
        lw = jnp.take_along_axis(log_w[:, None, :], idx, axis=2)  # (N,8,k)
        return jnp.mean(logsumexp(lw, axis=2) - jnp.log(k))  # mean over x and bootstraps

    ladder = jnp.array([iwae_at(k, jax.random.fold_in(key, k)) for k in LADDER])
    elbo = float(jnp.mean(jnp.mean(log_w, axis=1)))
    logp = float(ladder[-1])

    # Self-normalized weights, ESS/M, posterior moment match, Gaussianity.
    w = jax.nn.softmax(log_w, axis=1)  # (N, M)
    ess = jnp.mean(1.0 / jnp.sum(w**2, axis=1)) / M_SAMPLES  # in [1/M, 1]

    ep_z = jnp.einsum("nm,nmd->nd", w, zs)  # SNIS gold-standard E_p[z|x]
    eq_z = jnp.mean(zs, axis=1)  # E_q[z|x] (unweighted q-samples)

    # Weighted excess kurtosis of z under the true posterior (0 == Gaussian).
    centered = zs - ep_z[:, None, :]
    var = jnp.einsum("nm,nmd->nd", w, centered**2)
    m4 = jnp.einsum("nm,nmd->nd", w, centered**4)
    exkurt = jnp.mean(m4 / (var**2 + 1e-12) - 3.0)

    return {
        "kind": model.__class__.__name__, "elbo": elbo, "logp": logp,
        "gap": logp - elbo, "ess": float(ess), "exkurt": float(exkurt),
        "ladder": [float(v) for v in ladder],
        "ep_z": ep_z, "eq_z": eq_z,
    }


def main() -> None:
    key = jax.random.PRNGKey(0)
    k_data, k_pts = jax.random.split(key)
    train_data, test_data, ceiling = H.make_data(k_data)
    xs = test_data[:N_POINTS]
    print(f"toy MoG-in-R^{H.OBS_DIM}, ceiling {ceiling:.3f}\n")

    results = []
    for kind in ["diagonal", "chain", "chordal"]:
        model = H.build_model(kind)
        params = train_params(model, train_data, 2000, jax.random.fold_in(key, hash(kind) % 100))
        r = analyze(model, params, xs, jax.random.fold_in(k_pts, 1))
        r["kind"] = kind
        results.append(r)
        print(f"{kind:8s}  ELBO {r['elbo']:7.2f}  logp~ {r['logp']:7.2f}  "
              f"gap {r['gap']:5.2f}  ESS/K {r['ess']:.3f}  exkurt {r['exkurt']:+.2f}")

    # Figure: IWAE ladder, ESS bars, posterior-mean scatter (chordal).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for r in results:
        axes[0].plot(LADDER, r["ladder"], "-o", color=COLORS[r["kind"]], label=r["kind"], ms=4)
    axes[0].axhline(ceiling, color="green", ls=":", label="data ceiling")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("K (importance samples)")
    axes[0].set_ylabel("IWAE(K)  [ELBO at K=1 -> log p(x)]")
    axes[0].set_title("Inference gap: IWAE ladder")
    axes[0].legend(fontsize=8)

    kinds = [r["kind"] for r in results]
    axes[1].bar(range(3), [r["ess"] for r in results], color=[COLORS[k] for k in kinds])
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(kinds)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("ESS / K   (1 == q equals true posterior)")
    axes[1].set_title("Posterior fidelity of the conjugate recognition")
    for i, r in enumerate(results):
        axes[1].text(i, r["ess"], f"{r['ess']:.2f}", ha="center", va="bottom", fontsize=9)

    rc = next(r for r in results if r["kind"] == "chordal")
    lo = float(min(rc["ep_z"].min(), rc["eq_z"].min()))
    hi = float(max(rc["ep_z"].max(), rc["eq_z"].max()))
    axes[2].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
    axes[2].scatter(rc["eq_z"].ravel(), rc["ep_z"].ravel(), s=10, color=COLORS["chordal"], alpha=0.6)
    axes[2].set_xlabel(r"$E_q[z\mid x]$ (conjugate recognition)")
    axes[2].set_ylabel(r"$E_p[z\mid x]$ (gold standard, SNIS)")
    axes[2].set_title("Posterior-mean match (chordal)")

    results_dir = example_paths(__file__).results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "validate_conjugation.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
