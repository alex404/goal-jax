"""Fit a two-view canonical correlation analysis model.

Two observable Gaussians of different dimension and different covariance structure share
one latent. The graph is a fork rather than a chain,

.. code-block:: text

       z
      / \\
     x   y

so the model has two roots, its interaction holds one clique per branch, and its
conjugation parameters are the sum of the two branches'.

Data comes from an explicit two-view process --- draw a shared latent, project it into each
view, add independent noise --- rather than from the model's own parameters, so that the
shared structure is known and strong by construction. The run records the training curve,
how well the shared latent is recovered, and how well the *cross-view* covariance is
reproduced, that last being the structure a two-view model exists to capture.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import Diagonal, PositiveDefinite
from goal.models import CanonicalCorrelationAnalysis

from ..shared import example_paths, jax_cli
from .types import CCAResults


def generate_two_views(
    key: Array,
    n_samples: int,
    fst_loadings: Array,
    snd_loadings: Array,
    noise_std: float,
) -> tuple[Array, Array]:
    """Draw a shared latent and project it into two noisy views.

    Returns ``(latents, observations)`` with observations laid out as ``[x | y]``, which is
    the order the model's observable pair expects.
    """
    key_lat, key_fst, key_snd = jax.random.split(key, 3)
    lat_dim = fst_loadings.shape[1]
    latents = jax.random.normal(key_lat, (n_samples, lat_dim))
    fst = latents @ fst_loadings.T + noise_std * jax.random.normal(
        key_fst, (n_samples, fst_loadings.shape[0])
    )
    snd = latents @ snd_loadings.T + noise_std * jax.random.normal(
        key_snd, (n_samples, snd_loadings.shape[0])
    )
    return latents, jnp.concatenate([fst, snd], axis=-1)


def linear_align(source: Array, target: Array) -> Array:
    """Map ``source`` onto ``target`` by least squares, both centred.

    A two-view latent is identified only up to an invertible linear map, so a recovered
    latent must be aligned by a general linear map --- not a rotation --- before it can be
    compared to the one that generated the data.
    """
    src = source - jnp.mean(source, axis=0)
    tgt = target - jnp.mean(target, axis=0)
    weights, *_ = jnp.linalg.lstsq(src, tgt, rcond=None)
    return src @ weights


def cross_covariance(
    model: CanonicalCorrelationAnalysis[Any, Any, Any], obs: Array
) -> Array:
    """Empirical covariance between the two views, shape ``(fst_dim, snd_dim)``."""
    fst = obs[:, : model.fst_dim] - jnp.mean(obs[:, : model.fst_dim], axis=0)
    snd = obs[:, model.fst_dim :] - jnp.mean(obs[:, model.fst_dim :], axis=0)
    return (fst.T @ snd) / fst.shape[0]


def main() -> None:
    jax_cli()
    # Gradient ascent on Gaussian natural parameters has to stay inside the
    # positive-definite cone, and single precision is not reliably enough to do so here.
    jax.config.update("jax_enable_x64", True)
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)

    fst_dim, snd_dim, lat_dim = 4, 3, 2
    n_samples, n_steps, noise_std = 2000, 8000, 0.4
    learning_rate, grad_clip = 5e-3, 0.5
    n_model_samples = 20000

    model = CanonicalCorrelationAnalysis(
        fst_dim=fst_dim,
        fst_rep=PositiveDefinite(),
        snd_dim=snd_dim,
        snd_rep=Diagonal(),
        lat_dim=lat_dim,
        pst_rep=PositiveDefinite(),
    )
    print(
        f"CCA over {model.clq_set.n_nodes} nodes, levels {model.clq_set.level_sets}"
    )
    print(f"  cliques {model.clq_set.canonical_cliques}")
    print(f"  blocks  {model.clique_dims}  (dim {model.dim})")

    fst_loadings = jnp.array([[1.4, 0.2], [0.9, -0.8], [0.1, 1.3], [-1.1, 0.5]])
    snd_loadings = jnp.array([[1.2, -0.4], [-0.3, 1.5], [0.8, 0.9]])

    key, key_data = jax.random.split(key)
    true_latents, raw_obs = generate_two_views(
        key_data, n_samples, fst_loadings, snd_loadings, noise_std
    )
    # Standardize per dimension: the two views are on different scales, and an
    # unconstrained natural-parameter optimizer leaves the positive-definite cone if the
    # curvature across coordinates is too uneven.
    obs = (raw_obs - jnp.mean(raw_obs, axis=0)) / jnp.std(raw_obs, axis=0)

    key, key_init = jax.random.split(key)
    params = model.initialize(key_init, shape=0.1)
    # Clipping keeps Adam from overshooting the positive-definite boundary, which shows
    # up as a large transient drop in the likelihood; decaying the step size settles the
    # last few percent, which is what the cross-view covariance is sensitive to.
    schedule = optax.cosine_decay_schedule(learning_rate, n_steps, alpha=0.02)
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(schedule))
    opt_state = optimizer.init(params)

    def objective(p: Array) -> Array:
        return -model.average_log_observable_density(p, obs)

    def step(
        carry: tuple[Array, optax.OptState], _: None
    ) -> tuple[tuple[Array, optax.OptState], Array]:
        p, state = carry
        loss, grad = jax.value_and_grad(objective)(p)
        updates, state = optimizer.update(grad, state, p)
        return (cast(Array, optax.apply_updates(p, updates)), state), -loss

    print(f"Fitting {n_steps} steps on {n_samples} samples...")
    (params, _), log_likelihoods = jax.lax.scan(
        step, (params, opt_state), None, length=n_steps
    )
    start, end = float(log_likelihoods[0]), float(log_likelihoods[-1])
    print(f"  log-likelihood {start:.3f} -> {end:.3f}")

    inferred = jax.vmap(
        lambda x: model.pst_man.to_mean(model.posterior_at(params, x))[:lat_dim]
    )(obs)
    aligned = linear_align(inferred, true_latents)
    latent_rmse = float(
        jnp.sqrt(jnp.mean(jnp.sum((aligned - true_latents) ** 2, axis=1)))
    )
    print(f"  latent alignment RMSE {latent_rmse:.4f}")

    key, key_model_sample = jax.random.split(key)
    model_obs = model.sample(key_model_sample, params, n_model_samples)[
        :, : model.obs_man.data_dim
    ]
    data_cross = cross_covariance(model, obs)
    model_cross = cross_covariance(model, model_obs)
    cross_error = float(
        jnp.linalg.norm(data_cross - model_cross) / jnp.linalg.norm(data_cross)
    )
    print(f"  cross-covariance relative error {cross_error:.4f}")

    results = CCAResults(
        fst_dim=fst_dim,
        snd_dim=snd_dim,
        lat_dim=lat_dim,
        n_samples=n_samples,
        n_steps=n_steps,
        log_likelihoods=log_likelihoods.tolist(),
        true_latents=true_latents.tolist(),
        inferred_latents=aligned.tolist(),
        latent_rmse=latent_rmse,
        true_cross_covariance=data_cross.tolist(),
        learned_cross_covariance=model_cross.tolist(),
        cross_covariance_error=cross_error,
    )
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
