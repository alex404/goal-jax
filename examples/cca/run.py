"""Fit a two-view canonical correlation analysis model by gradient ascent and by EM.

Two observable Gaussians with different covariance structure share one latent. The graph is
a fork rather than a chain,

.. code-block:: text

       z
      / \\
     x   y

so the model has two roots, its interaction holds one clique per branch, and its
conjugation parameters are the sum of the two branches'.

Data comes from an explicit two-view process --- draw a shared one-dimensional latent, project
it into each of two two-dimensional views, and add noise private to each view: correlated noise
in $x$ (a full covariance) and independent noise of different sizes in $y$ (a diagonal
covariance) --- so that the shared structure is known and strong by construction, and both
views can be drawn. The same data is fit twice: by gradient ascent on the log-likelihood, with
a posterior of any covariance structure, and by exact expectation-maximization, which needs a
full-covariance latent. For each fit the run records the training curve, the covariance the
model gives each view, how well the shared latent is recovered, how well one view predicts the
other through the latent, and how well the *cross-view* covariance is reproduced, that last
being the structure a two-view model exists to capture.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from jax import Array

from goal.geometry import Diagonal, PositiveDefinite
from goal.models import (
    AnalyticCanonicalCorrelationAnalysis,
    CanonicalCorrelationAnalysis,
)

from ..shared import example_paths, jax_cli
from .types import CCAFit, CCAResults, ViewFit

type AnyCCA = (
    CanonicalCorrelationAnalysis[Any, Any, Any]
    | AnalyticCanonicalCorrelationAnalysis[Any, Any]
)


def generate_two_views(
    key: Array,
    n_samples: int,
    fst_loading: Array,
    snd_loading: Array,
    fst_noise_cov: Array,
    snd_noise_std: Array,
) -> tuple[Array, Array]:
    """Draw a shared scalar latent and project it into two noisy views.

    Returns ``(latents, observations)``, of shapes ``(n_samples, 1)`` and ``(n_samples, 4)``, with
    observations laid out as ``[x | y]``, which is the order the model's observable pair expects.
    """
    key_lat, key_fst, key_snd = jax.random.split(key, 3)
    latents = jax.random.normal(key_lat, (n_samples, 1))
    fst_noise = jax.random.multivariate_normal(
        key_fst, jnp.zeros(2), fst_noise_cov, (n_samples,)
    )
    snd_noise = snd_noise_std * jax.random.normal(key_snd, (n_samples, 2))
    fst = latents * fst_loading + fst_noise
    snd = latents * snd_loading + snd_noise
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


def signal_and_noise(view: Array, latents: Array) -> tuple[Array, Array]:
    """Split a view into the part the latent explains and the rest, by regressing it on the latent.

    Returns the shift of the view's mean for a one standard deviation change of the latent (the
    sign and scale of a latent are not identified, but this shift is, up to sign) and the
    covariance of the residual, the noise private to the view.
    """
    cov = jnp.cov(jnp.concatenate([view, latents], axis=1), rowvar=False)
    dim = view.shape[1]
    cov_vv, cov_vz, cov_zz = cov[:dim, :dim], cov[:dim, dim:], cov[dim:, dim:]
    loading = cov_vz @ jnp.linalg.inv(cov_zz)
    signal = (loading @ jnp.linalg.cholesky(cov_zz))[:, 0]
    return signal, cov_vv - loading @ cov_zz @ loading.T


def cross_covariance(fst_dim: int, obs: Array) -> Array:
    """Empirical covariance between the two views, shape ``(fst_dim, snd_dim)``."""
    fst = obs[:, :fst_dim] - jnp.mean(obs[:, :fst_dim], axis=0)
    snd = obs[:, fst_dim:] - jnp.mean(obs[:, fst_dim:], axis=0)
    return (fst.T @ snd) / fst.shape[0]


def evaluate_fit(
    key: Array,
    method: str,
    model: AnyCCA,
    params: Array,
    log_likelihoods: Array,
    obs: Array,
    true_latents: Array,
    n_model_samples: int,
) -> CCAFit:
    """Measure the view covariances, latent recovery, cross-view prediction and cross-view covariance of a fit.

    The covariances of the fitted model, and the regression of $y$ on $x$ they imply, are estimated
    from ``n_model_samples`` samples of the model.
    """
    start, end = float(log_likelihoods[0]), float(log_likelihoods[-1])
    print(f"  log-likelihood {start:.3f} -> {end:.3f}")

    inferred = jax.vmap(
        lambda x: model.pst_man.to_mean(model.posterior_at(params, x))[: model.lat_dim]
    )(obs)
    aligned = linear_align(inferred, true_latents)
    latent_rmse = float(
        jnp.sqrt(jnp.mean(jnp.sum((aligned - true_latents) ** 2, axis=1)))
    )
    print(f"  latent alignment RMSE {latent_rmse:.4f}")

    fst_dim = model.fst_dim
    obs_dim = model.obs_man.data_dim
    model_samples = model.sample(key, params, n_model_samples)
    model_obs = model_samples[:, :obs_dim]
    model_lat = model_samples[:, obs_dim : obs_dim + model.lat_dim]
    fst_signal, fst_noise = signal_and_noise(model_obs[:, :fst_dim], model_lat)
    snd_signal, snd_noise = signal_and_noise(model_obs[:, fst_dim:], model_lat)
    mean = jnp.mean(model_obs, axis=0)
    cov = jnp.cov(model_obs, rowvar=False)
    cov_ff, cov_fs = cov[:fst_dim, :fst_dim], cov[:fst_dim, fst_dim:]

    # E[y | x] under the fitted model: the second view predicted from the first, through the latent
    coefs = jnp.linalg.solve(cov_ff, cov_fs)
    predicted = mean[fst_dim:] + (obs[:, :fst_dim] - mean[:fst_dim]) @ coefs
    prediction_rmse = float(
        jnp.sqrt(jnp.mean(jnp.sum((predicted - obs[:, fst_dim:]) ** 2, axis=1)))
    )
    print(f"  prediction of y from x, RMSE {prediction_rmse:.4f}")

    data_cross = cross_covariance(fst_dim, obs)
    model_cross = cov_fs
    cross_error = float(
        jnp.linalg.norm(data_cross - model_cross) / jnp.linalg.norm(data_cross)
    )
    print(f"  cross-covariance relative error {cross_error:.4f}")

    return CCAFit(
        method=method,
        log_likelihoods=log_likelihoods.tolist(),
        fst_view=ViewFit(
            mean=mean[:fst_dim].tolist(),
            signal=fst_signal.tolist(),
            noise_cov=fst_noise.tolist(),
        ),
        snd_view=ViewFit(
            mean=mean[fst_dim:].tolist(),
            signal=snd_signal.tolist(),
            noise_cov=snd_noise.tolist(),
        ),
        inferred_latents=aligned.tolist(),
        latent_rmse=latent_rmse,
        predicted_snd=predicted.tolist(),
        prediction_rmse=prediction_rmse,
        learned_cross_covariance=model_cross.tolist(),
        cross_covariance_error=cross_error,
    )


def main() -> None:
    jax_cli()
    # Gradient ascent on Gaussian natural parameters has to stay inside the
    # positive-definite cone, and single precision is not reliably enough to do so here.
    jax.config.update("jax_enable_x64", True)
    paths = example_paths(__file__)
    key = jax.random.PRNGKey(0)

    fst_dim, snd_dim, lat_dim = 2, 2, 1
    n_samples = 2000
    n_grad_steps, learning_rate, grad_clip = 8000, 5e-3, 0.5
    n_em_steps = 100
    n_model_samples = 20000

    grad_model = CanonicalCorrelationAnalysis(
        fst_dim=fst_dim,
        fst_rep=PositiveDefinite(),
        snd_dim=snd_dim,
        snd_rep=Diagonal(),
        lat_dim=lat_dim,
        pst_rep=PositiveDefinite(),
    )
    em_model = AnalyticCanonicalCorrelationAnalysis(
        fst_dim=fst_dim,
        fst_rep=PositiveDefinite(),
        snd_dim=snd_dim,
        snd_rep=Diagonal(),
        lat_dim=lat_dim,
    )
    print(f"CCA over {em_model.n_nodes} nodes, crossings {em_model.crs_clqs}")
    print(f"  cliques {em_model.cliques}")
    print(f"  coordinate blocks {em_model.clq_dims}  (dim {em_model.dim})")

    # The latent moves x along (1, 0.6) and y along (-0.4, 1); x's noise is correlated along
    # (1, -1), y's noise is independent and larger in its first coordinate.
    fst_loading = jnp.array([1.0, 0.6])
    snd_loading = jnp.array([-0.4, 1.0])
    fst_noise_cov = jnp.array([[0.35, -0.25], [-0.25, 0.35]])
    snd_noise_std = jnp.array([0.6, 0.25])

    key, key_data = jax.random.split(key)
    true_latents, raw_obs = generate_two_views(
        key_data,
        n_samples,
        fst_loading,
        snd_loading,
        fst_noise_cov,
        snd_noise_std,
    )
    # Standardize per dimension: an unconstrained natural-parameter optimizer leaves the
    # positive-definite cone if the curvature across coordinates is too uneven.
    obs = (raw_obs - jnp.mean(raw_obs, axis=0)) / jnp.std(raw_obs, axis=0)

    # Both fits start from the same parameters: with a full-covariance posterior the two
    # models have the same parameter layout.
    key, key_init = jax.random.split(key)
    init_params = em_model.initialize(key_init, shape=0.1)

    ### Gradient ascent ###

    # Clipping keeps Adam from overshooting the positive-definite boundary, which shows
    # up as a large transient drop in the likelihood; decaying the step size settles the
    # last few percent, which is what the cross-view covariance is sensitive to.
    schedule = optax.cosine_decay_schedule(learning_rate, n_grad_steps, alpha=0.02)
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(schedule))
    opt_state = optimizer.init(init_params)

    def objective(p: Array) -> Array:
        return -grad_model.average_log_observable_density(p, obs)

    def grad_step(
        carry: tuple[Array, optax.OptState], _: None
    ) -> tuple[tuple[Array, optax.OptState], Array]:
        p, state = carry
        loss, grad = jax.value_and_grad(objective)(p)
        updates, state = optimizer.update(grad, state, p)
        return (cast(Array, optax.apply_updates(p, updates)), state), -loss

    print(f"\nGradient ascent: {n_grad_steps} steps on {n_samples} samples...")
    (grad_params, _), grad_lls = jax.lax.scan(
        grad_step, (init_params, opt_state), None, length=n_grad_steps
    )
    key, key_grad_sample = jax.random.split(key)
    grad_fit = evaluate_fit(
        key_grad_sample,
        "Gradient ascent",
        grad_model,
        grad_params,
        grad_lls,
        obs,
        true_latents,
        n_model_samples,
    )

    ### Expectation-maximization ###

    def em_step(p: Array, _: None) -> tuple[Array, Array]:
        ll = em_model.average_log_observable_density(p, obs)
        return em_model.expectation_maximization(p, obs), ll

    print(f"\nExpectation-maximization: {n_em_steps} steps on {n_samples} samples...")
    em_params, em_lls = jax.lax.scan(em_step, init_params, None, length=n_em_steps)
    key, key_em_sample = jax.random.split(key)
    em_fit = evaluate_fit(
        key_em_sample,
        "Expectation-maximization",
        em_model,
        em_params,
        em_lls,
        obs,
        true_latents,
        n_model_samples,
    )

    def true_view(view: Array) -> ViewFit:
        signal, noise = signal_and_noise(view, true_latents)
        return ViewFit(
            mean=jnp.mean(view, axis=0).tolist(),
            signal=signal.tolist(),
            noise_cov=noise.tolist(),
        )

    results = CCAResults(
        fst_dim=fst_dim,
        snd_dim=snd_dim,
        lat_dim=lat_dim,
        n_samples=n_samples,
        observations=obs.tolist(),
        true_latents=true_latents.tolist(),
        true_fst_view=true_view(obs[:, :fst_dim]),
        true_snd_view=true_view(obs[:, fst_dim:]),
        true_cross_covariance=cross_covariance(fst_dim, obs).tolist(),
        gradient_ascent=grad_fit,
        expectation_maximization=em_fit,
    )
    paths.save_analysis(results)
    print(f"\nResults saved to {paths.analysis_path}")


if __name__ == "__main__":
    main()
