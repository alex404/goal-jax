"""Variational conjugation for harmoniums: approximate posterior with learnable conjugation correction and ELBO computation.

The approximate posterior has the same structure as the exact conjugate posterior, but with learned conjugation parameters ``\\rho`` replacing the analytically derived ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Triple
from ..manifold.embedding import LinearEmbedding
from ..manifold.map import AffineMap
from .base import Differentiable, ExponentialFamily
from .harmonium import Harmonium


class VariationalConjugated[
    Observable: Differentiable,
    Posterior: Differentiable,
    Conjugation: ExponentialFamily,
](
    Triple[Posterior, AffineMap[Posterior, Observable], Conjugation],
    ABC,
):
    """Variational harmonium with parameter layout ``[prior, lkl, rho]``.

    Reinterprets an underlying harmonium (``gen_hrm``) as a directed generative
    model: the harmonium's observable + interaction blocks supply the likelihood
    $p(x|z)$ and a separately-stored prior supplies $p(z; \\theta_Z)$. The
    variational layer adds learned conjugation parameters $\\rho$ that replace
    the analytically derived ones in the conjugate posterior

    $$q(z|x; \\rho) \\propto \\exp\\bigl((\\theta_Z + \\Theta_{XZ}^T s_X(x) - \\iota(\\rho)) \\cdot s_Z(z)\\bigr).$$

    Unlike :class:`AnalyticConjugated` (whose parameter layout *is* the underlying
    harmonium's layout), :class:`VariationalConjugated` stores its parameters as
    an explicit ``Triple[prior, lkl_fun, rho]``: the harmonium's redundant
    top-level layout is not used here. ``gen_hrm`` is exposed for forward
    computations (``posterior_at``, ``likelihood_at``, manifold types) but does
    not define this class's parameter layout.

    The ELBO decomposes as
    $\\mathcal{L}_\\beta(x) = \\mathbb{E}_q[\\log p(x|z)] - \\beta \\cdot D_{\\mathrm{KL}}(q \\| p)$,
    where the reconstruction term uses a score-function (REINFORCE) estimator for
    the intractable $\\mathbb{E}_q[\\psi_X]$ component.
    """

    # Contract

    @property
    @abstractmethod
    def gen_hrm(self) -> Harmonium[Observable, Posterior]:
        """Underlying generative harmonium --- supplies manifold structure and forward-computation methods. Not the source of this class's parameter layout."""

    @property
    @abstractmethod
    def rho_emb(self) -> LinearEmbedding[Conjugation, Posterior]:
        """Embedding from conjugation manifold to posterior manifold."""

    # Triple slots

    @property
    @override
    def fst_man(self) -> Posterior:
        return self.gen_hrm.pst_man

    @property
    @override
    def snd_man(self) -> AffineMap[Posterior, Observable]:
        return self.gen_hrm.lkl_fun_man

    @property
    @override
    def trd_man(self) -> Conjugation:
        return self.rho_emb.sub_man

    # Parameter access

    @property
    def rho_man(self) -> Conjugation:
        """Manifold for rho correction parameters."""
        return self.trd_man

    @property
    def obs_man(self) -> Observable:
        """Observable manifold of the underlying harmonium."""
        return self.gen_hrm.obs_man

    @property
    def pst_man(self) -> Posterior:
        """Posterior manifold of the underlying harmonium."""
        return self.gen_hrm.pst_man

    def conjugation_parameters(self, params: Array) -> Array:
        """Extract conjugation parameters $\\rho$ from full parameters."""
        _, _, rho = self.split_coords(params)
        return rho

    def prior_params(self, params: Array) -> Array:
        """Extract prior parameters $\\theta_Z$ defining $p(z; \\theta_Z)$."""
        prior, _, _ = self.split_coords(params)
        return prior

    def likelihood_function(self, params: Array) -> Array:
        """Extract the stored likelihood affine map $\\eta \\mapsto \\theta_X + \\Theta_{XZ} \\cdot \\eta$."""
        _, lkl, _ = self.split_coords(params)
        return lkl

    def generative_params(self, params: Array) -> Array:
        """Extract prior + likelihood concatenated --- the generative-model subset (no $\\rho$).

        Useful for optimizers that train the generative model independently of the variational correction (e.g. analytical-$\\rho$ training, where $\\rho$ is set by least-squares rather than gradient descent). Inverse: :meth:`join_generative_and_rho`.
        """
        prior, lkl, _ = self.split_coords(params)
        return jnp.concatenate([prior, lkl])

    def join_generative_and_rho(self, gen_params: Array, rho: Array) -> Array:
        """Inverse of :meth:`generative_params`: reconstruct full params from the generative subset and $\\rho$."""
        prior_dim = self.pst_man.dim
        prior = gen_params[:prior_dim]
        lkl = gen_params[prior_dim:]
        return self.join_coords(prior, lkl, rho)

    # Approximate conjugation

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Compute likelihood natural parameters $p(x|z)$ at latent $z$."""
        _, lkl, _ = self.split_coords(params)
        mz = self.pst_man.sufficient_statistic(z)
        return self.gen_hrm.lkl_fun_man(lkl, mz)

    def approximate_posterior_at(self, params: Array, x: Array) -> Array:
        """Compute approximate posterior natural parameters $\\eta_q = \\theta_Z + \\Theta_{XZ}^T s_X(x) - \\iota(\\rho)$."""
        prior, lkl, rho = self.split_coords(params)
        obs_params, int_params = self.gen_hrm.lkl_fun_man.split_coords(lkl)
        hrm_params = self.gen_hrm.join_coords(obs_params, int_params, prior)
        exact_posterior = self.gen_hrm.posterior_at(hrm_params, x)
        return self.rho_emb.translate(exact_posterior, -rho)

    # ELBO

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Compute $D_{\\mathrm{KL}}(q(z|x) \\| p(z))$."""
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)
        return self.pst_man.relative_entropy(q_params, p_params)

    def elbo_reconstruction_term(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Compute $\\mathbb{E}_q[\\log p(x|z)]$ via analytic/MC split with score correction.

        Decomposes as

        .. math::

            \\mathbb{E}_q[\\log p(x|z)] =
            \\underbrace{s_X(x) \\cdot (\\theta_X + \\Theta_{XZ} \\cdot \\mathbb{E}_q[s_Z(z)])}_{\\text{analytic}}
            - \\underbrace{\\mathbb{E}_q[\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z))]}_{\\text{MC}}
            + \\log h_X(x)

        The analytic term computes $\\mathbb{E}_q[s_Z]$ exactly via ``to_mean()``,
        so its gradients are exact (no MC, no score correction needed).

        The MC term estimates $\\mathbb{E}_q[\\psi_X]$ using the score-function
        (REINFORCE) estimator: samples are treated as fixed evaluation points,
        and a surrogate term recovers the gradient through the sampling
        distribution. This is correct for all posteriors, including those with
        non-differentiable samplers (e.g. VonMises rejection sampling).

        The full gradient of $-\\mathbb{E}_q[\\psi_X]$ decomposes by the Leibniz rule into:

        1. Direct: $-\\mathbb{E}_q[\\partial\\psi_X/\\partial\\theta]$ --- captured by ``mc_term``
        2. Score: $-\\mathbb{E}_q[\\psi_X \\cdot \\nabla_\\theta\\log q]$ --- captured by ``reinforce``
        """
        s_x = self.obs_man.sufficient_statistic(x)
        _, lkl, _ = self.split_coords(params)
        q_params = self.approximate_posterior_at(params, x)

        # --- Analytic term: s_X(x) . (theta_X + Theta_XZ . E_q[s_Z(z)]) ---
        # E_q[s_Z] = to_mean(q_params), which is differentiable w.r.t. q_params.
        # No MC, no score correction needed.
        mean_q = self.pst_man.to_mean(q_params)
        lkl_at_mean = self.gen_hrm.lkl_fun_man(lkl, mean_q)
        analytic_term = jnp.dot(s_x, lkl_at_mean)

        # --- MC term: E_q[psi_X(.)] via score-function estimator ---

        # [SG-1] Score-function estimator: samples are evaluation points, not
        # carriers of gradient. The gradient of E_q[psi_X] w.r.t. parameters
        # of q comes from nabla(log q) in the REINFORCE term below, not from
        # moving the sample locations.
        z_samples = jax.lax.stop_gradient(self.pst_man.sample(key, q_params, n_samples))

        def psi_at_z(z: Array) -> Array:
            mz = self.pst_man.sufficient_statistic(z)
            lkl_params = self.gen_hrm.lkl_fun_man(lkl, mz)
            return self.obs_man.log_partition_function(lkl_params)

        psi_vals = jax.vmap(psi_at_z)(z_samples)

        # mc_term captures the direct gradient E_q[d(psi_X)/d(theta)] at
        # fixed z: how changing (theta_X, Theta) directly changes psi_X.
        mc_term = jnp.mean(psi_vals)

        # --- REINFORCE correction: score gradient of E_q[psi_X] ---

        # [SG-2] psi_vals must be stopped here because it is used as the
        # scalar reward signal in the REINFORCE surrogate. If its dependence
        # on (theta_X, Theta) were retained, differentiating would produce a
        # spurious cross-term E_q[d(psi_X)/d(theta) . log q] that has no
        # probabilistic meaning and would corrupt the gradient. The direct
        # gradient E_q[d(psi_X)/d(theta)] is already captured by mc_term.
        psi_stopped = jax.lax.stop_gradient(psi_vals)
        baseline = jnp.mean(psi_stopped)
        centered = psi_stopped - baseline

        log_q = jax.vmap(lambda z: self.pst_man.log_density(q_params, z))(z_samples)

        # The surrogate: its VALUE is meaningless, but its GRADIENT w.r.t.
        # parameters of q gives -E_q[(psi_X - b) . nabla(log q)], which is
        # the score-function estimate of -nabla E_q[psi_X] (with variance-
        # reducing baseline b). Combined with mc_term, this gives the full
        # Leibniz-rule gradient.
        reinforce = -jnp.mean(centered * log_q)

        return analytic_term - mc_term + self.obs_man.log_base_measure(x) + reinforce

    def elbo_at(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> Array:
        """Compute the ELBO $\\mathcal{L}_\\beta(x) = \\mathbb{E}_q[\\log p(x|z)] - \\beta \\cdot D_{\\mathrm{KL}}(q \\| p)$."""
        recon = self.elbo_reconstruction_term(key, params, x, n_samples)
        kl = self.elbo_divergence(params, x)
        return recon - kl_weight * kl

    def mean_elbo(
        self,
        key: Array,
        params: Array,
        xs: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> Array:
        """Compute mean ELBO over a batch of observations."""
        batch_size = xs.shape[0]
        keys = jax.random.split(key, batch_size)

        elbos = jax.vmap(lambda k, x: self.elbo_at(k, params, x, n_samples, kl_weight))(
            keys, xs
        )

        return jnp.mean(elbos)

    # Reduced learning signal

    def reduced_learning_signal(self, params: Array, z: Array) -> Array:
        """Compute the reduced learning signal $\\tilde{f}(z) = \\rho \\cdot \\pi(s_Z(z)) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z))$.

        The score-function gradient of $-\\mathbb{E}_q[\\psi_X(\\eta(z))]$ admits an arbitrary control variate; $\\rho \\cdot s_Z(z)$ is the optimal linear-in-$s_Z$ choice, and $\\tilde{f}$ is what remains of the learning signal after subtracting it. The per-$x$ KL gap is exactly $\\log \\mathbb{E}_q[e^{\\tilde{f}}] - \\mathbb{E}_q[\\tilde{f}]$, so $\\tilde{f}$ constant $\\iff$ zero ammorization gap; its variance under the prior is used as a training regularizer.
        """
        _, lkl, rho = self.split_coords(params)
        s_z = self.pst_man.sufficient_statistic(z)
        s_rho = self.rho_emb.project(s_z)
        term1 = jnp.dot(rho, s_rho)
        lkl_params = self.gen_hrm.lkl_fun_man(lkl, s_z)
        term2 = self.obs_man.log_partition_function(lkl_params)
        return term1 - term2

    # Generative

    def sample(self, key: Array, params: Array, n: int) -> Array:
        """Sample from the generative model $p(x, z)$ via ancestral sampling.

        Samples from the explicit prior $p(z; \\theta_Z)$, NOT the harmonium
        marginal. Conjugation parameters don't affect the generative model.
        """
        key1, key2 = jax.random.split(key)
        p_params = self.prior_params(params)
        z_samples = self.pst_man.sample(key1, p_params, n)

        def sample_x_given_z(subkey: Array, z: Array) -> Array:
            lkl_params = self.likelihood_at(params, z)
            return self.obs_man.sample(subkey, lkl_params, 1)[0]

        x_keys = jax.random.split(key2, n)
        x_samples = jax.vmap(sample_x_given_z)(x_keys, z_samples)

        return jnp.concatenate([x_samples, z_samples], axis=-1)

    def log_density(self, params: Array, xz: Array) -> Array:
        """Compute joint log density $\\log p(x, z) = \\log p(z) + \\log p(x|z)$ for a joint data point."""
        x = xz[..., : self.obs_man.data_dim]
        z = xz[..., self.obs_man.data_dim :]
        prior, lkl, _ = self.split_coords(params)
        log_pz = self.pst_man.log_density(prior, z)
        mz = self.pst_man.sufficient_statistic(z)
        lkl_params = self.gen_hrm.lkl_fun_man(lkl, mz)
        log_px_given_z = self.obs_man.log_density(lkl_params, x)
        return log_pz + log_px_given_z

    # Initialization

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize full model parameters with zero conjugation correction."""
        rho = jnp.zeros(self.rho_man.dim)
        hrm_params = self.gen_hrm.initialize(key, location, shape)
        obs_params, int_params, prior_params = self.gen_hrm.split_coords(hrm_params)
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        return self.join_coords(prior_params, lkl_params, rho)

    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize parameters using sample data for observable biases."""
        rho = jnp.zeros(self.rho_man.dim)
        hrm_params = self.gen_hrm.initialize_from_sample(key, sample, location, shape)
        obs_params, int_params, prior_params = self.gen_hrm.split_coords(hrm_params)
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        return self.join_coords(prior_params, lkl_params, rho)


# --- Free functions: diagnostics, fitting helpers, downstream uses ---


def regress_conjugation_parameters[
    Observable: Differentiable,
    Posterior: Differentiable,
    Conjugation: ExponentialFamily,
](
    model: VariationalConjugated[Observable, Posterior, Conjugation],
    key: Array,
    params: Array,
    n_samples: int,
) -> tuple[Array, Array, Array, Array]:
    """Fit conjugation parameters by least-squares regression against samples from the prior.

    Solves :math:`\\rho = \\arg\\min_\\rho \\sum_k (\\chi + \\rho \\cdot s_\\rho(z_k) - \\psi_X(\\theta_X + \\Theta \\cdot s_Z(z_k)))^2`. Returns ``(rho, r_squared, chi, residual_var)`` where ``chi`` is the intercept and ``residual_var`` is the conjugation error $\\mathrm{Var}[\\tilde{f}]$ estimated from the regression samples.

    A pragmatic fitting/initialization heuristic --- theoretically a black sheep in the variational conjugation family, hence its free-function status.
    """
    prior_params, lkl, _ = model.split_coords(params)

    # [SG-3] The sampler may be non-differentiable (e.g. VonMises uses
    # rejection sampling via while_loop, which JAX cannot reverse-diff).
    # The samples are just evaluation points for the regression; the
    # gradient we need flows through the regression targets psi_X, which
    # depend smoothly on (theta_X, Theta).
    z_samples = jax.lax.stop_gradient(model.pst_man.sample(key, prior_params, n_samples))

    def design_and_target(z: Array) -> tuple[Array, Array]:
        s_z = model.pst_man.sufficient_statistic(z)
        s_rho = model.rho_emb.project(s_z)
        lkl_params = model.gen_hrm.lkl_fun_man(lkl, s_z)
        psi_x = model.obs_man.log_partition_function(lkl_params)
        return s_rho, psi_x

    s_rho_all, psi_all = jax.vmap(design_and_target)(z_samples)

    design = jnp.concatenate([jnp.ones((n_samples, 1)), s_rho_all], axis=1)
    coeffs = jnp.linalg.lstsq(design, psi_all, rcond=None)[0]
    chi = coeffs[0]
    rho = coeffs[1:]

    residuals = psi_all - design @ coeffs
    ss_res = jnp.sum(residuals**2)
    ss_tot = jnp.sum((psi_all - jnp.mean(psi_all)) ** 2)
    r_squared = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-8)

    return rho, r_squared, chi, jnp.var(residuals)


def conjugation_metrics[
    Observable: Differentiable,
    Posterior: Differentiable,
    Conjugation: ExponentialFamily,
](
    model: VariationalConjugated[Observable, Posterior, Conjugation],
    key: Array,
    params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array]:
    """Compute conjugation quality metrics (variance, std, $R^2$) under the prior.

    $R^2 = 1 - \\mathrm{Var}[\\tilde{f}] / \\mathrm{Var}[\\psi_X(\\eta(z))]$ measures how well the linear correction $\\rho \\cdot s(z)$ approximates $\\psi_X(\\eta(z))$: 1 = perfect conjugation, 0 = no explanatory power, <0 = worse than constant.
    """
    prior_params, lkl, rho = model.split_coords(params)
    z_samples = model.pst_man.sample(key, prior_params, n_samples)

    def compute_both(z: Array) -> tuple[Array, Array]:
        s_z = model.pst_man.sufficient_statistic(z)
        s_rho = model.rho_emb.project(s_z)
        lkl_params = model.gen_hrm.lkl_fun_man(lkl, s_z)
        psi_x = model.obs_man.log_partition_function(lkl_params)
        f_tilde = jnp.dot(rho, s_rho) - psi_x
        return f_tilde, psi_x

    f_vals, psi_vals = jax.vmap(compute_both)(z_samples)

    var_f = jnp.var(f_vals)
    std_f = jnp.sqrt(var_f)
    var_psi = jnp.var(psi_vals)

    variance_threshold = 1e-6
    raw_r2 = 1.0 - var_f / jnp.maximum(var_psi, variance_threshold)
    r_squared = jnp.where(
        var_psi > variance_threshold,
        jnp.clip(raw_r2, -10.0, 1.0),
        jnp.nan,
    )

    return var_f, std_f, r_squared


def reconstruct[
    Observable: Differentiable,
    Posterior: Differentiable,
    Conjugation: ExponentialFamily,
](
    model: VariationalConjugated[Observable, Posterior, Conjugation],
    params: Array,
    x: Array,
) -> Array:
    """Reconstruct observable means via mean-field approximation: posterior mean stats through likelihood."""
    q_params = model.approximate_posterior_at(params, x)
    z_mean_stats = model.pst_man.to_mean(q_params)
    _, lkl, _ = model.split_coords(params)
    lkl_natural = model.gen_hrm.lkl_fun_man(lkl, z_mean_stats)
    return model.obs_man.to_mean(lkl_natural)


def reconstruction_error[
    Observable: Differentiable,
    Posterior: Differentiable,
    Conjugation: ExponentialFamily,
](
    model: VariationalConjugated[Observable, Posterior, Conjugation],
    params: Array,
    xs: Array,
) -> Array:
    """Compute mean squared reconstruction error over a batch."""
    recons = jax.vmap(lambda x: reconstruct(model, params, x))(xs)
    return jnp.mean((xs - recons) ** 2)
