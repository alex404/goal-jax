"""Variational conjugation for harmoniums: approximate posterior with learnable conjugation correction and ELBO computation.

The approximate posterior has the same structure as the exact conjugate posterior, but with learned conjugation parameters ``\\rho`` replacing the analytically derived ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Pair
from ..manifold.embedding import LinearEmbedding
from .base import Differentiable, ExponentialFamily
from .harmonium import Harmonium


@dataclass(frozen=True)
class VariationalConjugated[
    Observable: Differentiable,
    Posterior: Differentiable,
    Conjugation: ExponentialFamily,
](
    Pair[Conjugation, Harmonium[Observable, Posterior]],
    ABC,
):
    """Variational harmonium whose full parameter space is ``Pair[rho_params, harmonium_params]``.

    Reinterprets a harmonium as a directed generative model with explicit prior
    $p(z; \\theta_Z)$ and likelihood $p(x|z)$, then approximates the intractable
    posterior with a learnable member of the conjugate family:

    $$q(z|x; \\rho) \\propto \\exp\\bigl((\\theta_Z + \\Theta_{XZ}^T s_X(x) - \\iota(\\rho)) \\cdot s_Z(z)\\bigr)$$

    Mathematically, this has the same form as the true conjugate posterior, but
    $\\rho$ is learned rather than computed from conjugation. The ELBO decomposes as
    $\\mathcal{L}_\\beta(x) = \\mathbb{E}_q[\\log p(x|z)] - \\beta \\cdot D_{\\mathrm{KL}}(q \\| p)$,
    where the reconstruction term uses a score-function (REINFORCE) estimator for
    the intractable $\\mathbb{E}_q[\\psi_X]$ component.
    """

    # Fields

    hrm: Harmonium[Observable, Posterior]
    """The underlying harmonium model."""

    # Contract

    @property
    @abstractmethod
    def rho_emb(self) -> LinearEmbedding[Conjugation, Posterior]:
        """Embedding from conjugation manifold to posterior manifold."""

    # Overrides

    @property
    def rho_man(self) -> Conjugation:
        """Manifold for rho correction parameters."""
        return self.rho_emb.sub_man

    @property
    @override
    def fst_man(self) -> Conjugation:
        return self.rho_man

    @property
    @override
    def snd_man(self) -> Harmonium[Observable, Posterior]:
        return self.hrm

    # Methods

    @property
    def obs_man(self) -> Observable:
        """Observable manifold of the underlying harmonium."""
        return self.hrm.obs_man

    @property
    def pst_man(self) -> Posterior:
        """Posterior manifold of the underlying harmonium."""
        return self.hrm.pst_man

    def conjugation_parameters(self, params: Array) -> Array:
        """Extract conjugation parameters $\\rho$ from full parameters."""
        rho, _ = self.split_coords(params)
        return rho

    def generative_params(self, params: Array) -> Array:
        """Extract harmonium (generative model) parameters from full parameters."""
        _, hrm_params = self.split_coords(params)
        return hrm_params

    def approximate_posterior_at(self, params: Array, x: Array) -> Array:
        """Compute approximate posterior natural parameters $\\eta_q = \\theta_Z + \\Theta_{XZ}^T s_X(x) - \\iota(\\rho)$."""
        rho, hrm_params = self.split_coords(params)
        exact_posterior = self.hrm.posterior_at(hrm_params, x)
        return self.rho_emb.translate(exact_posterior, -rho)

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Compute likelihood natural parameters $p(x|z)$ at latent $z$."""
        _, hrm_params = self.split_coords(params)
        return self.hrm.likelihood_at(hrm_params, z)

    def prior_params(self, params: Array) -> Array:
        """Extract prior parameters $\\theta_Z$ defining $p(z; \\theta_Z)$, NOT the harmonium marginal."""
        _, hrm_params = self.split_coords(params)
        _, _, lat_params = self.hrm.split_coords(hrm_params)
        return lat_params

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Compute $D_{\\mathrm{KL}}(q(z|x) \\| p(z))$."""
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)
        return self.pst_man.relative_entropy(q_params, p_params)

    def elbo_reconstruction_term(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Compute $\\mathbb{E}_q[\\log p(x|z)]$ via analytic/MC split with score correction.

        Decomposes as:
        $$\\mathbb{E}_q[\\log p(x|z)] =
          \\underbrace{s_X(x) \\cdot (\\theta_X + \\Theta_{XZ} \\cdot \\mathbb{E}_q[s_Z(z)])}_{\\text{analytic}}
          - \\underbrace{\\mathbb{E}_q[\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z))]}_{\\text{MC}}
          + \\log h_X(x)$$

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
        _, hrm_params = self.split_coords(params)
        q_params = self.approximate_posterior_at(params, x)

        # --- Analytic term: s_X(x) . (theta_X + Theta_XZ . E_q[s_Z(z)]) ---
        # E_q[s_Z] = to_mean(q_params), which is differentiable w.r.t. q_params.
        # No MC, no score correction needed.
        mean_q = self.pst_man.to_mean(q_params)
        lkl_at_mean = self.hrm.lkl_fun_man(
            self.hrm.likelihood_function(hrm_params), mean_q
        )
        analytic_term = jnp.dot(s_x, lkl_at_mean)

        # --- MC term: E_q[psi_X(.)] via score-function estimator ---

        # [SG-1] Score-function estimator: samples are evaluation points, not
        # carriers of gradient. The gradient of E_q[psi_X] w.r.t. parameters
        # of q comes from nabla(log q) in the REINFORCE term below, not from
        # moving the sample locations.
        z_samples = jax.lax.stop_gradient(
            self.pst_man.sample(key, q_params, n_samples)
        )

        def psi_at_z(z: Array) -> Array:
            lkl_params = self.hrm.likelihood_at(hrm_params, z)
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

    def regress_conjugation_parameters(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, Array, Array]:
        """Fit conjugation parameters by least-squares regression.

        Solves $\\rho = \\arg\\min_\\rho \\sum_k (\\chi + \\rho \\cdot s_\\rho(z_k) - \\psi_X(\\theta_X + \\Theta \\cdot s_Z(z_k)))^2$.
        Returns ``(rho, r_squared, chi)`` where ``chi`` is the intercept.
        """
        _, hrm_params = self.split_coords(params)
        p_params = self.prior_params(params)

        # [SG-3] The sampler may be non-differentiable (e.g. VonMises uses
        # rejection sampling via while_loop, which JAX cannot reverse-diff).
        # The samples are just evaluation points for the regression; the
        # gradient we need flows through the regression targets psi_X, which
        # depend smoothly on (theta_X, Theta).
        z_samples = jax.lax.stop_gradient(
            self.pst_man.sample(key, p_params, n_samples)
        )

        def design_and_target(z: Array) -> tuple[Array, Array]:
            s_z = self.pst_man.sufficient_statistic(z)
            s_rho = self.rho_emb.project(s_z)
            lkl_params = self.hrm.likelihood_at(hrm_params, z)
            psi_x = self.obs_man.log_partition_function(lkl_params)
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

        return rho, r_squared, chi

    def reduced_learning_signal(self, params: Array, z: Array) -> Array:
        """Compute the reduced learning signal $\\tilde{f}(z) = \\rho \\cdot \\pi(s_Z(z)) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z))$.

        For a perfectly conjugate model, $\\tilde{f}(z)$ would be constant.
        """
        rho, hrm_params = self.split_coords(params)
        s_z = self.pst_man.sufficient_statistic(z)
        s_rho = self.rho_emb.project(s_z)
        term1 = jnp.dot(rho, s_rho)
        lkl_params = self.hrm.likelihood_at(hrm_params, z)
        term2 = self.obs_man.log_partition_function(lkl_params)
        return term1 - term2

    def conjugation_error(
        self, key: Array, params: Array, n_samples: int = 100
    ) -> Array:
        """Compute conjugation error $\\mathrm{Var}[\\tilde{f}(z)]$ under the prior (zero if perfectly conjugate)."""
        p_params = self.prior_params(params)
        z_samples = self.pst_man.sample(key, p_params, n_samples)
        f_vals = jax.vmap(lambda z: self.reduced_learning_signal(params, z))(z_samples)
        return jnp.var(f_vals)

    def conjugation_metrics(
        self, key: Array, params: Array, n_samples: int = 100
    ) -> tuple[Array, Array, Array]:
        """Compute conjugation quality metrics (variance, std, $R^2$) under the prior.

        $R^2 = 1 - \\mathrm{Var}[\\tilde{f}] / \\mathrm{Var}[\\psi_X(\\eta(z))]$ measures how well
        the linear correction $\\rho \\cdot s(z)$ approximates $\\psi_X(\\eta(z))$:
        1 = perfect conjugation, 0 = no explanatory power, <0 = worse than constant.
        """
        rho, hrm_params = self.split_coords(params)
        p_params = self.prior_params(params)
        z_samples = self.pst_man.sample(key, p_params, n_samples)

        def compute_both(z: Array) -> tuple[Array, Array]:
            s_z = self.pst_man.sufficient_statistic(z)
            s_rho = self.rho_emb.project(s_z)
            lkl_params = self.hrm.likelihood_at(hrm_params, z)
            psi_x = self.obs_man.log_partition_function(lkl_params)
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
        hrm_params = self.generative_params(params)
        prior_p = self.prior_params(params)
        log_pz = self.pst_man.log_density(prior_p, z)
        lkl_params = self.hrm.likelihood_at(hrm_params, z)
        log_px_given_z = self.obs_man.log_density(lkl_params, x)
        return log_pz + log_px_given_z

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize full model parameters with zero conjugation correction."""
        rho = jnp.zeros(self.rho_man.dim)
        hrm_params = self.hrm.initialize(key, location, shape)
        return self.join_coords(rho, hrm_params)

    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize parameters using sample data for observable biases."""
        rho = jnp.zeros(self.rho_man.dim)
        hrm_params = self.hrm.initialize_from_sample(key, sample, location, shape)
        return self.join_coords(rho, hrm_params)
