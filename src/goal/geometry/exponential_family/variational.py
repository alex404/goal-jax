"""Variational conjugation for harmoniums.

This module provides the abstract base class for variational inference on harmoniums,
where the approximate posterior has the same structure as the exact conjugate
posterior, with learnable conjugation parameters.

Mathematical framework:

We reinterpret the harmonium as a directed generative model:

- The harmonium defines
  $p(x,z) \\propto \\exp(\\theta_X \\cdot s_X(x) + s_X(x) \\cdot \\Theta_{XZ} \\cdot s_Z(z) + \\theta_Z \\cdot s_Z(z))$
- We treat $\\theta_Z$ as defining an explicit prior $p(z; \\theta_Z)$
- The likelihood is $p(x|z)$ with natural parameters $\\theta_X + \\Theta_{XZ} \\cdot s_Z(z)$
- Note: this is NOT the same as the harmonium marginal
  $p(z) \\propto \\exp(\\theta_Z \\cdot s_Z(z) + \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z)))$

Approximate posterior:
$$q(z|x; \\rho) \\propto \\exp\\bigl((\\theta_Z + \\Theta_{XZ}^T s_X(x) - \\iota(\\rho)) \\cdot s_Z(z)\\bigr)$$

ELBO:
$$\\mathcal{L}(x) = \\mathbb{E}_q[\\log p(x|z)] - D_{\\mathrm{KL}}(q(z|x) \\| p(z))$$

Key insight: same form as true conjugate posterior, but $\\rho$ is learned rather than
computed from conjugation. The conjugation parameters and generative (harmonium)
parameters together form the complete model parameterization.

All ELBO and diagnostic methods depend only on the abstract EF interface
(``pst_man.sample()``, ``pst_man.to_mean()``, ``pst_man.relative_entropy()``,
``obs_man.log_partition_function()``, etc.). They work identically for simple
posteriors (Bernoullis, VonMises) and mixture posteriors (CompleteMixture)
because the mixture IS a Differentiable exponential family.

Classes:
    VariationalConjugated: Base class with ELBO computation and diagnostics
"""

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
    """Variational harmonium with learnable conjugation correction and ELBO computation.

    The full parameter space is Pair[rho_params, harmonium_params].

    The rho correction is applied via an embedding that maps Conjugation
    to Posterior space:

    $$q(z|x) \\propto \\exp\\bigl((\\theta_Z + \\Theta_{XZ}^T s_X(x) - \\iota(\\rho)) \\cdot s_Z(z)\\bigr)$$

    This generalizes:
    - Simple case: Conjugation = Posterior, embedding = Identity, so q = posterior - rho
    - Mixture case: Conjugation = BaseLatent, embedding maps to obs_bias of mixture

    ELBO decomposes as:
    $$\\mathcal{L}_\\beta(x) = \\mathbb{E}_q[\\log p(x|z)] - \\beta \\cdot D_{\\mathrm{KL}}(q \\| p)$$

    Attributes:
        hrm: The underlying Harmonium model
    """

    # Field

    hrm: Harmonium[Observable, Posterior]
    """The underlying harmonium model."""

    # Contract: subclasses provide the embedding

    @property
    @abstractmethod
    def rho_emb(self) -> LinearEmbedding[Conjugation, Posterior]:
        """Embedding from conjugation manifold to posterior manifold.

        Defines how rho correction is applied to the posterior parameters.
        """

    # Pair implementation

    @property
    def rho_man(self) -> Conjugation:
        """Manifold for rho correction parameters."""
        return self.rho_emb.sub_man

    @property
    @override
    def fst_man(self) -> Conjugation:
        """First component: conjugation parameter manifold."""
        return self.rho_man

    @property
    @override
    def snd_man(self) -> Harmonium[Observable, Posterior]:
        """Second component: the harmonium manifold."""
        return self.hrm

    # Convenience properties

    @property
    def obs_man(self) -> Observable:
        """Observable manifold of the underlying harmonium."""
        return self.hrm.obs_man

    @property
    def pst_man(self) -> Posterior:
        """Posterior manifold of the underlying harmonium."""
        return self.hrm.pst_man

    # Parameter access

    def conjugation_parameters(self, params: Array) -> Array:
        """Extract conjugation parameters (rho) from full parameters.

        Args:
            params: Full model parameters [rho, harmonium_params]

        Returns:
            Conjugation parameters rho
        """
        rho, _ = self.split_coords(params)
        return rho

    def generative_params(self, params: Array) -> Array:
        """Extract harmonium (generative model) parameters from full parameters.

        Args:
            params: Full model parameters [rho, harmonium_params]

        Returns:
            Harmonium parameters [theta_X, Theta_XZ, theta_Z]
        """
        _, hrm_params = self.split_coords(params)
        return hrm_params

    # Core methods

    def approximate_posterior_at(self, params: Array, x: Array) -> Array:
        """Compute approximate posterior natural parameters at observation x.

        $$\\eta_q = \\theta_Z + \\Theta_{XZ}^T s_X(x) - \\iota(\\rho)$$

        Args:
            params: Full model parameters [rho, harmonium_params]
            x: Observation

        Returns:
            Natural parameters of approximate posterior q(z|x)
        """
        rho, hrm_params = self.split_coords(params)
        exact_posterior = self.hrm.posterior_at(hrm_params, x)
        return self.rho_emb.translate(exact_posterior, -rho)

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Compute likelihood natural parameters at latent z.

        Delegates to the underlying harmonium.

        Args:
            params: Full model parameters
            z: Latent state

        Returns:
            Natural parameters of likelihood p(x|z)
        """
        _, hrm_params = self.split_coords(params)
        return self.hrm.likelihood_at(hrm_params, z)

    def prior_params(self, params: Array) -> Array:
        """Extract prior parameters theta_Z from the harmonium.

        These define the explicit prior $p(z; \\theta_Z)$, NOT the harmonium
        marginal which also includes the log-partition contribution.

        Args:
            params: Full model parameters

        Returns:
            Prior natural parameters theta_Z
        """
        _, hrm_params = self.split_coords(params)
        _, _, lat_params = self.hrm.split_coords(hrm_params)
        return lat_params

    # ELBO component: KL divergence (analytic)

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Compute KL divergence $D_{\\mathrm{KL}}(q(z|x) \\| p(z))$.

        Uses the Bregman divergence form:
        $$D_{\\mathrm{KL}} = \\psi(\\theta_p) - \\psi(\\theta_q)
          + (\\theta_q - \\theta_p) \\cdot \\nabla\\psi(\\theta_q)$$

        Args:
            params: Full model parameters
            x: Observation

        Returns:
            KL divergence (scalar)
        """
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)
        return self.pst_man.relative_entropy(q_params, p_params)

    # ELBO component: Expected log-likelihood (analytic + MC + score correction)

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

        1. Direct: $-\\mathbb{E}_q[\\partial\\psi_X/\\partial\\theta]$ — captured by ``mc_term``
        2. Score: $-\\mathbb{E}_q[\\psi_X \\cdot \\nabla_\\theta\\log q]$ — captured by ``reinforce``

        Args:
            key: JAX random key
            params: Full model parameters
            x: Observation
            n_samples: Number of MC samples

        Returns:
            Expected log-likelihood (scalar)
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

    # ELBO computation

    def elbo_at(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> Array:
        """Compute the ELBO at observation x.

        $$\\mathcal{L}_\\beta(x) = \\mathbb{E}_q[\\log p(x|z)]
          - \\beta \\cdot D_{\\mathrm{KL}}(q(z|x) \\| p(z))$$

        Args:
            key: JAX random key
            params: Full model parameters
            x: Observation
            n_samples: Number of MC samples
            kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

        Returns:
            ELBO value (scalar)
        """
        recon = self.elbo_reconstruction_term(key, params, x, n_samples)
        kl = self.elbo_divergence(params, x)
        return recon - kl_weight * kl

    def elbo_components(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> tuple[Array, Array, Array]:
        """Compute ELBO and its components for monitoring.

        Args:
            key: JAX random key
            params: Full model parameters
            x: Observation
            n_samples: Number of MC samples
            kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

        Returns:
            Tuple of (elbo, reconstruction_term, kl_term)
        """
        recon = self.elbo_reconstruction_term(key, params, x, n_samples)
        kl = self.elbo_divergence(params, x)
        elbo = recon - kl_weight * kl
        return elbo, recon, kl

    def mean_elbo(
        self,
        key: Array,
        params: Array,
        xs: Array,
        n_samples: int,
        kl_weight: float = 1.0,
    ) -> Array:
        """Compute mean ELBO over a batch of observations.

        Args:
            key: JAX random key
            params: Full model parameters
            xs: Batch of observations (shape: n_obs, obs_dim)
            n_samples: Number of MC samples per observation
            kl_weight: Weight on KL term (1.0 = standard VAE, <1.0 = beta-VAE)

        Returns:
            Mean ELBO (scalar)
        """
        batch_size = xs.shape[0]
        keys = jax.random.split(key, batch_size)

        elbos = jax.vmap(lambda k, x: self.elbo_at(k, params, x, n_samples, kl_weight))(
            keys, xs
        )

        return jnp.mean(elbos)

    # Conjugation parameter regression

    def regress_conjugation_parameters(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, Array, Array]:
        """Fit conjugation parameters by least-squares regression.

        Solves: $\\rho = \\arg\\min_\\rho \\sum_k (\\chi + \\rho \\cdot s_\\rho(z_k)
        - \\psi_X(\\theta_X + \\Theta \\cdot s_Z(z_k)))^2$

        Args:
            key: JAX random key
            params: Full model parameters
            n_samples: Number of samples from prior for regression

        Returns:
            Tuple of (rho, r_squared, chi) where chi is the intercept
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

    # Conjugation error (measures departure from exact conjugation)

    def reduced_learning_signal(self, params: Array, z: Array) -> Array:
        """Compute the reduced learning signal $\\tilde{f}(z)$.

        $$\\tilde{f}(z) = \\rho \\cdot \\pi(s_Z(z)) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z))$$

        where $\\pi$ projects sufficient statistics onto the conjugation manifold.
        For simple models, $\\pi = \\mathrm{id}$ (identity embedding).
        For hierarchical mixtures, $\\pi$ extracts the base latent component.

        For a perfectly conjugate model, $\\tilde{f}(z)$ would be constant.

        Args:
            params: Full model parameters
            z: Latent sample

        Returns:
            Scalar value of reduced learning signal
        """
        rho, hrm_params = self.split_coords(params)
        s_z = self.pst_man.sufficient_statistic(z)

        # Project sufficient statistics to conjugation manifold
        s_rho = self.rho_emb.project(s_z)

        # Linear term: rho . s_rho(z)
        term1 = jnp.dot(rho, s_rho)

        # Log partition of likelihood: psi_X(theta_X + Theta_XZ . s_Z(z))
        lkl_params = self.hrm.likelihood_at(hrm_params, z)
        term2 = self.obs_man.log_partition_function(lkl_params)

        return term1 - term2

    def conjugation_error(
        self, key: Array, params: Array, n_samples: int = 100
    ) -> Array:
        """Compute the conjugation error: $\\mathrm{Var}[\\tilde{f}(z)]$ under the prior.

        If perfectly conjugate, $\\tilde{f}(z) = \\mathrm{const}$ and variance = 0.

        Args:
            key: JAX random key
            params: Full model parameters
            n_samples: Number of samples for variance estimation

        Returns:
            Conjugation error (variance of f_tilde)
        """
        p_params = self.prior_params(params)
        z_samples = self.pst_man.sample(key, p_params, n_samples)

        f_vals = jax.vmap(lambda z: self.reduced_learning_signal(params, z))(z_samples)

        return jnp.var(f_vals)

    def conjugation_metrics(
        self, key: Array, params: Array, n_samples: int = 100
    ) -> tuple[Array, Array, Array]:
        """Compute conjugation quality metrics under the prior.

        Returns variance, standard deviation, and $R^2$ measuring how well
        the linear correction $\\rho \\cdot s(z)$ approximates $\\psi_X(\\eta(z))$.

        $$R^2 = 1 - \\mathrm{Var}[\\tilde{f}] / \\mathrm{Var}[\\psi_X(\\eta(z))]$$

        - $R^2 = 1$: Perfect conjugation ($\\psi_X$ is exactly linear in $s(z)$)
        - $R^2 = 0$: Linear correction explains none of the variance
        - $R^2 < 0$: Linear correction makes things worse

        Args:
            key: JAX random key
            params: Full model parameters
            n_samples: Number of samples for estimation

        Returns:
            Tuple of (variance, std, r_squared)
        """
        rho, hrm_params = self.split_coords(params)
        p_params = self.prior_params(params)
        z_samples = self.pst_man.sample(key, p_params, n_samples)

        # Compute both f_tilde and psi_X in a single vmap (shared likelihood_at)
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

        # R^2 = 1 - Var[residual] / Var[target]
        # Use threshold to avoid numerical instability
        variance_threshold = 1e-6
        raw_r2 = 1.0 - var_f / jnp.maximum(var_psi, variance_threshold)
        r_squared = jnp.where(
            var_psi > variance_threshold,
            jnp.clip(raw_r2, -10.0, 1.0),
            jnp.nan,
        )

        return var_f, std_f, r_squared

    # Sampling and log density

    def sample(self, key: Array, params: Array, n: int) -> Array:
        """Sample from the generative model p(x, z).

        Uses ancestral sampling: $z \\sim p(z; \\theta_Z)$, then $x \\sim p(x|z)$.
        Note: samples from the explicit prior $p(z; \\theta_Z)$, NOT the harmonium
        marginal $p(z) \\propto \\exp(\\theta_Z \\cdot s(z) + \\psi_X(\\eta(z)))$.
        Conjugation parameters don't affect the generative model.

        Args:
            key: JAX random key
            params: Full model parameters
            n: Number of samples

        Returns:
            Samples [x, z] of shape (n, obs_dim + lat_data_dim)
        """
        key1, key2 = jax.random.split(key)

        # Get prior parameters
        p_params = self.prior_params(params)

        # Sample z from prior
        z_samples = self.pst_man.sample(key1, p_params, n)

        # Sample x from likelihood for each z
        def sample_x_given_z(subkey: Array, z: Array) -> Array:
            lkl_params = self.likelihood_at(params, z)
            return self.obs_man.sample(subkey, lkl_params, 1)[0]

        x_keys = jax.random.split(key2, n)
        x_samples = jax.vmap(sample_x_given_z)(x_keys, z_samples)

        return jnp.concatenate([x_samples, z_samples], axis=-1)

    def log_density(self, params: Array, x: Array, z: Array) -> Array:
        """Compute joint log density log p(x, z) for the directed model.

        $$\\log p(x, z) = \\log p(z) + \\log p(x|z)$$

        Args:
            params: Full model parameters
            x: Observation
            z: Latent state

        Returns:
            Joint log density (scalar)
        """
        hrm_params = self.generative_params(params)
        prior_params = self.prior_params(params)
        log_pz = self.pst_man.log_density(prior_params, z)
        lkl_params = self.hrm.likelihood_at(hrm_params, z)
        log_px_given_z = self.obs_man.log_density(lkl_params, x)
        return log_pz + log_px_given_z

    # Initialization

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize full model parameters (conjugation + harmonium).

        Conjugation parameters are initialized to zeros (so q = exact posterior initially).
        Harmonium parameters use the standard harmonium initialization.

        Args:
            key: JAX random key
            location: Location parameter for initialization
            shape: Shape parameter for initialization

        Returns:
            Full model parameters [rho, harmonium_params]
        """
        rho = jnp.zeros(self.rho_man.dim)
        hrm_params = self.hrm.initialize(key, location, shape)
        return self.join_coords(rho, hrm_params)

    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize parameters using sample data for observable biases.

        Conjugation parameters are initialized to zeros.
        Harmonium uses sample-based initialization.

        Args:
            key: JAX random key
            sample: Sample data for initialization
            location: Location parameter
            shape: Shape parameter

        Returns:
            Full model parameters
        """
        rho = jnp.zeros(self.rho_man.dim)
        hrm_params = self.hrm.initialize_from_sample(key, sample, location, shape)
        return self.join_coords(rho, hrm_params)
