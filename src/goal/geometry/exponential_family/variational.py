"""Variational conjugation for harmoniums: approximate posterior with learnable conjugation correction and ELBO computation.

The approximate posterior has the same structure as the exact conjugate posterior, but with learned conjugation parameters ``\\rho`` replacing the analytically derived ones. The class mirrors the analytic ``Conjugated`` hierarchy: a ``Prior`` type parameter and a ``pst_prr_emb`` capture the (potentially asymmetric) Posterior \\subset Prior relationship, and the abstract ``conjugation_parameters(params)`` returns the full Prior-shape conjugation (by peeling the stored ``\\rho`` from ``params`` and applying any model-specific analytic completion). Hierarchical and mixture subclasses override this method to inline the same analytic completion patterns used on the analytic side (zero-padding for chains, mixture completion for mixtures).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.combinators import Triple
from ..manifold.embedding import IdentityEmbedding, LinearEmbedding
from ..manifold.map import AffineMap
from .base import Differentiable, ExponentialFamily
from .harmonium import Harmonium


class VariationalConjugated[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
    Conjugation: ExponentialFamily,
](
    Triple[Prior, AffineMap[Posterior, Observable], Conjugation],
    ABC,
):
    """Variational harmonium with parameter layout ``[prior_params, lkl_fun, rho]``.

    Reinterprets an underlying harmonium (``gen_hrm``) as a directed generative
    model: the harmonium's observable + interaction blocks supply the likelihood
    $p(x|z)$, and a separately-stored prior $p(z; \\theta_Z)$ lives in ``Prior``
    space (which may be a superset of ``Posterior``). The variational layer adds
    learned conjugation parameters $\\rho$ that replace the analytically-derived
    ones in the conjugate posterior

    $$q(z|x; \\rho) \\propto \\exp\\bigl(\\pi(\\theta_Z - \\iota(\\rho)) \\cdot s_Z(z) + \\Theta_{XZ}^\\top s_X(x) \\cdot s_Z(z)\\bigr),$$

    where $\\iota(\\rho)$ is the full Prior-shape conjugation derived from the
    stored ``rho`` via :meth:`conjugation_parameters` (analog of
    :class:`Conjugated.conjugation_parameters` on the analytic side), and $\\pi$
    is :meth:`pst_prr_emb.project`. In the symmetric case (Posterior=Prior),
    $\\pi$ is identity and the formula collapses to
    $(\\theta_Z + \\Theta_{XZ}^\\top s_X(x) - \\iota(\\rho)) \\cdot s_Z(z)$.

    The four type parameters separate four roles that the original two-parameter
    design conflated:

    - ``Observable`` --- observable variable family
    - ``Posterior`` --- family of $q(z|x)$; what the posterior formula returns
    - ``Prior`` --- family of $p(z)$; what the stored prior lives in
    - ``Conjugation`` --- storage space for the learned $\\rho$; may be a
      sub-manifold of Prior when only some prior components need variational
      approximation (e.g. for hierarchical/mixture priors, only the
      observable-adjacent component lacks a closed form)

    The :meth:`conjugation_parameters` abstract method bridges ``Conjugation``
    and ``Prior``: subclasses peel the stored $\\rho$ from ``params`` and
    apply whatever analytic completion their graphical structure calls for.
    Unlike analytic ``Conjugated.conjugation_parameters(lkl_params)`` which
    *computes* $\\rho$ from likelihood, the variational $\\rho$ is *stored* ---
    so the natural argument here is the full ``params`` rather than a separate
    ``lkl`` / ``rho`` pair.
    """

    # Contract

    @property
    @abstractmethod
    def gen_hrm(self) -> Harmonium[Observable, Posterior]:
        """Underlying generative harmonium --- supplies the likelihood structure and observable/posterior manifolds."""

    @property
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[Posterior, Prior]:
        """Embedding of the posterior manifold into the prior manifold (analogous to :attr:`Conjugated.pst_prr_emb`)."""

    @property
    @abstractmethod
    def cnj_man(self) -> Conjugation:
        """Manifold for the variationally-learned $\\rho$ (the third Triple slot)."""

    # Triple slots

    @property
    @override
    def fst_man(self) -> Prior:
        return self.prr_man

    @property
    @override
    def snd_man(self) -> AffineMap[Posterior, Observable]:
        return self.gen_hrm.lkl_fun_man

    @property
    @override
    def trd_man(self) -> Conjugation:
        return self.cnj_man

    # Manifold access

    @property
    def obs_man(self) -> Observable:
        """Observable manifold of the underlying harmonium."""
        return self.gen_hrm.obs_man

    @property
    def pst_man(self) -> Posterior:
        """Posterior manifold (family of $q(z|x)$)."""
        return self.gen_hrm.pst_man

    @property
    def prr_man(self) -> Prior:
        """Prior manifold (family of stored $p(z)$)."""
        return self.pst_prr_emb.amb_man

    # Parameter access

    def prior_params(self, params: Array) -> Array:
        """Extract prior parameters $\\theta_Z$ defining $p(z; \\theta_Z)$ (in ``Prior`` space)."""
        prior, _, _ = self.split_coords(params)
        return prior

    def likelihood_function(self, params: Array) -> Array:
        """Extract the stored likelihood affine map $\\eta \\mapsto \\theta_X + \\Theta_{XZ} \\cdot \\eta$."""
        _, lkl, _ = self.split_coords(params)
        return lkl

    # Approximate conjugation

    def conjugation_parameters(self, params: Array) -> Array:
        """Return the full Prior-shape conjugation $\\iota(\\rho)$ for the given model parameters.

        Default: ``Conjugation == Prior`` --- the stored $\\rho$ already has
        the right shape, so just peel it out of ``params`` and return it.
        Subclasses with $\\text{Conjugation} \\subset \\text{Prior}$
        (hierarchical chains, mixtures) override to apply the analytic
        completion that lifts the learned $\\rho$ into the full Prior shape
        (zero-padding into unconnected slots for hierarchical chains,
        log-partition differences for mixture categorical slots, etc. ---
        mirroring how the analytic-side ``Conjugated`` subclasses construct
        their full-Prior conjugation from likelihood blocks).

        Unlike the analytic ``Conjugated.conjugation_parameters(lkl_params)``,
        this is *not* a function of likelihood alone --- the variational $\\rho$
        is stored, not computed --- so the natural argument is the full
        ``params`` from which the method peels what it needs.
        """
        _, _, rho = self.split_coords(params)
        return rho

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Compute likelihood natural parameters $p(x|z)$ at latent $z$."""
        _, lkl, _ = self.split_coords(params)
        mz = self.pst_man.sufficient_statistic(z)
        return self.gen_hrm.lkl_fun_man(lkl, mz)

    def approximate_posterior_at(self, params: Array, x: Array) -> Array:
        """Compute approximate posterior natural parameters $q(z|x)$ (in ``Posterior`` space).

        Computes the corrected latent bias $\\pi(\\theta_Z - \\iota(\\rho))$ in
        ``Posterior`` space (where $\\iota = $ :meth:`conjugation_parameters` and
        $\\pi = $ :meth:`pst_prr_emb.project`), then applies the standard
        harmonium posterior formula via ``gen_hrm.posterior_at``.
        """
        prior, lkl, _ = self.split_coords(params)
        rho_full = self.conjugation_parameters(params)
        lat_eff = self.pst_prr_emb.project(prior - rho_full)
        obs_params, int_params = self.gen_hrm.lkl_fun_man.split_coords(lkl)
        hrm_params = self.gen_hrm.join_coords(obs_params, int_params, lat_eff)
        return self.gen_hrm.posterior_at(hrm_params, x)

    # ELBO

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Compute $D_{\\mathrm{KL}}(q(z|x) \\| p(z))$.

        $q$ lives in ``Posterior`` space, $p$ in ``Prior`` space; the KL is
        evaluated in ``Prior`` space by embedding $q$ via :meth:`pst_prr_emb.embed`.
        In the symmetric case this is identity and reduces to the natural
        same-space KL.
        """
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)
        q_in_prior = self.pst_prr_emb.embed(q_params)
        return self.prr_man.relative_entropy(q_in_prior, p_params)

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
        """Compute the reduced learning signal $\\tilde{f}(z) = \\iota(\\rho) \\cdot \\phi(s_Z(z)) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z))$.

        The score-function gradient of $-\\mathbb{E}_q[\\psi_X(\\eta(z))]$ admits an arbitrary control variate; the full Prior-shape conjugation $\\iota(\\rho)$ inner-producted with the embedded sufficient statistic $\\phi(s_Z(z))$ is the optimal linear-in-$s_Z$ choice, and $\\tilde{f}$ is what remains of the learning signal after subtracting it. The per-$x$ KL gap is exactly $\\log \\mathbb{E}_q[e^{\\tilde{f}}] - \\mathbb{E}_q[\\tilde{f}]$, so $\\tilde{f}$ constant $\\iff$ zero amortization gap; its variance under the prior is used as a training regularizer.
        """
        _, lkl, _ = self.split_coords(params)
        rho_full = self.conjugation_parameters(params)
        s_z = self.pst_man.sufficient_statistic(z)
        s_z_in_prior = self.pst_prr_emb.embed(s_z)
        term1 = jnp.dot(rho_full, s_z_in_prior)
        lkl_params = self.gen_hrm.lkl_fun_man(lkl, s_z)
        term2 = self.obs_man.log_partition_function(lkl_params)
        return term1 - term2

    # Generative

    def sample(self, key: Array, params: Array, n: int) -> Array:
        """Sample from the generative model $p(x, z)$ via ancestral sampling.

        Samples from the explicit prior $p(z; \\theta_Z)$ in ``Prior`` space, NOT the harmonium marginal. Conjugation parameters don't affect the generative model.
        """
        key1, key2 = jax.random.split(key)
        p_params = self.prior_params(params)
        z_samples = self.prr_man.sample(key1, p_params, n)

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
        log_pz = self.prr_man.log_density(prior, z)
        mz = self.pst_man.sufficient_statistic(z)
        lkl_params = self.gen_hrm.lkl_fun_man(lkl, mz)
        log_px_given_z = self.obs_man.log_density(lkl_params, x)
        return log_pz + log_px_given_z

    # Initialization

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize full model parameters with zero conjugation correction.

        Initializes the underlying harmonium, takes its observable + interaction
        as the likelihood and embeds its latent slot into ``Prior`` space to seed
        ``prior_params``. ``rho`` is zeroed.
        """
        rho = jnp.zeros(self.cnj_man.dim)
        hrm_params = self.gen_hrm.initialize(key, location, shape)
        obs_params, int_params, lat_params = self.gen_hrm.split_coords(hrm_params)
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        prior_params = self.pst_prr_emb.embed(lat_params)
        return self.join_coords(prior_params, lkl_params, rho)

    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize parameters using sample data for observable biases."""
        rho = jnp.zeros(self.cnj_man.dim)
        hrm_params = self.gen_hrm.initialize_from_sample(key, sample, location, shape)
        obs_params, int_params, lat_params = self.gen_hrm.split_coords(hrm_params)
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        prior_params = self.pst_prr_emb.embed(lat_params)
        return self.join_coords(prior_params, lkl_params, rho)


class SymmetricVariationalConjugated[
    Observable: Differentiable,
    Latent: Differentiable,
    Conjugation: ExponentialFamily,
](
    VariationalConjugated[Observable, Latent, Latent, Conjugation],
    ABC,
):
    """Variational conjugation in the symmetric case where Posterior and Prior share the same manifold (``pst_man == prr_man``).

    Mirrors :class:`SymmetricConjugated` on the analytic side: provides
    ``pst_prr_emb`` as an :class:`IdentityEmbedding` so subclasses only need to
    supply ``gen_hrm``, ``cnj_man``, and ``conjugation_parameters``.
    """

    # Contract

    @property
    @abstractmethod
    def lat_man(self) -> Latent:
        """The shared posterior/prior manifold."""

    # Overrides

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Latent]:
        return IdentityEmbedding(self.lat_man)


# --- Free functions: diagnostics, fitting helpers, downstream uses ---


def regress_conjugation_parameters[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
    Conjugation: ExponentialFamily,
](
    model: VariationalConjugated[Observable, Posterior, Prior, Conjugation],
    key: Array,
    params: Array,
    n_samples: int,
) -> tuple[Array, Array, Array, Array]:
    """Fit the variationally-learned $\\rho$ by least-squares regression against samples from the prior.

    Solves $\\rho = \\arg\\min_\\rho \\sum_k (\\chi + \\iota(\\rho) \\cdot \\phi(s_Z(z_k)) - \\psi_X(\\theta_X + \\Theta \\cdot s_Z(z_k)))^2$, where $\\iota = $ :meth:`conjugation_parameters` (applied to a synthetic ``params`` with the candidate $\\rho$ swapped in) and $\\phi = $ :meth:`pst_prr_emb.embed`. The design matrix is obtained by linearizing the (linear) dependence of $\\iota(\\rho)\\cdot\\phi(s_Z)$ on the stored $\\rho$ via :func:`jax.grad`, which lets the same regression code work whether ``conjugation_parameters`` is identity (simple symmetric case) or a richer analytic completion (mixture/hierarchical case). The constant offset from the rho-independent part of $\\iota(\\rho)\\cdot\\phi(s_Z)$ (e.g. the analytic categorical contribution in a mixture) is subtracted from the regression target.

    Returns ``(rho, r_squared, chi, residual_var)`` where ``chi`` is the intercept and ``residual_var`` is the conjugation error $\\mathrm{Var}[\\tilde{f}]$ estimated from the regression samples.

    A pragmatic fitting/initialization heuristic --- theoretically a black sheep in the variational conjugation family, hence its free-function status.
    """
    prior_params, lkl, _ = model.split_coords(params)
    cnj_dim = model.cnj_man.dim
    zero_rho = jnp.zeros(cnj_dim)
    params_zero_rho = model.join_coords(prior_params, lkl, zero_rho)

    # [SG-3] The sampler may be non-differentiable (e.g. VonMises uses
    # rejection sampling via while_loop, which JAX cannot reverse-diff).
    # The samples are just evaluation points for the regression; the
    # gradient we need flows through the regression targets psi_X, which
    # depend smoothly on (theta_X, Theta).
    z_samples = jax.lax.stop_gradient(
        model.prr_man.sample(key, prior_params, n_samples)
    )

    def design_and_target(z: Array) -> tuple[Array, Array, Array]:
        s_z = model.pst_man.sufficient_statistic(z)
        s_z_in_prior = model.pst_prr_emb.embed(s_z)

        def rho_dot_s(r: Array) -> Array:
            trial_params = model.join_coords(prior_params, lkl, r)
            return jnp.dot(model.conjugation_parameters(trial_params), s_z_in_prior)

        # Linear coefficient: gradient wrt stored rho at zero.
        s_proj = jax.grad(rho_dot_s)(zero_rho)
        # Constant offset (rho-independent contribution of iota(0).phi(s_z)).
        offset = jnp.dot(model.conjugation_parameters(params_zero_rho), s_z_in_prior)

        lkl_params = model.gen_hrm.lkl_fun_man(lkl, s_z)
        psi_x = model.obs_man.log_partition_function(lkl_params)

        return s_proj, psi_x, offset

    s_proj_all, psi_all, offset_all = jax.vmap(design_and_target)(z_samples)
    adj_psi = psi_all - offset_all

    design = jnp.concatenate([jnp.ones((n_samples, 1)), s_proj_all], axis=1)
    coeffs = jnp.linalg.lstsq(design, adj_psi, rcond=None)[0]
    chi = coeffs[0]
    rho = coeffs[1:]

    residuals = adj_psi - design @ coeffs
    ss_res = jnp.sum(residuals**2)
    ss_tot = jnp.sum((adj_psi - jnp.mean(adj_psi)) ** 2)
    r_squared = 1.0 - ss_res / jnp.maximum(ss_tot, 1e-8)

    return rho, r_squared, chi, jnp.var(residuals)


def conjugation_metrics[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
    Conjugation: ExponentialFamily,
](
    model: VariationalConjugated[Observable, Posterior, Prior, Conjugation],
    key: Array,
    params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array]:
    """Compute conjugation quality metrics (variance, std, $R^2$) under the prior.

    $R^2 = 1 - \\mathrm{Var}[\\tilde{f}] / \\mathrm{Var}[\\psi_X(\\eta(z))]$ measures how well the linear correction $\\iota(\\rho) \\cdot \\phi(s_Z)$ approximates $\\psi_X(\\eta(z))$: 1 = perfect conjugation, 0 = no explanatory power, <0 = worse than constant.
    """
    prior_params, lkl, _ = model.split_coords(params)
    rho_full = model.conjugation_parameters(params)
    z_samples = model.prr_man.sample(key, prior_params, n_samples)

    def compute_both(z: Array) -> tuple[Array, Array]:
        s_z = model.pst_man.sufficient_statistic(z)
        s_z_in_prior = model.pst_prr_emb.embed(s_z)
        lkl_params = model.gen_hrm.lkl_fun_man(lkl, s_z)
        psi_x = model.obs_man.log_partition_function(lkl_params)
        f_tilde = jnp.dot(rho_full, s_z_in_prior) - psi_x
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
    Prior: Differentiable,
    Conjugation: ExponentialFamily,
](
    model: VariationalConjugated[Observable, Posterior, Prior, Conjugation],
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
    Prior: Differentiable,
    Conjugation: ExponentialFamily,
](
    model: VariationalConjugated[Observable, Posterior, Prior, Conjugation],
    params: Array,
    xs: Array,
) -> Array:
    """Compute mean squared reconstruction error over a batch."""
    recons = jax.vmap(lambda x: reconstruct(model, params, x))(xs)
    return jnp.mean((xs - recons) ** 2)
