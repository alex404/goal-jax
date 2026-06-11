"""Variational conjugation: harmoniums with a learned conjugation correction and a conjugate-form recognition model.

A harmonium is exactly conjugated when the conjugation equation

$$\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) = \\rho_Z \\cdot \\mathbf s_Z(z) + \\psi_X(\\theta_X)$$

holds for some conjugation parameters $\\rho_Z$; the posterior then stays in the prior family with natural parameters affine in $\\mathbf s_X(x)$ (see :class:`~goal.geometry.exponential_family.harmonium.Conjugated`). When no exact $\\rho_Z$ exists or it cannot be computed, this module keeps the conjugate functional form and learns the correction instead: the recognition model is

$$q(z \\mid x) = p(z; \\hat\\theta_{Z \\mid X}(x)), \\qquad \\hat\\theta_{Z \\mid X}(x) = \\theta_Z - \\rho_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ},$$

with $\\rho_Z$ trained jointly with the generative parameters under the ELBO.

The central object of the module is the **conjugation residual**

$$r(z) = \\rho_Z \\cdot \\mathbf s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) + \\psi_X(\\theta_X),$$

the pointwise difference between the two sides of the conjugation equation: $r \\equiv 0$ iff the likelihood is exactly conjugate with conjugation parameters $\\rho_Z$. Substituting the exponential-family forms into the ELBO integrand $f(x, z) = \\log p(x, z) - \\log q(z \\mid x)$ cancels every term that couples $x$ and $z$, splitting it as

$$f(x, z) = c(x) + r(z), \\qquad \\mathcal{L}(x) = c(x) + \\mathbb{E}_{q(z \\mid x)}[r(Z)],$$

where $c(x)$ collects the $z$-independent terms --- the **standard form** of the ELBO. Every entry point of the module consumes the residual in one of three ways:

- **ELBO** --- :meth:`VariationalDifferentiable.elbo_at` evaluates $c(x)$ analytically and estimates $\\mathbb{E}_q[r]$ by Monte Carlo with a score-function gradient correction.
- **Regularizers** --- $\\mathrm{Var}_q[r]$ equals the amortization gap, so penalizing the variance of the residual under the prior (:meth:`VariationalConjugated.prior_conjugation_loss`) or the recognition model (:meth:`VariationalDifferentiable.recognition_conjugation_loss_at`) pushes the model toward exact conjugation.
- **Fitting and diagnostics** --- :func:`regress_conjugation_parameters` minimizes the sampled $\\mathrm{Var}_p[r]$ over $\\rho_Z$ in closed form by least squares, and :func:`conjugation_metrics` reports it as an $R^2$.

The classes ladder by latent capability, mirroring the analytic ``Conjugated`` hierarchy: :class:`VariationalConjugated` requires only :class:`Generative` latents, :class:`VariationalDifferentiable` adds everything closed-form log-partitions unlock, and :class:`VariationalSymmetric` specializes to a shared Posterior/Prior manifold.

$\\beta$-VAE-style KL warmup is the callers' responsibility: add $(1 - \\beta) \\cdot \\mathrm{KL}(q \\Vert p)$ via :meth:`VariationalDifferentiable.elbo_divergence`. The ELBO methods implement only the standard form.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Manifold
from ..manifold.combinators import Triple
from ..manifold.embedding import IdentityEmbedding, LinearEmbedding
from ..manifold.map import AffineMap
from .base import Differentiable, Generative
from .harmonium import Harmonium


class VariationalConjugated[
    Observable: Differentiable,
    Posterior: Generative,
    Prior: Generative,
    Conjugation: Manifold,
](
    Generative,
    Triple[Prior, AffineMap[Posterior, Observable], Conjugation],
    ABC,
):
    """Variational harmonium with parameter layout ``[prior_params, lkl_params, rho]``.

    Mathematically, the generative model is a directed harmonium

    $$p(x, z) = p(z; \\theta_Z) \\, p(x \\mid z; \\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)),$$

    and the recognition model retains the conjugate exponential-family form with the learned correction $\\rho_Z$ in place of the analytic conjugation parameters,

    $$q(z \\mid x) = p(z; \\hat\\theta_{Z \\mid X}(x)), \\qquad \\hat\\theta_{Z \\mid X}(x) = \\theta_Z - \\rho_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}.$$

    Training jointly optimizes $(\\theta_Z, \\theta_X, \\Theta_{XZ}, \\rho_Z)$ under the ELBO, moving the model toward a regime in which the conjugate-form posterior is accurate.

    The four type parameters separate four roles:

    - ``Observable`` --- observable family; :class:`Differentiable` because $\\psi_X$ appears in the residual.
    - ``Posterior`` --- family of $q(z \\mid x)$; the base class needs only :class:`Generative`.
    - ``Prior`` --- family of $p(z)$; may be a superset of ``Posterior``, connected via :attr:`pst_prr_emb`.
    - ``Conjugation`` --- storage manifold for the correction. By default a flat Prior-shaped $\\rho_Z$; :meth:`conjugation_parameters` maps the stored coordinates to Prior shape and is the extension point for structural completions and input-dependent corrections $\\rho_Z(x)$.

    The base class requires only sampling on ``Posterior`` and ``Prior``: it provides the model structure, the conjugation residual, joint sampling, and the prior conjugation regularizer. The ELBO and the recognition-side regularizer evaluate latent log-densities, so they live on :class:`VariationalDifferentiable`.
    """

    # Contract

    @property
    @abstractmethod
    def gen_hrm(self) -> Harmonium[Observable, Posterior]:
        """The underlying harmonium; supplies the likelihood $p(x \\mid z)$ and the observable/posterior manifolds."""

    @property
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[Posterior, Prior]:
        """Embedding of the posterior manifold into the prior manifold."""

    @property
    @abstractmethod
    def cnj_man(self) -> Conjugation:
        """The manifold in which $\\rho_Z$ is stored."""

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
        """The observable manifold."""
        return self.gen_hrm.obs_man

    @property
    def pst_man(self) -> Posterior:
        """The posterior manifold (family of $q(z \\mid x)$)."""
        return self.gen_hrm.pst_man

    @property
    def prr_man(self) -> Prior:
        """The prior manifold (family of $p(z; \\theta_Z)$)."""
        return self.pst_prr_emb.amb_man

    # Parameter access

    def prior_params(self, params: Array) -> Array:
        """Extract the prior natural parameters $\\theta_Z$."""
        prior, _, _ = self.split_coords(params)
        return prior

    def likelihood_function(self, params: Array) -> Array:
        """Extract the affine likelihood parameters $(\\theta_X, \\Theta_{XZ})$."""
        _, lkl, _ = self.split_coords(params)
        return lkl

    # Approximate conjugation

    def conjugation_parameters(self, params: Array, x: Array | None = None) -> Array:
        """Map the stored ``Conjugation`` coordinates to the Prior-shaped correction $\\rho_Z$, optionally as a function of the observation.

        The default treats the stored coordinates as $\\rho_Z$ itself and ignores ``x``. Subclasses override to apply a structural completion (e.g. mixture completion zero-pads the categorical interaction slots) or to evaluate a parametric input-dependent correction $\\rho_Z(x)$.
        """
        del x
        _, _, rho = self.split_coords(params)
        return rho

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Natural parameters of the likelihood $p(x \\mid z)$ at the given latent state."""
        _, lkl, _ = self.split_coords(params)
        mz = self.pst_man.sufficient_statistic(z)
        return self.gen_hrm.lkl_fun_man(lkl, mz)

    def approximate_posterior_at(self, params: Array, x: Array) -> Array:
        """Natural parameters of the recognition model $q(z \\mid x)$ at the given observation.

        Computes $\\hat\\theta_{Z \\mid X}(x) = \\theta_Z - \\rho_Z(x) + \\mathbf s_X(x) \\cdot \\Theta_{XZ}$ in Prior space and projects into ``Posterior`` via :attr:`pst_prr_emb`; in the symmetric case the projection is the identity and $\\rho_Z$ is input-independent.
        """
        prior, lkl, _ = self.split_coords(params)
        rho_full = self.conjugation_parameters(params, x)
        lat_eff = self.pst_prr_emb.project(prior - rho_full)
        obs_params, int_params = self.gen_hrm.lkl_fun_man.split_coords(lkl)
        hrm_params = self.gen_hrm.join_coords(obs_params, int_params, lat_eff)
        return self.gen_hrm.posterior_at(hrm_params, x)

    # Conjugation residual

    def conjugation_residual(
        self, params: Array, z: Array, x: Array | None = None
    ) -> Array:
        """Evaluate the conjugation residual $r(z) = \\rho_Z \\cdot \\mathbf s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) + \\psi_X(\\theta_X)$ at the given natural parameters.

        Mathematically, $r$ is the difference between the two sides of the conjugation equation $\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) = \\rho_Z \\cdot \\mathbf s_Z(z) + \\psi_X(\\theta_X)$, so $r \\equiv 0$ iff the likelihood is exactly conjugate with conjugation parameters $\\rho_Z$. It is the per-sample summand of the standard-form ELBO and the integrand of both conjugation regularizers (see module docstring).

        ``x`` is forwarded to :meth:`conjugation_parameters` for input-dependent corrections and ignored otherwise. In the asymmetric case both $\\rho_Z$ and $\\mathbf s_Z(z)$ are taken in Prior space, via :meth:`conjugation_parameters` and :attr:`pst_prr_emb` respectively.
        """
        _, lkl, _ = self.split_coords(params)
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(lkl)
        rho_full = self.conjugation_parameters(params, x)
        s_z = self.pst_man.sufficient_statistic(z)
        s_z_in_prior = self.pst_prr_emb.embed(s_z)
        rho_term = jnp.dot(rho_full, s_z_in_prior)
        lkl_params_at_z = self.gen_hrm.lkl_fun_man(lkl, s_z)
        psi_at_z = self.obs_man.log_partition_function(lkl_params_at_z)
        psi_at_bias = self.obs_man.log_partition_function(obs_params)
        return rho_term - psi_at_z + psi_at_bias

    # Conjugation regularizers

    def prior_conjugation_loss(
        self, key: Array, params: Array, n_samples: int
    ) -> Array:
        """Prior conjugation regularizer $\\mathcal{R}_p = \\mathrm{Var}_{p(z)}[r(Z)]$, estimated over prior samples.

        $x$-independent, matching the intuition that conjugation is a property of the likelihood and prior alone --- though the penalty may therefore act on regions of latent space that inference never visits.

        Gradient note: autodiff flows only through $r$ at fixed samples; no score-function correction is applied. The dropped score term $\\mathbb{E}_{p}[(r - \\bar r)^2 \\nabla \\log p]$ would contribute only to the $\\theta_Z$ gradient (the sampling distribution depends on nothing else, and $r$ itself does not depend on $\\theta_Z$), so omitting it means $\\mathcal{R}_p$ penalizes non-conjugation without pulling the prior around.
        """
        prior_p = self.prior_params(params)
        z_samples = jax.lax.stop_gradient(self.prr_man.sample(key, prior_p, n_samples))
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z))(z_samples)
        return jnp.var(r_vals)

    # Generative / ExponentialFamily contract on the joint $(x, z)$

    @property
    @override
    def data_dim(self) -> int:
        """Dimension of a joint $(x, z)$ datapoint --- delegates to the inner harmonium."""
        return self.gen_hrm.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Sufficient statistic of a joint $(x, z)$ datapoint --- delegates to the inner harmonium."""
        return self.gen_hrm.sufficient_statistic(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Log base measure of a joint $(x, z)$ datapoint --- delegates to the inner harmonium."""
        return self.gen_hrm.log_base_measure(x)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Sample the joint via ancestral sampling: $z \\sim p(z; \\theta_Z)$ from the explicit prior, then $x \\sim p(x \\mid z)$.

        The conjugation correction $\\rho_Z$ plays no role here: it parameterizes only the recognition model.
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
        prior = self.prior_params(params)
        log_pz = self.prr_man.log_density(prior, z)  # pyright: ignore[reportAttributeAccessIssue]
        lkl_params = self.likelihood_at(params, z)
        log_px_given_z = self.obs_man.log_density(lkl_params, x)
        return log_pz + log_px_given_z

    # Initialization

    @override
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

    @override
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


class VariationalDifferentiable[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
    Conjugation: Manifold,
](
    VariationalConjugated[Observable, Posterior, Prior, Conjugation],
    ABC,
):
    """Variational conjugation where ``Posterior`` and ``Prior`` have closed-form log-partition functions, enabling the standard-form ELBO.

    Mathematically, the ELBO integrand splits as $f(x, z) = c(x) + r(z)$ (see module docstring); with $\\psi_Z$ available, the $x$-dependent piece $c(x)$ is computed analytically (:meth:`conjugation_baseline`) and only the residual is left to Monte Carlo (:meth:`elbo_at`). The same machinery yields the closed-form KL between recognition model and prior (:meth:`elbo_divergence`) and the recognition-side conjugation regularizer (:meth:`recognition_conjugation_loss_at`).

    Mirrors :class:`DifferentiableConjugated` on the analytic side.
    """

    # ELBO

    def conjugation_baseline(self, params: Array, x: Array) -> Array:
        """The $z$-independent ELBO term $c(x) = \\mathbf s_X(x) \\cdot \\theta_X + \\psi_Z(\\hat\\theta_{Z \\mid X}(x)) - \\psi_Z(\\theta_Z) - \\psi_X(\\theta_X) + \\log h_X(x)$ at the given natural parameters.

        Mathematically, $c(x)$ is what remains of the ELBO integrand $f(x, z) = \\log p(x, z) - \\log q(z \\mid x)$ after the $z$-dependent terms are collected into the residual: substituting the exponential-family forms cancels the shared $\\mathbf s_Z \\cdot \\theta_Z$ and $\\mathbf s_X \\cdot \\Theta_{XZ} \\cdot \\mathbf s_Z$ terms, and the $\\pm \\psi_X(\\theta_X)$ convention makes the remainder $r(z)$ vanish exactly at conjugation.

        $c(x)$ is also the formula of :meth:`DifferentiableConjugated.log_observable_density` with the learned $\\rho_Z$ in place of the analytic conjugation parameters --- the log-marginal the model would have if conjugation were exact. The ELBO $\\mathcal{L}(x) = c(x) + \\mathbb{E}_q[r]$ accordingly collapses to $\\log p(x)$ when $r \\equiv 0$; in general $\\log p(x) = c(x) + \\mathbb{E}_q[r] + \\mathrm{KL}(q \\Vert p(z \\mid x))$.
        """
        prior_p, lkl, _ = self.split_coords(params)
        obs_p, _ = self.gen_hrm.lkl_fun_man.split_coords(lkl)
        q_params = self.approximate_posterior_at(params, x)
        q_in_prior = self.pst_prr_emb.embed(q_params)
        s_x = self.obs_man.sufficient_statistic(x)
        return (
            jnp.dot(s_x, obs_p)
            + self.prr_man.log_partition_function(q_in_prior)
            - self.prr_man.log_partition_function(prior_p)
            - self.obs_man.log_partition_function(obs_p)
            + self.obs_man.log_base_measure(x)
        )

    def elbo_at(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
    ) -> Array:
        """Estimate the standard-form ELBO $\\mathcal{L}(x) = c(x) + \\mathbb{E}_{q(z \\mid x)}[r(Z)]$: the baseline analytically, the residual term by Monte Carlo.

        The split absorbs $c(x)$ as an exact $x$-dependent baseline, so only the residual is sampled and only it drives the score-function correction,

        $$\\nabla \\mathcal{L}(x) = \\mathbb{E}_q[\\nabla \\log p(x, Z)] + \\mathbb{E}_q[(r - b) \\nabla \\log q],$$

        where the score term accounts for the dependence of the sampling distribution $q$ on the parameters and the sample-mean baseline $b$ reduces finite-sample variance. The surrogate ``c_x + direct + score - sg(score)`` equals $c(x)$ plus the plain MC average of $r$ in value, with three ``stop_gradient`` sites:

        1. **samples** --- the sampler may be non-differentiable (e.g. VonMises rejection); samples are evaluation points, not gradient carriers;
        2. **$r$ inside the score term** --- its direct gradient is already counted in ``direct``;
        3. **the final ``- sg(score)``** --- cancels the score term's value contribution, so the value stays a clean MC estimate while the gradient survives.
        """
        q_params = self.approximate_posterior_at(params, x)
        z_samples = jax.lax.stop_gradient(self.pst_man.sample(key, q_params, n_samples))

        c_x = self.conjugation_baseline(params, x)

        # E_q[r(Z)] via MC + score-function correction
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z, x))(z_samples)
        log_q_vals = jax.vmap(lambda z: self.pst_man.log_density(q_params, z))(z_samples)

        direct = jnp.mean(r_vals)
        r_sg = jax.lax.stop_gradient(r_vals)
        b = jnp.mean(r_sg)
        score = jnp.mean((r_sg - b) * log_q_vals)
        return c_x + direct + score - jax.lax.stop_gradient(score)

    def mean_elbo(
        self,
        key: Array,
        params: Array,
        xs: Array,
        n_samples: int,
    ) -> Array:
        """Mean ELBO over a batch of observations."""
        batch_size = xs.shape[0]
        keys = jax.random.split(key, batch_size)
        elbos = jax.vmap(lambda k, x: self.elbo_at(k, params, x, n_samples))(keys, xs)
        return jnp.mean(elbos)

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Closed-form $\\mathrm{KL}(q(z \\mid x) \\Vert p(z))$ at the given natural parameters, evaluated in Prior space via :attr:`pst_prr_emb`.

        Not used by :meth:`elbo_at` --- the standard form bundles the KL into $c(x)$ and $\\mathbb{E}_q[r]$. Exposed for $\\beta$-VAE-style warmup (callers add $(1 - \\beta) \\cdot \\mathrm{KL}$ to the ELBO) and as a diagnostic in its own right.
        """
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)
        q_in_prior = self.pst_prr_emb.embed(q_params)
        return self.prr_man.relative_entropy(q_in_prior, p_params)

    # Conjugation regularizers

    def recognition_conjugation_loss_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Recognition conjugation regularizer $\\mathcal{R}_q(x) = \\mathrm{Var}_{q(z \\mid x)}[r(Z)]$, estimated over recognition samples.

        Mathematically, $\\mathrm{Var}_q[r(Z)] = \\mathrm{Var}_q[\\log (q(Z \\mid x) / p(Z \\mid x))]$, so $\\mathcal{R}_q(x)$ is exactly the amortization gap at $x$: it vanishes iff the recognition model matches the true posterior. It focuses the penalty on the latents inference actually visits, and can reuse the samples drawn for the ELBO.

        Unlike :meth:`VariationalConjugated.prior_conjugation_loss`, the sampling distribution shares parameters with $r$, so the gradient requires both a direct piece through $r$ and a score-function correction through $q$ --- see :func:`_variance_with_score_correction`.
        """
        q_params = self.approximate_posterior_at(params, x)
        z_samples = jax.lax.stop_gradient(self.pst_man.sample(key, q_params, n_samples))
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z, x))(z_samples)
        log_q_vals = jax.vmap(lambda z: self.pst_man.log_density(q_params, z))(z_samples)
        return _variance_with_score_correction(r_vals, log_q_vals)

    def mean_recognition_conjugation_loss(
        self, key: Array, params: Array, xs: Array, n_samples: int
    ) -> Array:
        """Mean of :meth:`recognition_conjugation_loss_at` over a batch."""
        batch_size = xs.shape[0]
        keys = jax.random.split(key, batch_size)
        losses = jax.vmap(
            lambda k, x: self.recognition_conjugation_loss_at(k, params, x, n_samples)
        )(keys, xs)
        return jnp.mean(losses)


class VariationalSymmetric[
    Observable: Differentiable,
    Latent: Differentiable,
    Conjugation: Manifold,
](
    VariationalDifferentiable[Observable, Latent, Latent, Conjugation],
    ABC,
):
    """Variational conjugation in the symmetric case where Posterior and Prior share the same manifold (``pst_man == prr_man``).

    Mirrors :class:`SymmetricConjugated` on the analytic side: extends :class:`VariationalDifferentiable` and provides ``pst_prr_emb`` as an :class:`IdentityEmbedding` so subclasses only need to supply ``gen_hrm``, ``cnj_man``, and ``conjugation_parameters``.
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


def _variance_with_score_correction(r_vals: Array, log_q_vals: Array) -> Array:
    """Estimate $\\mathrm{Var}_q[r]$ as a loss whose autodiff gradient carries both pieces of $\\nabla \\mathrm{Var}_q[r] = 2\\,\\mathbb{E}_q[(r - \\bar r) \\nabla r] + \\mathbb{E}_q[(r - \\bar r)^2 \\nabla \\log q]$.

    Same ``direct + score - sg(score)`` surrogate as :meth:`VariationalDifferentiable.elbo_at`, with $g = (r - \\bar r)^2$ as the integrand:

    - ``direct = mean((r - mean(r))**2)`` --- biased-variance MC value; autodiff carries the first term.
    - ``score = mean((g_sg - b) * log_q)`` with sample-mean baseline $b$ --- autodiff carries the second.
    - ``- sg(score)`` cancels the score's value contribution, leaving the value equal to ``direct``.
    """
    r_sg = jax.lax.stop_gradient(r_vals)
    bar_r_sg = jnp.mean(r_sg)
    g_vals = (r_vals - jnp.mean(r_vals)) ** 2
    g_sg = (r_sg - bar_r_sg) ** 2
    b = jnp.mean(g_sg)
    direct = jnp.mean(g_vals)
    score = jnp.mean((g_sg - b) * log_q_vals)
    return direct + score - jax.lax.stop_gradient(score)


# --- Free functions: diagnostics, fitting helpers, downstream uses ---


def regress_conjugation_parameters[
    Observable: Differentiable,
    Posterior: Generative,
    Prior: Generative,
    Conjugation: Manifold,
](
    model: VariationalConjugated[Observable, Posterior, Prior, Conjugation],
    key: Array,
    params: Array,
    n_samples: int,
) -> tuple[Array, Array, Array, Array]:
    """Fit $\\rho$ by least squares against prior samples; the returned $\\rho$ minimizes the sampled $\\mathrm{Var}_p[r]$ in closed form.

    Solves $\\min_{\\chi, \\rho} \\sum_k (\\chi + \\iota(\\rho) \\cdot \\phi(\\mathbf s_Z(z_k)) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z_k)))^2$ over samples $z_k \\sim p(z)$, where $\\iota$ is :meth:`conjugation_parameters` and $\\phi$ is the embedding :attr:`pst_prr_emb`. Because the intercept $\\chi$ is free and the variance is shift-invariant, the minimum over $\\rho$ is exactly the minimum of the Monte-Carlo $\\mathcal{R}_p = \\mathrm{Var}_p[r]$ --- the regression and the prior regularizer agree on the optimum, and the fit is invariant to the $\\psi_X(\\theta_X)$ convention in :meth:`conjugation_residual`.

    The design matrix linearizes the (affine) dependence of $\\iota(\\rho) \\cdot \\phi(\\mathbf s_Z)$ on the stored $\\rho$ via :func:`jax.grad` at $\\rho = 0$, subtracting the $\\rho$-independent offset from the target; the same code therefore works whether ``conjugation_parameters`` is the identity or a richer analytic completion (e.g. mixture completion). Samples are stop-gradiented because the sampler may be non-differentiable; the gradients callers need flow through the regression targets $\\psi_X$.

    Returns ``(rho, r_squared, chi, residual_var)``: the fitted correction, the regression $R^2$, the intercept, and the sampled $\\mathrm{Var}[r]$ at the fit.

    A fitting/initialization heuristic rather than part of the variational objective, hence a free function.
    """
    prior_params, lkl, _ = model.split_coords(params)
    cnj_dim = model.cnj_man.dim
    zero_rho = jnp.zeros(cnj_dim)
    params_zero_rho = model.join_coords(prior_params, lkl, zero_rho)

    # The sampler may be non-differentiable (e.g. VonMises uses rejection
    # sampling via while_loop). The samples are just evaluation points for the
    # regression; the gradient we need flows through the regression targets
    # psi_X, which depend smoothly on (theta_X, Theta).
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
    Posterior: Generative,
    Prior: Generative,
    Conjugation: Manifold,
](
    model: VariationalConjugated[Observable, Posterior, Prior, Conjugation],
    key: Array,
    params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array]:
    """Compute conjugation quality metrics ``(var_f, std_f, r_squared)`` under the prior.

    $R^2 = 1 - \\mathrm{Var}_p[r] / \\mathrm{Var}_p[\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(Z))]$ measures how much of the variation of the log-partition the affine correction explains: 1 = exact conjugation, 0 = no better than a constant, < 0 = worse. Both variances are shift-invariant, so the $\\psi_X(\\theta_X)$ convention in :meth:`conjugation_residual` does not affect the metric. Returns ``NaN`` for $R^2$ when $\\mathrm{Var}_p[\\psi_X]$ is degenerate.
    """
    var_key, psi_key = jax.random.split(key)
    var_f = model.prior_conjugation_loss(var_key, params, n_samples)
    std_f = jnp.sqrt(var_f)

    prior_params, lkl, _ = model.split_coords(params)
    z_samples = model.prr_man.sample(psi_key, prior_params, n_samples)

    def psi_at(z: Array) -> Array:
        s_z = model.pst_man.sufficient_statistic(z)
        lkl_params = model.gen_hrm.lkl_fun_man(lkl, s_z)
        return model.obs_man.log_partition_function(lkl_params)

    psi_vals = jax.vmap(psi_at)(z_samples)
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
    Conjugation: Manifold,
](
    model: VariationalDifferentiable[Observable, Posterior, Prior, Conjugation],
    params: Array,
    x: Array,
) -> Array:
    """Reconstruct observable means via mean-field approximation: posterior mean stats through likelihood.

    Requires ``Posterior: Differentiable`` for the closed-form ``to_mean`` --- defined on :class:`VariationalDifferentiable` only.
    """
    q_params = model.approximate_posterior_at(params, x)
    z_mean_stats = model.pst_man.to_mean(q_params)
    _, lkl, _ = model.split_coords(params)
    lkl_natural = model.gen_hrm.lkl_fun_man(lkl, z_mean_stats)
    return model.obs_man.to_mean(lkl_natural)


def reconstruction_error[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
    Conjugation: Manifold,
](
    model: VariationalDifferentiable[Observable, Posterior, Prior, Conjugation],
    params: Array,
    xs: Array,
) -> Array:
    """Compute mean squared reconstruction error over a batch."""
    recons = jax.vmap(lambda x: reconstruct(model, params, x))(xs)
    return jnp.mean((xs - recons) ** 2)
