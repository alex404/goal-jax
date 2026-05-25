"""Variational conjugation for harmoniums.

Implements the standard-form ELBO

$$\\mathcal{L}(x) = \\mathbb{E}_{q(z\\mid x)}[\\log p(x, Z) - \\log q(Z \\mid x)] = c(x) + \\mathbb{E}_q[r(Z)],$$

where the **conjugation residual** $r(z) = \\rho_Z \\cdot s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ}\\cdot s_Z(z)) + \\psi_X(\\theta_X)$ is the difference between the LHS and RHS of the conjugation equation: it vanishes pointwise iff the likelihood and prior are exactly conjugate. The residual is the central object of the module --- it drives the ELBO via $\\mathbb{E}_q[r]$ and the conjugation regularizers $\\mathcal{R}_p$ / $\\mathcal{R}_q$ via $\\mathrm{Var}[r]$.

The $\\beta$-VAE-style KL warmup is handled by callers, who add $(1-\\beta)\\cdot\\mathrm{KL}(q \\Vert p)$ to the ELBO externally via :meth:`elbo_divergence`. Keeping $\\beta$ out of :meth:`elbo_at` lets the residual machinery extend without alteration to hierarchical models, where no closed-form KL exists between joint $q$ and joint $p$ and the closed-form-KL lever the recon+KL form depends on disappears.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Manifold
from ..manifold.combinators import Pair, Triple
from ..manifold.embedding import IdentityEmbedding, LinearEmbedding
from ..manifold.map import AffineMap, Map
from .base import Differentiable, Generative
from .graphical import ObservableEmbedding
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
    """Variational harmonium with parameter layout ``[prior_params, lkl_fun, rho]``.

    Models a directed generative process

    $$p(x, z) = p(z; \\theta_Z) \\, p(x \\mid z; \\theta_X, \\Theta_{XZ}),$$

    with an exponential-family prior $p(z; \\theta_Z) \\in $ ``Prior`` and a likelihood whose natural parameters are affine in the latent sufficient statistic,

    $$\\theta_{X \\mid Z}(z) = \\theta_X + \\Theta_{XZ} \\cdot s_Z(z).$$

    When exact conjugation is unavailable, we approximate the posterior by retaining the same exponential-family form as the prior but replacing the analytic conjugation correction with a learned parameter $\\rho_Z$:

    $$q(z \\mid x) = p(z; \\hat\\theta_{Z \\mid X}(x)), \\qquad \\hat\\theta_{Z \\mid X}(x) = \\theta_Z + s_X(x) \\cdot \\Theta_{XZ} - \\rho_Z.$$

    By default $\\rho_Z$ is input-independent: a single Bernoullis/Gaussian-shaped vector stored in the ``Conjugation`` slot of the Triple. The :meth:`conjugation_parameters` API accepts an optional ``x`` argument so that subclasses can implement input-dependent corrections $\\rho_Z(x) = f(\\hat\\theta_Y(x), \\Theta_{YZ}; \\phi)$ --- in that case the ``Conjugation`` slot stores the parametric map's parameters (e.g. MLP weights and biases) rather than a flat $\\rho_Z$.

    Training jointly optimizes $(\\theta_Z, \\theta_X, \\Theta_{XZ}, \\rho_Z)$ under the ELBO so that the model moves toward a regime in which the conjugate-form posterior is accurate.

    The base :meth:`elbo_at` is a Monte-Carlo + score-function estimator that requires only :class:`Generative` capabilities on ``Posterior`` and ``Prior`` --- enough to drive the joint hierarchical case where the prior's log-partition is not closed-form. Subclasses where ``Posterior`` and ``Prior`` are :class:`Differentiable` (see :class:`VariationalDifferentiable`) override with the lower-variance $c(x) + \\mathbb{E}_q[r]$ form and gain :meth:`conjugation_baseline` / :meth:`elbo_divergence`.

    The four type parameters separate four roles:

    - ``Observable`` --- observable variable family
    - ``Posterior`` --- family of $q(z \\mid x)$ (sample, sufficient_statistic required; :class:`Generative` minimum)
    - ``Prior`` --- family of $p(z)$; may be a superset of ``Posterior``. :class:`Generative` minimum
    - ``Conjugation`` --- storage manifold for the variational correction. The default :meth:`conjugation_parameters` treats this slot as a flat $\\rho_Z$ in ``Prior``-shape; subclasses with a richer structure (mixture completion, parametric input-dependent maps) override to apply the appropriate completion or evaluation.

    :meth:`conjugation_parameters` peels $\\rho_Z$ from ``params`` and applies the analytic completion appropriate to the graphical structure (zero-padding for chains, mixture completion for mixtures, MLP evaluation for input-dependent corrections). In the symmetric flat case (Posterior = Prior = Conjugation) it returns $\\rho_Z$ unchanged.
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
        """Return the conjugation correction in Prior shape.

        Default: $\\rho_Z$ already lives in ``Prior``, so return it unchanged and ignore ``x``. Subclasses with richer structure override:

        - ``Conjugation`` a proper sub-manifold of ``Prior`` (hierarchical chains, mixtures): apply the analytic structural completion (zero-padding for chains, mixture completion for mixtures).
        - Input-dependent corrections (parametric $\\rho_Z(x) = f(\\hat\\theta_Y(x), \\Theta_{YZ}; \\phi)$): use ``x`` to evaluate the parametric map. ``x`` is the observation that defines the conditional; subclasses determine how to thread it (e.g. precompute $\\hat\\theta_Y(x)$ for an MLP input).
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
        """Natural parameters of the approximate posterior $q(z \\mid x)$ at the given observation.

        Computes $\\hat\\theta_{Z \\mid X}(x) = \\theta_Z + s_X(x) \\cdot \\Theta_{XZ} - \\rho_Z(x)$ and projects into ``Posterior`` via :attr:`pst_prr_emb`; in the symmetric flat case the projection is identity and $\\rho_Z(x) = \\rho_Z$ is input-independent.
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
        """Conjugation residual $r(z) = \\rho_Z(x) \\cdot s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ}\\cdot s_Z(z)) + \\psi_X(\\theta_X)$.

        Mathematically, $r$ is the difference between the LHS and RHS of the conjugation equation $\\rho_Z \\cdot s_Z(z) + \\psi_X(\\theta_X) = \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z))$, so $r \\equiv 0$ iff the likelihood is exactly conjugate to the prior. Used as the per-sample summand in the standard-form ELBO ($\\mathcal{L}(x) = c(x) + \\mathbb{E}_q[r(Z)]$) and as the integrand of the conjugation regularizers $\\mathrm{Var}_q[r]$ and $\\mathrm{Var}_p[r]$.

        ``x`` is forwarded to :meth:`conjugation_parameters` and is ignored in the input-independent case. Generalizes correctly through :meth:`conjugation_parameters` (which embeds $\\rho_Z$ into Prior shape for asymmetric cases) and :attr:`pst_prr_emb` (which embeds the posterior sufficient statistic into Prior space).
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

    # ELBO

    def elbo_at(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
    ) -> Array:
        """ELBO $\\mathcal{L}(x) = \\mathbb{E}_{q(z \\mid x)}[\\log p(x, Z) - \\log q(Z \\mid x)]$ via MC + score-function correction.

        Estimates the full integrand $f(z) = \\log p(x \\mid z) + \\log p(z) - \\log q(z \\mid x)$ per recognition sample, then applies a score-function correction $\\mathbb{E}_q[(f - b) \\nabla \\log q]$ with sample-mean baseline $b$. Works without any closed-form log-partition assumption on ``Prior`` --- the formulation that drives the hierarchical case (where the joint $\\psi$ is intractable) and the bivariate one as a special case.

        Subclasses where ``Posterior`` and ``Prior`` are :class:`Differentiable` (see :class:`VariationalDifferentiable`) override with the lower-variance $c(x) + \\mathbb{E}_q[r(Z)]$ form, which Rao-Blackwellizes the $x$-dependent piece via the analytic posterior log-partition.

        Three ``stop_gradient`` calls:

        1. **On samples** --- the recognition sampler may be non-differentiable (e.g. VonMises rejection); samples are evaluation points, not gradient carriers.
        2. **On $f$ inside the score correction** --- its direct gradient is already in ``direct``, so letting autodiff flow through $f$ here would double-count it.
        3. **On the final ``score`` value subtraction** --- keeps the surrogate value-correct as an MC estimate while preserving the gradient contribution.

        $\\beta$-VAE-style KL warmup is the caller's responsibility (see :meth:`VariationalDifferentiable.elbo_divergence` / :meth:`VariationalHierarchical.elbo_divergence`).
        """
        q_params = self.approximate_posterior_at(params, x)
        z_samples = jax.lax.stop_gradient(self.pst_man.sample(key, q_params, n_samples))
        p_params = self.prior_params(params)

        def f_theta(z: Array) -> tuple[Array, Array]:
            """log p(x, z) - log q(z | x), returned alongside log q for the score term."""
            lkl_natural = self.likelihood_at(params, z)
            log_p_x_given_z = self.obs_man.log_density(lkl_natural, x)
            log_p_z = self.prr_man.log_density(p_params, z)  # pyright: ignore[reportAttributeAccessIssue]
            log_q_z = self.pst_man.log_density(q_params, z)  # pyright: ignore[reportAttributeAccessIssue]
            return log_p_x_given_z + log_p_z - log_q_z, log_q_z

        f_vals, log_q_vals = jax.vmap(f_theta)(z_samples)

        direct = jnp.mean(f_vals)
        f_sg = jax.lax.stop_gradient(f_vals)
        baseline = jnp.mean(f_sg)
        score = jnp.mean((f_sg - baseline) * log_q_vals)
        return direct + score - jax.lax.stop_gradient(score)

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

    # Conjugation regularizers

    def prior_conjugation_loss(
        self, key: Array, params: Array, n_samples: int
    ) -> Array:
        """Prior-based regularizer $\\mathcal{R}_p = \\mathrm{Var}_{p_Z}[r(Z)]$.

        $x$-independent and arguably more consistent with the intuition that conjugation is a property of the likelihood and prior. Can enforce conjugacy in regions of latent space that inference never visits.

        Gradient note: autodiff flows only through $r$ (the samples are sg'd, and no score-function correction is applied). The dropped score term $\\mathbb{E}_{p_Z}[(r-\\bar r)^2 \\nabla \\log p_Z]$ would have contributed only to the $\\theta_Z$ gradient --- but $r$ itself does not depend on $\\theta_Z$, so the regularizer's effect on $\\theta_Z$ is zero either way under a "$\\mathcal{R}_p$ should penalize non-conjugation, not pull the prior around" reading. Side-stepping the score-function machinery is intentional and matches manuscript §4.2's positioning of $\\mathcal{R}_p$.
        """
        prior_p = self.prior_params(params)
        z_samples = jax.lax.stop_gradient(self.prr_man.sample(key, prior_p, n_samples))
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z))(z_samples)
        return jnp.var(r_vals)

    def recognition_conjugation_loss_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Recognition-based regularizer $\\mathcal{R}_q(x) = \\mathrm{Var}_{q(Z\\mid X=x)}[r(Z)]$.

        Focuses on the latents inference actually visits. Directly connected to the amortization gap, since $\\mathrm{Var}_q[r(Z)] = \\mathrm{Var}_q[\\log q(Z\\mid x) / p(Z\\mid x)]$. Estimated with the same recognition samples one would draw for the ELBO.

        The sampling distribution $q$ shares parameters with $r$, so the full gradient

        $$\\nabla \\mathbb{V}_q[r] = 2\\mathbb{E}_q[(r-\\bar r)\\nabla r] + \\mathbb{E}_q[(r-\\bar r)^2 \\nabla \\log q]$$

        requires both a direct piece (through $r$) and a score-function correction (through $q$). The implementation uses the standard ``direct + score - sg(score)`` surrogate from :meth:`elbo_at`: ``direct`` carries the first term via autodiff through unsg'd $r$, ``score`` carries the second term via the score-function identity with a sample-mean baseline for variance reduction, and the ``- sg(score)`` keeps the value MC-clean.
        """
        q_params = self.approximate_posterior_at(params, x)
        z_samples = jax.lax.stop_gradient(self.pst_man.sample(key, q_params, n_samples))
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z, x))(z_samples)
        log_q_vals = jax.vmap(lambda z: self.pst_man.log_density(q_params, z))(z_samples)  # pyright: ignore[reportAttributeAccessIssue]
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
    """Variational conjugation where ``Posterior`` and ``Prior`` are :class:`Differentiable`.

    Adds the closed-form $c(x)$ baseline (:meth:`conjugation_baseline`) and the closed-form KL between approximate posterior and prior (:meth:`elbo_divergence`). Overrides :meth:`elbo_at` with the lower-variance $c(x) + \\mathbb{E}_q[r(Z)]$ form: $c(x)$ is computed analytically and only the residual $r(z)$ is estimated via MC, instead of the full integrand $\\log p(x \\mid z) + \\log p(z) - \\log q(z \\mid x)$ that the base class samples.

    Mirrors :class:`DifferentiableConjugated` on the analytic side: the base class is the universal abstraction; this subclass adds capabilities that the closed-form log-partition unlocks.
    """

    # ELBO

    def conjugation_baseline(self, params: Array, x: Array) -> Array:
        """The $x$-dependent piece $c(x)$ of the standard-form ELBO $\\mathcal{L}(x) = c(x) + \\mathbb{E}_q[r(Z)]$:

        $$c(x) = s_X(x) \\cdot \\theta_X + \\psi_Z(\\hat\\theta_{Z\\mid X}(x)) - \\psi_Z(\\theta_Z) - \\psi_X(\\theta_X) + \\log h_X(x).$$

        Derivation: substitute the exponential-family forms into $\\log p(x, z) - \\log q(z \\mid x)$ and cancel the shared $s_Z \\cdot \\theta_Z$ and $s_X \\cdot \\Theta_{XZ} \\cdot s_Z$ terms; the $z$-independent remainder is $c(x)$ and the $z$-dependent remainder is $r(z)$ (see :meth:`conjugation_residual`).

        Equivalently, $c(x)$ is the marginal log-density $\\log p(x)$ of the conjugated harmonium one would obtain by promoting the variationally-learned $\\rho_Z$ to an exact analytic conjugation parameter --- the same formula as :meth:`DifferentiableConjugated.log_observable_density`, just evaluated using the variational $\\rho_Z$ in place of the analytic one. The two coincide with the true marginal of the directed generative model iff conjugation is exact ($r \\equiv 0$); otherwise $\\log p(x) = c(x) + \\mathbb{E}_q[r] + \\mathrm{KL}(q \\Vert p(z \\mid x))$.
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

    @override
    def elbo_at(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
    ) -> Array:
        """ELBO $\\mathcal{L}(x) = c(x) + \\mathbb{E}_{q(z\\mid x)}[r(Z)]$ via the analytic baseline (see :meth:`conjugation_baseline`) plus MC over $r$.

        The surrogate ``c_x + direct + score - sg(score)`` is value-correct as an MC estimate of the ELBO and gradient-correct under autodiff: ``direct = mean(r)`` contributes $\\nabla c + \\mathbb{E}_q[\\nabla r]$, the score correction contributes $\\mathbb{E}_q[r \\nabla \\log q]$, and the ``- sg(score)`` cancels the score's value contribution. The population gradient needs no baseline since $c(x)$ already cancels under the zero-score identity; the sample-mean baseline ``b`` only reduces finite-sample variance.

        Lower-variance than the base :meth:`VariationalConjugated.elbo_at` because $c(x)$ is analytic rather than MC.

        Two ``stop_gradient`` calls:

        1. **On samples** --- the recognition sampler may be non-differentiable (e.g. VonMises rejection); samples are evaluation points, not gradient carriers.
        2. **On $r$ inside the score correction** --- its direct gradient is already in ``direct``, so letting autodiff flow through $r$ here would double-count it.

        $\\beta$-VAE-style KL warmup is the caller's responsibility: add $(1-\\beta)\\cdot$ :meth:`elbo_divergence` to the returned value.
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

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Closed-form $\\mathrm{KL}(q(z\\mid x) \\Vert p(z))$, evaluated in Prior space by embedding $q$ via :meth:`pst_prr_emb.embed`.

        Not on the critical path of :meth:`elbo_at` (the standard-form ELBO bundles this implicitly into $c(x)$ and $\\mathbb{E}_q[r]$), but useful externally: callers wanting $\\beta$-VAE-style warmup add $(1-\\beta)\\cdot$ this to :meth:`elbo_at`, and the KL is a natural diagnostic in its own right.
        """
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)
        q_in_prior = self.pst_prr_emb.embed(q_params)
        return self.prr_man.relative_entropy(q_in_prior, p_params)


@dataclass(frozen=True)
class HierarchicalConjugated[LwrCnj: Manifold, RhoMap: Manifold](Pair[LwrCnj, RhoMap]):
    """Combined storage manifold for the lower harmonium's input-independent $\\rho_Y$ and the upper-level parametric $\\rho_Z(x)$ map's parameters $\\phi$.

    A concrete :class:`~goal.geometry.manifold.combinators.Pair` whose first slot is the lower harmonium's conjugation manifold (a flat $\\rho_Y$ vector) and whose second slot is the manifold of the rho-map's parameters (e.g. an :class:`~goal.geometry.manifold.map.AffineMap` or :class:`~goal.geometry.manifold.map.MultilayerPerceptron`).
    """

    # Fields

    _lwr_cnj_man: LwrCnj
    _rho_map_man: RhoMap

    # Overrides

    @property
    @override
    def fst_man(self) -> LwrCnj:
        return self._lwr_cnj_man

    @property
    @override
    def snd_man(self) -> RhoMap:
        return self._rho_map_man


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


class VariationalHierarchical[
    Observable: Differentiable,
    Lower: "VariationalConjugated[Any, Any, Any, Any]",
    Upper: "VariationalConjugated[Any, Any, Any, Any]",
    LwrCnj: Manifold,
    RhoMap: Map[Any, Any],
](
    VariationalConjugated[Observable, Any, Upper, HierarchicalConjugated[LwrCnj, RhoMap]],
    ABC,
):
    """Variational hierarchical harmonium composing a lower and an upper variational conjugated harmonium.

    Models a three-level chain $p(x, y, z) = p(z) p(y \\mid z) p(x \\mid y)$ (with $z$ possibly itself a joint over further latents, e.g. a categorical-Bernoullis mixture), where the lower harmonium models $p(x \\mid y)$ with a learned, input-independent $\\rho_Y$ for the X--Y boundary, and the upper harmonium models $p(y, z)$ with its own internal variational structure.

    Mathematically, the joint recognition is itself a variational conjugated harmonium of the same algebraic shape as ``Upper`` --- ``Upper``'s parameters but with two corrections:

    - the observable-bias slot of ``Upper`` is shifted by $\\Delta_Y(x) = s_X(x) \\cdot \\Theta_{XY} - \\rho_Y$ (the lower harmonium's X-Y bias contribution),
    - the conjugation slot of ``Upper`` is replaced by an input-dependent $\\rho_Z(x) = f(\\hat\\theta_Y(x), \\Theta_{YZ}; \\phi)$ evaluated by :attr:`rho_map`.

    Because the recognition retains ``Upper``'s structure, ancestral sampling and log-density evaluation use ``Upper``'s existing :meth:`sample` and :meth:`log_density` directly (passing the shifted parameter vector).

    Extends :class:`VariationalConjugated` directly: ``pst_man = prr_man = Upper`` (joint $(Y, Z)$) and the inherited base :meth:`elbo_at` (MC + score-function) works without modification --- the joint log-densities ``Upper.log_density(recog_params, ·)`` and ``Upper.log_density(prior_params, ·)`` are the right thing whether ``Upper`` is a leaf variational harmonium or itself a :class:`VariationalHierarchical` (recursive composition). This is the analog of :class:`AnalyticHierarchical extends AnalyticConjugated`.

    Parameter layout (``Triple`` slots):

    1. ``fst`` (``Upper`` slot) --- the upper harmonium's full natural-parameter vector $(\\theta_{Y,\\mathrm{upr}}, \\Theta_{YZ}, \\theta_Z, \\rho_{Z,\\mathrm{stored}})$. The stored $\\rho_Z$ baseline is unused in the hierarchical recognition (replaced by ``rho_map``) but kept so ``Upper`` remains a self-contained variational harmonium that can be initialized, sampled, or evaluated standalone for diagnostics.
    2. ``snd`` (lower likelihood slot) --- $(\\theta_X, \\Theta_{XY})$ as an :class:`~goal.geometry.manifold.map.AffineMap`.
    3. ``trd`` (combined conjugation slot) --- :class:`HierarchicalConjugated` packing the lower harmonium's $\\rho_Y$ (input-independent) and the upper-level rho-map's parameters $\\phi$.

    Five type parameters separate the roles:

    - ``Observable`` --- the bottom-of-stack observable family (the data manifold, $X$).
    - ``Lower`` --- the lower harmonium's type, supplying the X-Y likelihood structure and a $\\rho_Y$ slot.
    - ``Upper`` --- the upper harmonium's type. Often :class:`VariationalSymmetric` or :class:`~goal.models.graphical.variational.VariationalHierarchicalMixture`, or itself a :class:`VariationalHierarchical` for deeper stacks.
    - ``LwrCnj`` --- the lower harmonium's conjugation manifold (typically a flat exponential family).
    - ``RhoMap`` --- a :class:`~goal.geometry.manifold.map.Map` from the upper harmonium's observable manifold to its conjugation manifold; an :class:`~goal.geometry.manifold.map.MultilayerPerceptron` with ``hidden_dims=()`` collapses to a plain affine map.
    """

    # Contract

    @property
    @abstractmethod
    def lwr_hrm(self) -> Lower:
        """The lower variational harmonium (observable $X$, latent $Y$)."""

    @property
    @abstractmethod
    def upr_hrm(self) -> Upper:
        """The upper variational harmonium (observable $Y$, latent $Z$). May itself be hierarchical."""

    @property
    @abstractmethod
    def rho_map(self) -> RhoMap:
        """Parametric map computing $\\rho_Z(x)$ from $\\hat\\theta_Y(x)$."""

    # Triple slots

    @property
    @override
    def fst_man(self) -> Any:
        return self.upr_hrm

    @property
    @override
    def snd_man(self) -> Any:
        return self.lwr_hrm.gen_hrm.lkl_fun_man

    @property
    @override
    def trd_man(self) -> HierarchicalConjugated[LwrCnj, RhoMap]:
        return HierarchicalConjugated(
            _lwr_cnj_man=self.lwr_hrm.cnj_man,
            _rho_map_man=self.rho_map,
        )

    # VariationalConjugated contract

    @property
    @override
    def gen_hrm(self) -> Harmonium[Observable, Any]:
        """The X--Y harmonium underlying the lower boundary; supplies the likelihood storage shape and the observable manifold."""
        return self.lwr_hrm.gen_hrm

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Upper]:
        """Identity --- the recognition and generative prior share ``Upper`` as their manifold."""
        return IdentityEmbedding(self.upr_hrm)

    @property
    @override
    def cnj_man(self) -> HierarchicalConjugated[LwrCnj, RhoMap]:
        """Combined storage for $(\\rho_Y, \\phi)$."""
        return self.trd_man

    # Manifold access

    @property
    @override
    def obs_man(self) -> Observable:
        """The bottom-of-stack observable manifold ($X$)."""
        return self.lwr_hrm.obs_man

    @property
    @override
    def pst_man(self) -> Upper:
        """Manifold of recognition parameters. The recognition is itself a variational conjugated harmonium of the same shape as ``Upper``."""
        return self.upr_hrm

    @property
    @override
    def prr_man(self) -> Upper:
        """Manifold of generative-prior parameters over latents; symmetric to :attr:`pst_man`."""
        return self.upr_hrm

    # Generative / ExponentialFamily contract on the joint $(x, y, z, ...)$

    @property
    @override
    def data_dim(self) -> int:
        """Total dimension of a joint $(x, y, z, ...)$ datapoint: $\\dim X + \\dim_{\\mathrm{data}} \\mathrm{Upper}$."""
        return self.obs_man.data_dim + self.upr_hrm.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Not provided: a flat hierarchical chain does not admit a single joint sufficient statistic in the exponential-family sense (the $\\rho_Y$ and $\\phi$ slots have no sufficient-statistic correspondence). Use :meth:`sample` / :meth:`log_density` instead."""
        raise NotImplementedError(
            "VariationalHierarchical does not provide a flat joint sufficient statistic;"
            + " use sample() and log_density() for evaluation."
        )

    @override
    def log_base_measure(self, x: Array) -> Array:
        """Joint log base measure over $(x, y, z, \\ldots)$ --- recursively composed."""
        x_part = x[..., : self.obs_man.data_dim]
        yz_part = x[..., self.obs_man.data_dim :]
        return self.obs_man.log_base_measure(x_part) + self.upr_hrm.log_base_measure(
            yz_part
        )

    # Parameter access

    @override
    def prior_params(self, params: Array) -> Array:
        """Extract the upper harmonium's full parameter vector."""
        upr, _, _ = self.split_coords(params)
        return upr

    @override
    def likelihood_function(self, params: Array) -> Array:
        """Extract the lower harmonium's likelihood function $(\\theta_X, \\Theta_{XY})$."""
        _, lkl, _ = self.split_coords(params)
        return lkl

    def split_conjugation_params(self, params: Array) -> tuple[Array, Array]:
        """Split the combined conjugation slot into $(\\rho_Y, \\phi)$."""
        _, _, cnj = self.split_coords(params)
        return self.cnj_man.split_coords(cnj)

    # Recognition construction

    def _shifted_upper_params(self, params: Array, x: Array) -> Array:
        """Build the recognition harmonium's parameters by injecting the X-shift into ``Upper``'s observable-bias slot and folding the input-dependent $\\rho_Z(x)$ into ``Upper``'s prior slot via ``Upper.conjugation_parameters``.

        The returned vector lives in ``Upper.dim`` shape and is suitable for direct use with :meth:`Upper.sample` and :meth:`Upper.log_density`: those methods read from the prior slot (which now carries the $\\rho_Z(x)$ correction subtracted out) and the likelihood slot (which carries the X-shifted Y-bias). ``Upper``'s rho slot is zeroed in the returned vector since its role has been absorbed into the prior.
        """
        upr_params, lwr_lkl, _ = self.split_coords(params)
        rho_y, phi = self.split_conjugation_params(params)

        # Decompose upper's params
        upr_prior, upr_lkl, _ = self.upr_hrm.split_coords(upr_params)
        upr_obs_p, upr_int_p = self.upr_hrm.gen_hrm.lkl_fun_man.split_coords(upr_lkl)

        # Compute Δ_Y(x) = s_X(x) · Θ_XY (Y-bias contribution from x via lower's interaction).
        s_x = self.lwr_hrm.obs_man.sufficient_statistic(x)
        _, lwr_int_p = self.lwr_hrm.gen_hrm.lkl_fun_man.split_coords(lwr_lkl)
        delta_y_from_x = self.lwr_hrm.gen_hrm.int_man.transpose_apply(lwr_int_p, s_x)

        # Shifted Y-bias: η̂_Y(x) = η_Y + Δ_Y(x) - ρ_Y.
        hat_eta_y = upr_obs_p + delta_y_from_x - rho_y

        # Input-dependent ρ_Z(x) via the rho-map, then embedded into Upper's prr_man-shape
        # via Upper's own conjugation_parameters (which applies any analytic structural
        # completion appropriate to Upper's graphical form, e.g. mixture completion).
        rho_z_x = self.rho_map(phi, hat_eta_y)
        synth_for_rho = self.upr_hrm.join_coords(upr_prior, upr_lkl, rho_z_x)
        rho_z_in_prr = self.upr_hrm.conjugation_parameters(synth_for_rho)

        # Fold the rho correction into the prior so that Upper.sample / Upper.log_density
        # (which read from the prior slot only) reflect the recognition's shifted distribution.
        recog_prior = upr_prior - rho_z_in_prr
        new_upr_lkl = self.upr_hrm.gen_hrm.lkl_fun_man.join_coords(hat_eta_y, upr_int_p)
        zero_rho = jnp.zeros(self.upr_hrm.cnj_man.dim)
        return self.upr_hrm.join_coords(recog_prior, new_upr_lkl, zero_rho)

    @override
    def approximate_posterior_at(self, params: Array, x: Array) -> Array:
        """Recognition harmonium parameters (in ``Upper``-shape) at observation $x$.

        The returned vector is a valid set of natural parameters for ``Upper``: passing it to :meth:`Upper.sample` yields joint $(y, z)$ samples from the recognition $q(Y, Z \\mid x)$, and passing it to :meth:`Upper.log_density` yields $\\log q(y, z \\mid x)$.
        """
        return self._shifted_upper_params(params, x)

    # Hierarchical conjugation parameters

    @override
    def conjugation_parameters(self, params: Array, x: Array | None = None) -> Array:
        """Return the hierarchical conjugation correction in ``Upper`` shape.

        Composes:

        - $\\rho_Y$ embedded into ``Upper``'s observable-bias slot via :class:`~goal.geometry.exponential_family.graphical.ObservableEmbedding` (analogous to :meth:`~goal.geometry.exponential_family.graphical.AnalyticHierarchical.conjugation_parameters`), and
        - $\\rho_Z(x)$ computed by :attr:`rho_map` at $\\hat\\theta_Y(x)$, embedded into ``Upper``'s rho slot via :meth:`Upper.conjugation_parameters` so that the analytic completion appropriate to ``Upper``'s graphical structure (mixture, hierarchical, ...) is applied.

        Requires ``x``; raises ``ValueError`` if ``x`` is ``None``.
        """
        if x is None:
            raise ValueError(
                "VariationalHierarchical.conjugation_parameters requires x to evaluate the input-dependent rho-map."
            )
        rho_y, phi = self.split_conjugation_params(params)

        # ρ_Y embedded into upper's observable slot (zero elsewhere)
        rho_y_in_upr = ObservableEmbedding(self.upr_hrm.gen_hrm).embed(rho_y)

        # ρ_Z(x) via rho_map, then sent through upper's conjugation_parameters to apply
        # upper's own analytic completion (mixture-shape, etc.).
        upr_params, lwr_lkl, _ = self.split_coords(params)
        upr_prior, upr_lkl, _ = self.upr_hrm.split_coords(upr_params)
        upr_obs_p, _ = self.upr_hrm.gen_hrm.lkl_fun_man.split_coords(upr_lkl)
        s_x = self.lwr_hrm.obs_man.sufficient_statistic(x)
        _, lwr_int_p = self.lwr_hrm.gen_hrm.lkl_fun_man.split_coords(lwr_lkl)
        delta_y_from_x = self.lwr_hrm.gen_hrm.int_man.transpose_apply(lwr_int_p, s_x)
        hat_eta_y = upr_obs_p + delta_y_from_x - rho_y
        rho_z_x = self.rho_map(phi, hat_eta_y)

        # Embed ρ_Z(x) into upper-shape via a synthetic upr_params with rho_z_x in upper's
        # rho slot; upr.conjugation_parameters applies upper's structural completion.
        synth_upr = self.upr_hrm.join_coords(upr_prior, upr_lkl, rho_z_x)
        rho_z_in_upr = self.upr_hrm.conjugation_parameters(synth_upr)

        return rho_y_in_upr + rho_z_in_upr

    # Conjugation residuals

    def lower_conjugation_residual(self, params: Array, y: Array) -> Array:
        """Lower-boundary residual $r_Y(y) = \\rho_Y \\cdot s_Y(y) - \\psi_X(\\theta_X + \\Theta_{XY} \\cdot s_Y(y)) + \\psi_X(\\theta_X)$ at a $Y$ sample."""
        _, lwr_lkl, _ = self.split_coords(params)
        rho_y, _ = self.split_conjugation_params(params)
        # Build a synthetic Lower-params vector with ρ_Y and current lkl. Lower's
        # prior slot is irrelevant for the residual computation.
        zero_prior = jnp.zeros(self.lwr_hrm.prr_man.dim)
        synth_lwr = self.lwr_hrm.join_coords(zero_prior, lwr_lkl, rho_y)
        return self.lwr_hrm.conjugation_residual(synth_lwr, y)

    def upper_conjugation_residual(self, params: Array, z: Array, x: Array) -> Array:
        """Upper-boundary residual $r_Z(z; x) = \\rho_Z(x) \\cdot s_Z(z) - \\psi_Y(\\hat\\theta_Y(x) + \\Theta_{YZ} \\cdot s_Z(z)) + \\psi_Y(\\hat\\theta_Y(x))$ at a $Z$ sample, evaluated at the X-shifted Y-bias."""
        shifted = self._shifted_upper_params(params, x)
        return self.upr_hrm.conjugation_residual(shifted, z)

    @override
    def conjugation_residual(
        self, params: Array, z: Array, x: Array | None = None
    ) -> Array:
        """Joint hierarchical residual $r(y, z; x) = r_Y(y) + r_Z(z; x)$.

        Sums the lower-boundary $r_Y$ (input-independent) and upper-boundary $r_Z$ (input-dependent through $\\rho_Z(x)$). Variance under joint samples gives the aggregate conjugation regularizer used by :meth:`prior_conjugation_loss` and :meth:`recognition_conjugation_loss_at`; for per-level diagnostics use :meth:`lower_conjugation_residual` / :meth:`upper_conjugation_residual` directly.
        """
        if x is None:
            raise ValueError(
                "VariationalHierarchical.conjugation_residual requires x to evaluate"
                + " the upper-boundary residual via the input-dependent rho-map."
            )
        y_data_dim = self.upr_hrm.obs_man.data_dim
        y = z[..., :y_data_dim]
        z_inner = z[..., y_data_dim:]
        return self.lower_conjugation_residual(params, y) + self.upper_conjugation_residual(
            params, z_inner, x
        )

    # Conjugation regularizers (per level)

    def lower_recognition_conjugation_loss_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Variance of the lower-boundary residual $r_Y(Y)$ under joint recognition samples $(Y, Z) \\sim q$.

        Marginalizes Z out by sampling jointly and computing $r_Y$ on the $Y$ component only. Same diagnostic role as :meth:`VariationalConjugated.recognition_conjugation_loss_at` --- vanishes when the X-Y boundary is exactly conjugate, and uses the same ``direct + score - sg(score)`` surrogate so the gradient picks up both the direct piece through $r_Y$ and the score-function correction through $q(Y, Z \\mid x)$.
        """
        recog_params = self._shifted_upper_params(params, x)
        yz_samples = jax.lax.stop_gradient(
            self.upr_hrm.sample(key, recog_params, n_samples)
        )
        y_data_dim = self.upr_hrm.obs_man.data_dim
        y_samples = yz_samples[:, :y_data_dim]
        r_vals = jax.vmap(lambda y: self.lower_conjugation_residual(params, y))(
            y_samples
        )
        log_q_vals = jax.vmap(lambda yz: self.upr_hrm.log_density(recog_params, yz))(
            yz_samples
        )
        return _variance_with_score_correction(r_vals, log_q_vals)

    def upper_recognition_conjugation_loss_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Variance of the upper-boundary residual $r_Z(Z; x)$ under joint recognition samples $(Y, Z) \\sim q$.

        Same surrogate as :meth:`lower_recognition_conjugation_loss_at`; score correction targets $q(Y, Z \\mid x)$.
        """
        recog_params = self._shifted_upper_params(params, x)
        yz_samples = jax.lax.stop_gradient(
            self.upr_hrm.sample(key, recog_params, n_samples)
        )
        y_data_dim = self.upr_hrm.obs_man.data_dim
        z_samples = yz_samples[:, y_data_dim:]
        r_vals = jax.vmap(lambda z: self.upper_conjugation_residual(params, z, x))(
            z_samples
        )
        log_q_vals = jax.vmap(lambda yz: self.upr_hrm.log_density(recog_params, yz))(
            yz_samples
        )
        return _variance_with_score_correction(r_vals, log_q_vals)

    # ELBO divergence (β-warmup KL)

    def elbo_divergence(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Partial-MC joint KL $\\mathrm{KL}(q(Y, Z \\mid x) \\Vert p(Y, Z))$ via chain rule on $(Y, Z)$:

        $$\\mathrm{KL}(q \\Vert p) = \\mathrm{KL}(q(Z \\mid x) \\Vert p(Z)) + \\mathbb{E}_{q(Z \\mid x)}\\left[\\mathrm{KL}(q(Y \\mid Z, x) \\Vert p(Y \\mid Z))\\right].$$

        The outer $Z$-marginal KL is closed-form (Upper's prior and Upper's $\\rho$-shifted prior are both :class:`Differentiable` exponential families, so :meth:`Upper.prr_man.relative_entropy` applies). The inner per-$Z$ KL between $q(Y \\mid Z, x)$ and $p(Y \\mid Z)$ is also closed-form (same family, differing only by an additive shift in the natural parameters) but requires sampling $Z$ from $q(Z \\mid x)$ to average over.

        Different signature from :meth:`VariationalDifferentiable.elbo_divergence` --- this one requires ``key`` and ``n_samples`` for the MC over $Z$. $\\beta$-warmup callers on a :class:`VariationalHierarchical` pass these alongside the usual ``params, x``.

        Requires both ``Upper.prr_man`` and ``Upper.gen_hrm.obs_man`` to be :class:`Differentiable`. For deeper recursion (``Upper`` itself :class:`VariationalHierarchical`), this method falls back to a full-MC estimator via :meth:`Upper.elbo_divergence` recursively.
        """
        # q's Upper-shape recognition and p's Upper-shape prior
        recog_params = self._shifted_upper_params(params, x)
        upr_params = self.prior_params(params)

        # Outer KL_Z: closed-form between two Upper-prior-shaped densities.
        # Upper's prior_params(...) extracts θ_Z (the Z-marginal natural params).
        q_z_params = self.upr_hrm.prior_params(recog_params)
        p_z_params = self.upr_hrm.prior_params(upr_params)
        kl_z = self.upr_hrm.prr_man.relative_entropy(q_z_params, p_z_params)

        # Inner E_{q(Z|x)}[KL(q(Y|Z,x) || p(Y|Z))]: sample Z from q's Z-marginal,
        # compute per-Z closed-form KL on Y given Z.
        z_samples = jax.lax.stop_gradient(
            self.upr_hrm.prr_man.sample(key, q_z_params, n_samples)
        )

        # Y-bias shifts. For each Z sample, q(Y|Z,x) and p(Y|Z) differ only by the
        # additive shift (Δ_Y(x) - ρ_Y) in the Y natural parameters. Both share the
        # same Θ_YZ · s_Z(z) contribution.
        q_obs_p, q_int_p = self.upr_hrm.gen_hrm.lkl_fun_man.split_coords(
            self.upr_hrm.likelihood_function(recog_params)
        )
        p_obs_p, p_int_p = self.upr_hrm.gen_hrm.lkl_fun_man.split_coords(
            self.upr_hrm.likelihood_function(upr_params)
        )

        def kl_y_given_z(z: Array) -> Array:
            s_z = self.upr_hrm.prr_man.sufficient_statistic(z)
            q_y_params = self.upr_hrm.gen_hrm.lkl_fun_man(
                self.upr_hrm.gen_hrm.lkl_fun_man.join_coords(q_obs_p, q_int_p), s_z
            )
            p_y_params = self.upr_hrm.gen_hrm.lkl_fun_man(
                self.upr_hrm.gen_hrm.lkl_fun_man.join_coords(p_obs_p, p_int_p), s_z
            )
            return self.upr_hrm.gen_hrm.obs_man.relative_entropy(q_y_params, p_y_params)

        kl_y_vals = jax.vmap(kl_y_given_z)(z_samples)
        return kl_z + jnp.mean(kl_y_vals)

    # Generative

    @override
    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Natural parameters of $p(x \\mid y)$ at a joint $(y, z')$ sample --- only $y$ is used, $z'$ is sliced away.

        Signature matches the base :meth:`VariationalConjugated.likelihood_at` so the inherited :meth:`sample`, :meth:`log_density`, and :meth:`elbo_at` work uniformly on the joint latent.
        """
        y_data_dim = self.upr_hrm.obs_man.data_dim
        y = z[..., :y_data_dim]
        lkl = self.likelihood_function(params)
        s_y = self.lwr_hrm.pst_man.sufficient_statistic(y)
        return self.lwr_hrm.gen_hrm.lkl_fun_man(lkl, s_y)

    # Initialization

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize with random Upper params, random Lower likelihood, zero $\\rho_Y$, and Glorot-initialized rho-map (if it supports it) else zero."""
        keys = jax.random.split(key, 4)
        upr_params = self.upr_hrm.initialize(keys[0], location, shape)
        lwr_full = self.lwr_hrm.initialize(keys[1], location, shape)
        # Strip lower's prior and rho slots, keep only lkl.
        _, lwr_lkl, _ = self.lwr_hrm.split_coords(lwr_full)
        rho_y = jnp.zeros(self.lwr_hrm.cnj_man.dim)
        phi = _initialize_map_params(self.rho_map, keys[2])
        cnj_params = self.cnj_man.join_coords(rho_y, phi)
        return self.join_coords(upr_params, lwr_lkl, cnj_params)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize using sample data to inform the lower harmonium's observable bias."""
        keys = jax.random.split(key, 4)
        upr_params = self.upr_hrm.initialize(keys[0], location, shape)
        lwr_full = self.lwr_hrm.initialize_from_sample(keys[1], sample, location, shape)
        _, lwr_lkl, _ = self.lwr_hrm.split_coords(lwr_full)
        rho_y = jnp.zeros(self.lwr_hrm.cnj_man.dim)
        phi = _initialize_map_params(self.rho_map, keys[2])
        cnj_params = self.cnj_man.join_coords(rho_y, phi)
        return self.join_coords(upr_params, lwr_lkl, cnj_params)


def _initialize_map_params(rho_map: Map[Any, Any], key: Array) -> Array:
    """Initialize a Map's parameters; uses Glorot if available, otherwise small-scale Gaussian."""
    if hasattr(rho_map, "glorot_initialize"):
        return rho_map.glorot_initialize(key)  # pyright: ignore[reportAttributeAccessIssue]
    return jax.random.normal(key, shape=(rho_map.dim,)) * 0.01


def _variance_with_score_correction(r_vals: Array, log_q_vals: Array) -> Array:
    """MC estimator of $\\mathrm{Var}_q[r]$ as a *loss* whose gradient picks up both the direct piece through $r$ and the score-function correction through $q$.

    Surrogate ``direct + score - sg(score)``:

    - ``direct = mean((r - mean(r))**2)`` --- value-correct biased variance estimate; autodiff carries $2\\mathbb{E}_q[(r-\\bar r) \\nabla r]$.
    - ``score = mean((g_sg - b) * log_q)`` with $g = (r - \\bar r)^2$, sample-mean baseline $b$ for variance reduction --- autodiff carries $\\mathbb{E}_q[(r-\\bar r)^2 \\nabla \\log q]$.
    - ``- sg(score)`` cancels the score's value contribution, leaving the surrogate value equal to ``direct``.
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
    """Fit the variationally-learned $\\rho$ by least-squares regression against samples from the prior.

    Solves $\\rho = \\arg\\min_\\rho \\sum_k (\\chi + \\iota(\\rho) \\cdot \\phi(s_Z(z_k)) - \\psi_X(\\theta_X + \\Theta \\cdot s_Z(z_k)))^2$, where $\\iota = $ :meth:`conjugation_parameters` (applied to a synthetic ``params`` with the candidate $\\rho$ swapped in) and $\\phi = $ :meth:`pst_prr_emb.embed`. The design matrix is obtained by linearizing the (linear) dependence of $\\iota(\\rho)\\cdot\\phi(s_Z)$ on the stored $\\rho$ via :func:`jax.grad`, which lets the same regression code work whether ``conjugation_parameters`` is identity (simple symmetric case) or a richer analytic completion (mixture/hierarchical case). The constant offset from the rho-independent part of $\\iota(\\rho)\\cdot\\phi(s_Z)$ (e.g. the analytic categorical contribution in a mixture) is subtracted from the regression target.

    Returns ``(rho, r_squared, chi, residual_var)`` where ``chi`` is the intercept and ``residual_var`` is the conjugation error $\\mathrm{Var}[r]$ estimated from the regression samples. The intercept absorbs the $z$-independent $\\psi_X(\\theta_X)$ shift in :meth:`conjugation_residual`, so the regression is invariant to that shift.

    A pragmatic fitting/initialization heuristic --- theoretically a black sheep in the variational conjugation family, hence its free-function status.
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
    """Compute conjugation quality metrics (variance, std, $R^2$) under the prior.

    $R^2 = 1 - \\mathrm{Var}[r] / \\mathrm{Var}[\\psi_X(\\eta(z))]$ measures how well the linear correction $\\iota(\\rho) \\cdot \\phi(s_Z)$ approximates $\\psi_X(\\eta(z))$: 1 = perfect conjugation, 0 = no explanatory power, <0 = worse than constant. Both variances are shift-invariant, so the $\\psi_X(\\theta_X)$ term in :meth:`conjugation_residual` does not affect the metric.
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

    Requires ``Posterior: Differentiable`` for the closed-form ``to_mean`` --- defined on :class:`VariationalDifferentiable` only. For :class:`VariationalHierarchical`, reconstruction must be MC-based (sample $y \\sim q(Y \\mid x)$, average $s_Y$, apply lower likelihood); not yet implemented.
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
