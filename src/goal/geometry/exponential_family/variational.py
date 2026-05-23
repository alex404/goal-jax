"""Variational conjugation for harmoniums.

Implements the standard-form ELBO

$$\\mathcal{L}(x) = \\mathbb{E}_{q(z\\mid x)}[\\log p(x, Z) - \\log q(Z \\mid x)] = c(x) + \\mathbb{E}_q[r(Z)],$$

where the **conjugation residual** $r(z) = \\rho_Z \\cdot s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ}\\cdot s_Z(z)) + \\psi_X(\\theta_X)$ is the difference between the LHS and RHS of the conjugation equation: it vanishes pointwise iff the likelihood and prior are exactly conjugate. The residual is the central object of the module --- it drives the ELBO via $\\mathbb{E}_q[r]$ and the conjugation regularizers $\\mathcal{R}_p$ / $\\mathcal{R}_q$ via $\\mathrm{Var}[r]$.

The $\\beta$-VAE-style KL warmup is handled by callers, who add $(1-\\beta)\\cdot\\mathrm{KL}(q \\Vert p)$ to the ELBO externally via :meth:`elbo_divergence`. Keeping $\\beta$ out of :meth:`elbo_at` lets the residual machinery extend without alteration to hierarchical models, where no closed-form KL exists between joint $q$ and joint $p$ and the closed-form-KL lever the recon+KL form depends on disappears.
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

    Models a directed generative process

    $$p(x, z) = p(z; \\theta_Z) \\, p(x \\mid z; \\theta_X, \\Theta_{XZ}),$$

    with an exponential-family prior $p(z; \\theta_Z) \\in $ ``Prior`` and a likelihood whose natural parameters are affine in the latent sufficient statistic,

    $$\\theta_{X \\mid Z}(z) = \\theta_X + \\Theta_{XZ} \\cdot s_Z(z).$$

    When exact conjugation is unavailable, we approximate the posterior by retaining the same exponential-family form as the prior but replacing the analytic conjugation correction with a learned, input-independent parameter $\\rho_Z$:

    $$q(z \\mid x) = p(z; \\hat\\theta_{Z \\mid X}(x)), \\qquad \\hat\\theta_{Z \\mid X}(x) = \\theta_Z + s_X(x) \\cdot \\Theta_{XZ} - \\rho_Z.$$

    Training jointly optimizes $(\\theta_Z, \\theta_X, \\Theta_{XZ}, \\rho_Z)$ under the ELBO so that the model moves toward a regime in which the conjugate-form posterior is accurate.

    The four type parameters separate four roles:

    - ``Observable`` --- observable variable family
    - ``Posterior`` --- family of $q(z \\mid x)$
    - ``Prior`` --- family of $p(z)$; may be a superset of ``Posterior``
    - ``Conjugation`` --- storage space for $\\rho_Z$; may be a sub-manifold of ``Prior`` when only part of the prior requires variational correction (e.g. for mixture priors, the analytic mixture completion handles the categorical part)

    :meth:`conjugation_parameters` peels $\\rho_Z$ from ``params`` and applies the analytic completion appropriate to the graphical structure (zero-padding for chains, mixture completion for mixtures). In the symmetric case (Posterior = Prior = Conjugation) it returns $\\rho_Z$ unchanged.
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

    def conjugation_parameters(self, params: Array) -> Array:
        """Return the conjugation correction in Prior shape.

        Default: $\\rho_Z$ already lives in ``Prior``, so return it unchanged. Subclasses with ``Conjugation`` a proper sub-manifold of ``Prior`` (hierarchical chains, mixtures) override to apply the analytic structural completion --- zero-padding for chains, mixture completion for mixtures.
        """
        _, _, rho = self.split_coords(params)
        return rho

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Natural parameters of the likelihood $p(x \\mid z)$ at the given latent state."""
        _, lkl, _ = self.split_coords(params)
        mz = self.pst_man.sufficient_statistic(z)
        return self.gen_hrm.lkl_fun_man(lkl, mz)

    def approximate_posterior_at(self, params: Array, x: Array) -> Array:
        """Natural parameters of the approximate posterior $q(z \\mid x)$ at the given observation.

        Computes $\\hat\\theta_{Z \\mid X}(x) = \\theta_Z + s_X(x) \\cdot \\Theta_{XZ} - \\rho_Z$ and projects into ``Posterior`` via :attr:`pst_prr_emb`; in the symmetric case the projection is identity.
        """
        prior, lkl, _ = self.split_coords(params)
        rho_full = self.conjugation_parameters(params)
        lat_eff = self.pst_prr_emb.project(prior - rho_full)
        obs_params, int_params = self.gen_hrm.lkl_fun_man.split_coords(lkl)
        hrm_params = self.gen_hrm.join_coords(obs_params, int_params, lat_eff)
        return self.gen_hrm.posterior_at(hrm_params, x)

    # Conjugation residual

    def conjugation_residual(self, params: Array, z: Array) -> Array:
        """Conjugation residual $r(z) = \\rho_Z \\cdot s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ}\\cdot s_Z(z)) + \\psi_X(\\theta_X)$.

        Mathematically, $r$ is the difference between the LHS and RHS of the conjugation equation $\\rho_Z \\cdot s_Z(z) + \\psi_X(\\theta_X) = \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot s_Z(z))$, so $r \\equiv 0$ iff the likelihood is exactly conjugate to the prior. Used as the per-sample summand in the standard-form ELBO ($\\mathcal{L}(x) = c(x) + \\mathbb{E}_q[r(Z)]$) and as the integrand of the conjugation regularizers $\\mathrm{Var}_q[r]$ and $\\mathrm{Var}_p[r]$.

        Generalizes correctly through :meth:`conjugation_parameters` (which embeds $\\rho_Z$ into Prior shape for asymmetric cases) and :attr:`pst_prr_emb` (which embeds the posterior sufficient statistic into Prior space).
        """
        _, lkl, _ = self.split_coords(params)
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(lkl)
        rho_full = self.conjugation_parameters(params)
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
        """ELBO $\\mathcal{L}(x) = c(x) + \\mathbb{E}_{q(z\\mid x)}[r(Z)]$ via MC over the recognition model.

        Decomposition: after substituting exponential-family forms into $\\log p(x,z) - \\log q(z\\mid x)$ and cancelling shared $s_Z\\cdot\\theta_Z$ and $s_X\\cdot\\Theta_{XZ}\\cdot s_Z$ terms,

        $$c(x) = s_X(x) \\cdot \\theta_X + \\psi_Z(\\hat\\theta_{Z\\mid X}(x)) - \\psi_Z(\\theta_Z) - \\psi_X(\\theta_X) + \\log h_X(x)$$

        carries the entire $x$-dependence, and $r(z)$ (see :meth:`conjugation_residual`) carries the entire $z$-dependence.

        The surrogate ``c_x + direct + score - sg(score)`` is value-correct as an MC estimate of the ELBO and gradient-correct under autodiff: ``direct = mean(r)`` contributes $\\nabla c + \\mathbb{E}_q[\\nabla r]$, the score correction contributes $\\mathbb{E}_q[r \\nabla \\log q]$, and the ``- sg(score)`` cancels the score's value contribution. The population gradient needs no baseline since $c(x)$ already cancels under the zero-score identity; the sample-mean baseline ``b`` only reduces finite-sample variance.

        Two ``stop_gradient`` calls:

        1. **On samples** --- the recognition sampler may be non-differentiable (e.g. VonMises rejection); samples are evaluation points, not gradient carriers.
        2. **On $r$ inside the score correction** --- its direct gradient is already in ``direct``, so letting autodiff flow through $r$ here would double-count it.

        $\\beta$-VAE-style KL warmup is the caller's responsibility: add $(1-\\beta)\\cdot$ :meth:`elbo_divergence` to the returned value.
        """
        prior_p, lkl, _ = self.split_coords(params)
        obs_p, _ = self.gen_hrm.lkl_fun_man.split_coords(lkl)

        q_params = self.approximate_posterior_at(params, x)
        z_samples = jax.lax.stop_gradient(self.pst_man.sample(key, q_params, n_samples))

        # c(x): closed-form, $x$-dependent piece
        q_in_prior = self.pst_prr_emb.embed(q_params)
        s_x = self.obs_man.sufficient_statistic(x)
        c_x = (
            jnp.dot(s_x, obs_p)
            + self.prr_man.log_partition_function(q_in_prior)
            - self.prr_man.log_partition_function(prior_p)
            - self.obs_man.log_partition_function(obs_p)
            + self.obs_man.log_base_measure(x)
        )

        # E_q[r(Z)] via MC + score-function correction
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z))(z_samples)
        log_q_vals = jax.vmap(lambda z: self.pst_man.log_density(q_params, z))(
            z_samples
        )

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
        """Closed-form $\\mathrm{KL}(q(z\\mid x) \\Vert p(z))$, evaluated in Prior space by embedding $q$ via :meth:`pst_prr_emb.embed`.

        Not on the critical path of :meth:`elbo_at` (the standard-form ELBO bundles this implicitly into $c(x)$ and $\\mathbb{E}_q[r]$), but useful externally: callers wanting $\\beta$-VAE-style warmup add $(1-\\beta)\\cdot$ this to :meth:`elbo_at`, and the KL is a natural diagnostic in its own right.
        """
        q_params = self.approximate_posterior_at(params, x)
        p_params = self.prior_params(params)
        q_in_prior = self.pst_prr_emb.embed(q_params)
        return self.prr_man.relative_entropy(q_in_prior, p_params)

    # Conjugation regularizers

    def prior_conjugation_loss(
        self, key: Array, params: Array, n_samples: int
    ) -> Array:
        """Prior-based regularizer $\\mathcal{R}_p = \\mathrm{Var}_{p_Z}[r(Z)]$.

        $x$-independent and arguably more consistent with the intuition that conjugation is a property of the likelihood and prior. Can enforce conjugacy in regions of latent space that inference never visits.
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
        """
        q_params = self.approximate_posterior_at(params, x)
        z_samples = jax.lax.stop_gradient(self.pst_man.sample(key, q_params, n_samples))
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z))(z_samples)
        return jnp.var(r_vals)

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
