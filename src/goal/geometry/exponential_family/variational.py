"""Variational conjugation: a graphical harmonium with learned conjugation parameters.

A :class:`~goal.geometry.exponential_family.graphical.DifferentiableGraphical` is
conjugated: the log-partition function of its observable at the likelihood,
$\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z))$, is affine in $\\mathbf s_Z(z)$, which
gives its prior, its log-partition function and the marginal density of its observable in
closed form (see :class:`~goal.geometry.exponential_family.harmonium.Conjugated`). When
the attached harmoniums are not conjugated, :class:`VariationalConjugated` wraps the
graphical harmonium (:attr:`~VariationalConjugated.gen_hrm`) and approximates these
operations, with conjugation parameters $\\rho$ that are learned, or computed where an
attached harmonium is conjugated.

Mathematically, let $\\theta = (\\theta_X, \\Theta_{XZ}, \\theta_Z)$ be the natural parameters of
the graphical harmonium and $\\rho = \\rho(\\theta_X, \\Theta_{XZ})$ conjugation parameters, a
function of the likelihood alone. The model is

$$\\tilde p(x, z) = \\tilde p_Z(z; \\theta_Z + \\rho) \\, p(x \\mid z; \\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)),$$

with the likelihood of the graphical harmonium and the prior of a conjugated harmonium.
The deep model $\\tilde p_Z$ is an exact family, or again a :class:`VariationalConjugated`,
which nests the model. Ancestral sampling of $\\tilde p$ is exact, and when $\\rho$ is the
exact conjugation parameters, $\\tilde p$ is the density of the graphical harmonium. The
recognition model is the deep model at the posterior parameters of the graphical
harmonium,

$$q(z \\mid x) = \\tilde p_Z(z; \\theta_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}).$$

The prior and the recognition model use the same conjugation function. Nested, the deep
model's conjugation parameters are a function of its likelihood, whose observable bias is
$\\theta_Y + \\rho_Y$ in the prior and $\\theta_Y + \\mathbf s_X(x) \\cdot \\Theta_{XY}$ in the
recognition model.

The model is a directed variational autoencoder in other coordinates. For a chain
$X \\leftarrow Y \\leftarrow Z$ with a directed prior $p(y \\mid z; \\theta^{dir}_Y + \\Theta_{YZ}
\\cdot \\mathbf s_Z(z))$, $p(z; \\theta^*_Z)$ and recognition parameters $\\rho_Y$, $\\rho_Z$ at
each level, the change of variables

$$\\theta^{dir}_Y = \\theta_Y + \\rho_Y, \\qquad \\theta^*_Z = \\theta_Z + \\rho_Z(\\theta^{dir}_Y),$$

with likelihoods and interactions shared, maps the generative and recognition models of
one onto the other, provided the conjugation parameters of the posterior and the prior at
each level are the same function of the observable bias.

The central quantity is the **conjugation residual** of each level,

$$r(z) = \\delta \\cdot \\mathbf s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) + \\psi_X(\\theta_X),$$

with the slope $\\delta$ the difference between the prior and the latent bias, $\\delta =
\\rho$. $r$ is constant if and only if $\\rho$ is exact. The log-density of the model is

$$\\log \\tilde p(x, z) = \\theta \\cdot \\mathbf s(x, z) + \\log h(x, z) - \\tilde\\Psi(\\theta) + \\sum r(z),$$

summed over every level, where $\\tilde\\Psi(\\theta) = \\tilde\\Psi_Z(\\theta_Z + \\rho) +
\\psi_X(\\theta_X)$ is the log-partition function the model would have if every conjugation
were exact. The ELBO integrand splits as

$$\\log \\tilde p(x, z) - \\log q(z \\mid x) = c(x) + r_X(z) + \\sum r^0(z) - \\sum r^X(z),$$

where $c(x)$ is the log-marginal under exact conjugation
(:meth:`VariationalConjugated.conjugation_baseline`), $r_X$ is the residual of this
level with slope $\\delta = \\theta^0 - \\theta^X + \\mathbf s_X(x) \\cdot \\Theta_{XZ}$ between
the prior $\\theta^0$ and the posterior $\\theta^X$, and $r^0$ and $r^X$ are the residuals of
the deep model at the prior and at the posterior. The slopes are read off the
parameters actually used, so overriding :meth:`VariationalConjugated.prior` or
:meth:`VariationalConjugated.posterior_at` (e.g. with a stability shim, in an example)
keeps both identities exact.

The residual is used in three ways:

- **ELBO** --- :meth:`VariationalConjugated.elbo_at` computes $c(x)$ in closed form and
  estimates the expected residuals by Monte Carlo, with a score-function gradient.
- **Regularizers** --- the variance of a residual vanishes if and only if its conjugation
  parameters are exact. :meth:`VariationalConjugated.prior_residual_variances` and
  :meth:`VariationalConjugated.recognition_residual_variances_at` penalize it under the
  model and under the recognition model.
- **Fitting and diagnostics** --- :func:`regress_conjugation_parameters` minimizes the
  sampled variance of the residual over $\\rho$ by least squares, and
  :func:`conjugation_metrics` reports it as an $R^2$.

$\\beta$-VAE-style KL warm-up is left to callers, e.g. with
:meth:`ExactPriorVariational.elbo_divergence`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Manifold
from ..manifold.combinators import Null, Pair, Triple
from ..manifold.embedding import LinearEmbedding
from .base import Differentiable
from .combinators import DifferentiableTuple
from .graphical import GraphicalHarmonium


class VariationalConjugated[
    Graphical: GraphicalHarmonium[Any],
    Conjugation: Manifold,
    Deep: Manifold,
](
    Triple[Graphical, Conjugation, Deep],
    ABC,
):
    """A graphical harmonium with conjugation parameters, approximating the operations of a conjugated harmonium.

    Its parameters are ``[hrm_params | cnj_params | dep_params]``: the natural parameters of
    :attr:`gen_hrm`, the coordinates of this level's conjugation parameters, and those of
    the deep model (none when the deep model is exact). It is not an exponential family:
    the conjugation coordinates parameterize the approximation, not the joint density.

    A subclass declares the graphical harmonium, the conjugation coordinates and how they
    give $\\rho$ (:meth:`conjugation_parameters`). The deep model is supplied by
    :class:`ExactPriorVariational` or :class:`NestedPriorVariational`.
    """

    # Contract

    @property
    @abstractmethod
    def gen_hrm(self) -> Graphical:
        """The graphical harmonium."""

    @property
    @abstractmethod
    def cnj_man(self) -> Conjugation:
        """The coordinates of this level's conjugation parameters."""

    @property
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[Any, Any]:
        """Embedding of the posterior of :attr:`gen_hrm` into the prior family."""

    @abstractmethod
    def conjugation_parameters(self, lkl_params: Array, cnj_params: Array) -> Array:
        """The conjugation parameters $\\rho$ of the given likelihood natural parameters, in natural coordinates of the prior family.

        A function of the likelihood alone, so the prior and the recognition model of a
        nested model use the same one. Where an attached harmonium is conjugated, its
        exact conjugation parameters can be used.
        """

    @property
    @abstractmethod
    def dep_man(self) -> Deep:
        """The coordinates of the deep model's conjugation parameters."""

    @abstractmethod
    def deep_params(self, params: Array, prr_params: Array) -> Array:
        """Parameters of the deep model at the given natural parameters of the prior family."""

    @property
    @abstractmethod
    def _deep_data_dim(self) -> int:
        """Data dimension of the deep model."""

    @abstractmethod
    def _deep_log_partition(self, dep_params: Array) -> Array:
        """$\\tilde\\Psi_Z$ of the deep model."""

    @abstractmethod
    def _deep_sample(self, key: Array, dep_params: Array, n: int) -> Array:
        """Ancestral samples of the deep model."""

    @abstractmethod
    def _deep_log_density(self, dep_params: Array, z: Array) -> Array:
        """Log-density of the deep model."""

    @abstractmethod
    def _deep_residuals(self, dep_params: Array, z: Array) -> tuple[Array, ...]:
        """The residuals of the deep model at its datapoint ``z``, bottom first; empty when it is exact."""

    @abstractmethod
    def _deep_residual_variances(
        self, key: Array, dep_params: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """The variance of each residual of the deep model under its own samples; empty when it is exact."""

    @abstractmethod
    def _deep_initialize(self, key: Array, location: float, shape: float) -> Array:
        """Initial coordinates of the deep model's conjugation parameters."""

    # Overrides

    @property
    @override
    def fst_man(self) -> Graphical:
        return self.gen_hrm

    @property
    @override
    def snd_man(self) -> Conjugation:
        return self.cnj_man

    @property
    @override
    def trd_man(self) -> Deep:
        return self.dep_man

    # Methods

    @property
    def obs_man(self) -> DifferentiableTuple[Any]:
        """The observables of the attached harmoniums, as a differentiable tuple: the residual needs $\\psi_X$."""
        return DifferentiableTuple(
            tuple(obs_hrm.obs_man for obs_hrm in self.gen_hrm.obs_hrms)
        )

    @property
    def pst_man(self) -> Any:
        """The posterior family of :attr:`gen_hrm`."""
        return self.gen_hrm.pst_man

    @property
    def prr_man(self) -> Any:
        """The prior family."""
        return self.pst_prr_emb.amb_man

    @property
    def data_dim(self) -> int:
        """Dimension of a joint datapoint $(x, z)$."""
        return self.obs_man.data_dim + self._deep_data_dim

    def likelihood_function(self, params: Array) -> Array:
        """The likelihood natural parameters $(\\theta_X, \\Theta_{XZ})$."""
        hrm_params, _, _ = self.split_coords(params)
        return self.gen_hrm.likelihood_function(hrm_params)

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Natural parameters of $p(x \\mid z)$ at a datapoint of the prior family, whose leading slice is the posterior's datapoint."""
        s_z = self.pst_man.sufficient_statistic(z[..., : self.pst_man.data_dim])
        return self.gen_hrm.lkl_fun_man(self.likelihood_function(params), s_z)

    def prior(self, params: Array) -> Array:
        """Natural parameters $\\theta_Z + \\rho$ of the prior family. A subclass may override it, e.g. with a stability shim."""
        hrm_params, cnj_params, _ = self.split_coords(params)
        obs_params, int_params, lat_params = self.gen_hrm.split_coords(hrm_params)
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        rho = self.conjugation_parameters(lkl_params, cnj_params)
        return self.pst_prr_emb.translate(rho, lat_params)

    def posterior_at(self, params: Array, x: Array) -> Array:
        """Posterior natural parameters $\\theta_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}$ of :attr:`gen_hrm`. A subclass may override it, e.g. with a stability shim."""
        hrm_params, _, _ = self.split_coords(params)
        return self.gen_hrm.posterior_at(hrm_params, x)

    def log_partition_function(self, params: Array) -> Array:
        """$\\tilde\\Psi(\\theta) = \\tilde\\Psi_Z(\\theta_Z + \\rho) + \\psi_X(\\theta_X)$: the log-partition function under exact conjugation."""
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        return self._deep_log_partition(
            self.deep_params(params, self.prior(params))
        ) + self.obs_man.log_partition_function(obs_params)

    def conjugation_baseline(self, params: Array, x: Array) -> Array:
        """The latent-independent ELBO term $c(x) = \\mathbf s_X(x) \\cdot \\theta_X + \\log h_X(x) - \\psi_X(\\theta_X) + \\tilde\\Psi_Z(\\theta^X) - \\tilde\\Psi_Z(\\theta^0)$, the log-marginal under exact conjugation, with $\\theta^0$ the prior and $\\theta^X$ the posterior.

        It is the formula of
        :meth:`~goal.geometry.exponential_family.harmonium.DifferentiableConjugated.log_observable_density`
        with $\\tilde\\Psi_Z$ in place of $\\psi_Z$, and equals $\\log \\tilde p(x)$ when every
        conjugation is exact.
        """
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        dep_prior = self.deep_params(params, self.prior(params))
        dep_post = self._recognition_params(params, x)
        return (
            jnp.dot(self.obs_man.sufficient_statistic(x), obs_params)
            + self.obs_man.log_base_measure(x)
            - self.obs_man.log_partition_function(obs_params)
            + self._deep_log_partition(dep_post)
            - self._deep_log_partition(dep_prior)
        )

    def log_density(self, params: Array, xz: Array) -> Array:
        """$\\log \\tilde p(x, z) = \\log \\tilde p_Z(z) + \\log p(x \\mid z)$ of a joint datapoint."""
        x = xz[..., : self.obs_man.data_dim]
        z = xz[..., self.obs_man.data_dim :]
        dep_params = self.deep_params(params, self.prior(params))
        return self._deep_log_density(dep_params, z) + self.obs_man.log_density(
            self.likelihood_at(params, z), x
        )

    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Ancestral samples of $\\tilde p(x, z)$: $z$ from the deep model at the prior, then $x \\sim p(x \\mid z)$."""
        key_z, key_x = jax.random.split(key)
        dep_params = self.deep_params(params, self.prior(params))
        zs = self._deep_sample(key_z, dep_params, n)

        def sample_x(k: Array, z: Array) -> Array:
            return self.obs_man.sample(k, self.likelihood_at(params, z), 1)[0]

        xs = jax.vmap(sample_x)(jax.random.split(key_x, n), zs)
        return jnp.concatenate([xs, zs], axis=-1)

    def observable_sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Samples of $\\tilde p(x)$."""
        return self.sample(key, params, n)[:, : self.obs_man.data_dim]

    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """The initialization of :attr:`gen_hrm`, with zero conjugation coordinates at this level and the deep model's initial ones."""
        k_hrm, k_dep = jax.random.split(key)
        hrm_params = self.gen_hrm.initialize(k_hrm, location, shape)
        dep_params = self._deep_initialize(k_dep, location, shape)
        return self.join_coords(hrm_params, self.cnj_man.zeros(), dep_params)

    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """As :meth:`initialize`, with the observable biases from the sample."""
        k_hrm, k_dep = jax.random.split(key)
        hrm_params = self.gen_hrm.initialize_from_sample(k_hrm, sample, location, shape)
        dep_params = self._deep_initialize(k_dep, location, shape)
        return self.join_coords(hrm_params, self.cnj_man.zeros(), dep_params)

    def conjugation_residuals(self, params: Array, z: Array) -> tuple[Array, ...]:
        """The residual of each level of $\\tilde p$ at the datapoint ``z`` of the prior family, bottom first.

        The slope of this level is the prior minus the embedded latent bias, $\\rho$ unless
        :meth:`prior` is overridden.
        """
        hrm_params, _, _ = self.split_coords(params)
        _, _, lat_params = self.gen_hrm.split_coords(hrm_params)
        prr_params = self.prior(params)
        slope = prr_params - self.pst_prr_emb.embed(lat_params)
        return (
            self._residual(params, slope, z),
            *self._deep_residuals(self.deep_params(params, prr_params), z),
        )

    def elbo_residual(self, params: Array, x: Array, z: Array) -> Array:
        """The ELBO integrand minus :meth:`conjugation_baseline`: $r_X(z) + \\sum r^0(z) - \\sum r^X(z)$.

        $r_X$ is this level's residual with slope the prior minus the posterior plus
        $\\mathbf s_X(x) \\cdot \\Theta_{XZ}$, and $r^0$ and $r^X$ are the residuals of the deep
        model at the prior and at the posterior.
        """
        dep_prior = self.deep_params(params, self.prior(params))
        dep_post = self._recognition_params(params, x)
        return (
            self._elbo_residual(params, x, z)
            + _total(self._deep_residuals(dep_prior, z))
            - _total(self._deep_residuals(dep_post, z))
        )

    def sample_recognition(self, key: Array, params: Array, x: Array, n: int) -> Array:
        """Samples of $q(z \\mid x)$, the deep model at the posterior, as rows."""
        return self._deep_sample(key, self._recognition_params(params, x), n)

    def recognition_log_density(self, params: Array, x: Array, z: Array) -> Array:
        """$\\log q(z \\mid x)$."""
        return self._deep_log_density(self._recognition_params(params, x), z)

    def elbo_at(self, key: Array, params: Array, x: Array, n_samples: int) -> Array:
        """Estimate the ELBO $\\mathcal L(x) = c(x) + \\mathbb E_q[r_X + \\sum r^0 - \\sum r^X]$: the baseline in closed form, the residuals by Monte Carlo.

        Only the residuals are sampled, and only they drive the score-function correction,

        $$\\nabla \\mathcal L(x) = \\mathbb E_q[\\nabla f] + \\mathbb E_q[(r - b) \\nabla \\log q],$$

        with the leave-one-out baseline $b$ of :func:`score_mean_estimate`. The samples
        carry no gradient: the sampler may be non-differentiable (e.g. von Mises rejection
        sampling).
        """
        dep_post = self._recognition_params(params, x)
        zs = jax.lax.stop_gradient(
            self._deep_sample(key, jax.lax.stop_gradient(dep_post), n_samples)
        )
        r_vals = jax.vmap(lambda z: self.elbo_residual(params, x, z))(zs)
        log_q_vals = jax.vmap(lambda z: self._deep_log_density(dep_post, z))(zs)
        return self.conjugation_baseline(params, x) + score_mean_estimate(
            r_vals, log_q_vals
        )

    def mean_elbo(self, key: Array, params: Array, xs: Array, n_samples: int) -> Array:
        """Mean of :meth:`elbo_at` over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        return jnp.mean(
            jax.vmap(lambda k, x: self.elbo_at(k, params, x, n_samples))(keys, xs)
        )

    def prior_residual_variances(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """The variance under $\\tilde p$ of the residual of each level, bottom first.

        The variance of each level is under samples of its own deep model, with a
        score-function term for the dependence of their distribution on the parameters
        (:func:`score_variance_estimate`). Needs ``n_samples >= 2``.
        """
        key_own, key_deep = jax.random.split(key)
        dep_prior = self.deep_params(params, self.prior(params))
        zs = jax.lax.stop_gradient(self._deep_sample(key_own, dep_prior, n_samples))
        r_vals = jax.vmap(lambda z: self.conjugation_residuals(params, z)[0])(zs)
        log_p = jax.vmap(lambda z: self._deep_log_density(dep_prior, z))(zs)
        return (
            score_variance_estimate(r_vals, log_p),
            *self._deep_residual_variances(key_deep, dep_prior, n_samples),
        )

    def recognition_residual_variances_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """The variance under $q(z \\mid x)$ of $r_X$, then of each residual $r^X$ of the deep model at the posterior.

        For an exact deep model, $\\mathrm{Var}_q[r_X] = \\mathrm{Var}_q[\\log(q(Z \\mid x) /
        p(Z \\mid x))]$, which vanishes if and only if the recognition model is the posterior.
        Gradients include the score-function term (:func:`score_variance_estimate`). Needs
        ``n_samples >= 2``.
        """
        key_own, key_deep = jax.random.split(key)
        dep_post = self._recognition_params(params, x)
        zs = jax.lax.stop_gradient(self._deep_sample(key_own, dep_post, n_samples))
        r_vals = jax.vmap(lambda z: self._elbo_residual(params, x, z))(zs)
        log_q = jax.vmap(lambda z: self._deep_log_density(dep_post, z))(zs)
        return (
            score_variance_estimate(r_vals, log_q),
            *self._deep_residual_variances(key_deep, dep_post, n_samples),
        )

    def mean_recognition_residual_variances(
        self, key: Array, params: Array, xs: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """Mean of :meth:`recognition_residual_variances_at` over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        vals = jax.vmap(
            lambda k, x: self.recognition_residual_variances_at(k, params, x, n_samples)
        )(keys, xs)
        return tuple(jnp.mean(v) for v in vals)

    # Internals

    def _recognition_params(self, params: Array, x: Array) -> Array:
        """Parameters of the deep model at the embedded posterior."""
        return self.deep_params(
            params, self.pst_prr_emb.embed(self.posterior_at(params, x))
        )

    def _coupling(self, params: Array, x: Array) -> Array:
        """The term $\\mathbf s_X(x) \\cdot \\Theta_{XZ}$ that the observation adds to the latent bias."""
        hrm_params, _, _ = self.split_coords(params)
        obs_params, int_params, _ = self.gen_hrm.split_coords(hrm_params)
        zero_lat = self.gen_hrm.join_coords(
            obs_params, int_params, self.pst_man.zeros()
        )
        return self.gen_hrm.posterior_at(zero_lat, x)

    def _residual(self, params: Array, slope: Array, z: Array) -> Array:
        """$\\delta \\cdot \\mathbf s(z) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) + \\psi_X(\\theta_X)$ with the slope $\\delta$ in natural coordinates of the prior family."""
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        return (
            jnp.dot(slope, self.prr_man.sufficient_statistic(z))
            - self.obs_man.log_partition_function(self.likelihood_at(params, z))
            + self.obs_man.log_partition_function(obs_params)
        )

    def _elbo_residual(self, params: Array, x: Array, z: Array) -> Array:
        """The ELBO's residual $r_X$ of this level, with slope the prior minus the posterior plus $\\mathbf s_X(x) \\cdot \\Theta_{XZ}$."""
        emb = self.pst_prr_emb
        slope = (
            self.prior(params)
            - emb.embed(self.posterior_at(params, x))
            + emb.embed(self._coupling(params, x))
        )
        return self._residual(params, slope, z)


class ExactPriorVariational[
    Graphical: GraphicalHarmonium[Any],
    Prior: Differentiable,
    Conjugation: Manifold,
](
    VariationalConjugated[Graphical, Conjugation, Null],
    ABC,
):
    """Variational conjugation with an exact deep model: the prior family is :class:`Differentiable`, and used as it is."""

    # Contract

    @property
    @override
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[Prior, Any]: ...

    # Overrides

    @property
    @override
    def prr_man(self) -> Prior:
        return self.pst_prr_emb.amb_man

    @property
    @override
    def dep_man(self) -> Null:
        return Null()

    @override
    def deep_params(self, params: Array, prr_params: Array) -> Array:
        return prr_params

    @property
    @override
    def _deep_data_dim(self) -> int:
        return self.prr_man.data_dim

    @override
    def _deep_log_partition(self, dep_params: Array) -> Array:
        return self.prr_man.log_partition_function(dep_params)

    @override
    def _deep_sample(self, key: Array, dep_params: Array, n: int) -> Array:
        return self.prr_man.sample(key, dep_params, n)

    @override
    def _deep_log_density(self, dep_params: Array, z: Array) -> Array:
        return self.prr_man.log_density(dep_params, z)

    @override
    def _deep_residuals(self, dep_params: Array, z: Array) -> tuple[Array, ...]:
        return ()

    @override
    def _deep_residual_variances(
        self, key: Array, dep_params: Array, n_samples: int
    ) -> tuple[Array, ...]:
        return ()

    @override
    def _deep_initialize(self, key: Array, location: float, shape: float) -> Array:
        return jnp.zeros(0)

    # Methods

    def recognition_at(self, params: Array, x: Array) -> Array:
        """Natural parameters of $q(z \\mid x)$ in the prior family."""
        return self._recognition_params(params, x)

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Closed-form $\\mathrm{KL}(q(z \\mid x) \\Vert \\tilde p(z))$, for $\\beta$-VAE-style warm-up and diagnostics."""
        return self.prr_man.relative_entropy(
            self.recognition_at(params, x), self.prior(params)
        )


class NestedPriorVariational[
    Graphical: GraphicalHarmonium[Any],
    DeepGraphical: GraphicalHarmonium[Any],
    Conjugation: Manifold,
](
    VariationalConjugated[Graphical, Conjugation, "ConjugationCoordinates"],
    ABC,
):
    """Variational conjugation with a variational deep model: the prior family is a graphical harmonium, approximated by :attr:`dep_vrt`.

    The deep model at prior-family parameters $\\theta'$ is :attr:`dep_vrt` at
    ``[theta' | dep_params]``: its conjugation coordinates are stored in this model's
    ``dep_params``, so its graphical harmonium's coordinates are not duplicated.
    """

    # Contract

    @property
    @abstractmethod
    def dep_vrt(self) -> VariationalConjugated[DeepGraphical, Any, Any]:
        """The variational model of the prior family, whose graphical harmonium is the ambient of :attr:`pst_prr_emb`."""

    @property
    @override
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[DeepGraphical, Any]: ...

    # Overrides

    @property
    @override
    def prr_man(self) -> DeepGraphical:
        return self.pst_prr_emb.amb_man

    @property
    @override
    def dep_man(self) -> ConjugationCoordinates:
        return ConjugationCoordinates(self.dep_vrt)

    @override
    def deep_params(self, params: Array, prr_params: Array) -> Array:
        _, _, dep_params = self.split_coords(params)
        return self.dep_vrt.join_coords(
            prr_params, *self.dep_man.split_coords(dep_params)
        )

    @property
    @override
    def _deep_data_dim(self) -> int:
        return self.dep_vrt.data_dim

    @override
    def _deep_log_partition(self, dep_params: Array) -> Array:
        return self.dep_vrt.log_partition_function(dep_params)

    @override
    def _deep_sample(self, key: Array, dep_params: Array, n: int) -> Array:
        return self.dep_vrt.sample(key, dep_params, n)

    @override
    def _deep_log_density(self, dep_params: Array, z: Array) -> Array:
        return self.dep_vrt.log_density(dep_params, z)

    @override
    def _deep_residuals(self, dep_params: Array, z: Array) -> tuple[Array, ...]:
        return self.dep_vrt.conjugation_residuals(
            dep_params, z[self.dep_vrt.obs_man.data_dim :]
        )

    @override
    def _deep_residual_variances(
        self, key: Array, dep_params: Array, n_samples: int
    ) -> tuple[Array, ...]:
        return self.dep_vrt.prior_residual_variances(key, dep_params, n_samples)

    @override
    def _deep_initialize(self, key: Array, location: float, shape: float) -> Array:
        _, cnj_params, dep_params = self.dep_vrt.split_coords(
            self.dep_vrt.initialize(key, location, shape)
        )
        return self.dep_man.join_coords(cnj_params, dep_params)


@dataclass(frozen=True)
class ConjugationCoordinates(Pair[Manifold, Manifold]):
    """The conjugation coordinates ``[cnj_params | dep_params]`` of a variational model: its parameters without those of its graphical harmonium."""

    # Fields

    vrt_man: VariationalConjugated[Any, Any, Any]

    # Overrides

    @property
    @override
    def fst_man(self) -> Manifold:
        return self.vrt_man.cnj_man

    @property
    @override
    def snd_man(self) -> Manifold:
        return self.vrt_man.dep_man


def _total(vals: tuple[Array, ...]) -> Array:
    return sum(vals, start=jnp.zeros(()))


def score_mean_estimate(vals: Array, log_density_vals: Array) -> Array:
    """Score-function estimate of $\\mathbb E_q[f]$ from $f(z_k)$ and $\\log q(z_k)$ at samples $z_k \\sim q$.

    The value is the sample mean of $f$, and its autodiff gradient is an unbiased estimate of $\\nabla \\mathbb E_q[f]$, including the dependence of the sampling distribution $q$ on the parameters: the mean carries $\\mathbb E_q[\\nabla f]$, and a score-function term adds $\\frac{1}{K} \\sum_k (f_k - b_k) \\nabla \\log q(z_k)$ with the leave-one-out baseline $b_k = \\frac{1}{K-1} \\sum_{j \\neq k} f_j$. That baseline is independent of $z_k$, so it leaves the gradient unbiased; the plain sample mean would shrink the score term by $(K-1)/K$. A single sample gets no baseline. The samples themselves must carry no gradient.
    """
    vals_sg = jax.lax.stop_gradient(vals)
    n = vals.shape[0]
    centered = vals_sg if n == 1 else n / (n - 1) * (vals_sg - jnp.mean(vals_sg))
    score = jnp.mean(centered * log_density_vals)
    return jnp.mean(vals) + score - jax.lax.stop_gradient(score)


def score_variance_estimate(vals: Array, log_density_vals: Array) -> Array:
    """Score-function estimate of $\\mathrm{Var}_q[f]$ from $f(z_k)$ and $\\log q(z_k)$ at $K \\geq 2$ samples $z_k \\sim q$.

    The value is the unbiased sample variance, and its autodiff gradient is an unbiased estimate of $\\nabla \\mathrm{Var}_q[f]$, including the dependence of the sampling distribution $q$ on the parameters. The samples themselves must carry no gradient.

    Mathematically, $\\mathrm{Var}_q[f] = \\frac{1}{2} \\mathbb E_{q \\otimes q}[(f(Z) - f(Z'))^2]$, so the mean of $\\frac{1}{2}(f_i - f_j)^2$ over the pairs $i \\neq j$ is unbiased, and so is its gradient: a direct piece through $f$, and a score piece $\\frac{1}{2}(f_i - f_j)^2 (\\nabla \\log q(z_i) + \\nabla \\log q(z_j))$ from the score of $q \\otimes q$.
    """
    n = vals.shape[0]
    pairs = 0.5 * (vals[:, None] - vals[None, :]) ** 2
    pairs_sg = jax.lax.stop_gradient(pairs)
    lq = log_density_vals
    score = jnp.sum(pairs_sg * (lq[:, None] + lq[None, :]))
    return (jnp.sum(pairs) + score - jax.lax.stop_gradient(score)) / (n * (n - 1))


def regress_conjugation_parameters(
    model: VariationalConjugated[Any, Any, Any],
    key: Array,
    params: Array,
    n_samples: int,
) -> tuple[Array, Array, Array, Array]:
    """Fit this level's conjugation coordinates by least squares against samples of the deep model at the prior; they minimize the sampled variance of the residual.

    Solves $\\min_{\\chi, c} \\sum_k (\\chi + \\rho(c) \\cdot \\mathbf s(z_k) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z_k)))^2$ over the conjugation coordinates $c$, with $\\rho$ from :meth:`~VariationalConjugated.conjugation_parameters` linearized at $c = 0$, so it is exact when $\\rho$ is affine in $c$. The intercept is free and the variance is shift-invariant, so the fit minimizes the sampled variance of the residual. The samples are drawn at the given parameters and held fixed, although with $\\rho$ in the prior they depend on $c$. They carry no gradient, since the sampler may be non-differentiable; gradients flow through the targets $\\psi_X$.

    Returns ``(cnj_params, r_squared, chi, residual_var)``. A fitting heuristic, not part of the variational objective.
    """
    lkl_params = model.likelihood_function(params)
    zero_cnj = model.cnj_man.zeros()
    obs_dim = model.obs_man.data_dim
    zs = jax.lax.stop_gradient(model.sample(key, params, n_samples)[:, obs_dim:])

    def design_and_target(z: Array) -> tuple[Array, Array, Array]:
        s_z = model.prr_man.sufficient_statistic(z)

        def rho_dot_s(c: Array) -> Array:
            return jnp.dot(model.conjugation_parameters(lkl_params, c), s_z)

        s_proj = jax.grad(rho_dot_s)(zero_cnj)
        offset = rho_dot_s(zero_cnj)
        psi_x = model.obs_man.log_partition_function(model.likelihood_at(params, z))
        return s_proj, psi_x, offset

    s_proj_all, psi_all, offset_all = jax.vmap(design_and_target)(zs)
    adj_psi = psi_all - offset_all

    design = jnp.concatenate([jnp.ones((n_samples, 1)), s_proj_all], axis=1)
    coeffs = jnp.linalg.lstsq(design, adj_psi, rcond=None)[0]
    chi = coeffs[0]
    cnj_params = coeffs[1:]

    residuals = adj_psi - design @ coeffs
    ss_res = jnp.sum(residuals**2)
    ss_tot = jnp.sum((adj_psi - jnp.mean(adj_psi)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    return cnj_params, r_squared, chi, jnp.var(residuals)


def conjugation_metrics(
    model: VariationalConjugated[Any, Any, Any],
    key: Array,
    params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array]:
    """Conjugation quality ``(var_r, std_r, r_squared)`` of this level under the model.

    $R^2 = 1 - \\mathrm{Var}[r] / \\mathrm{Var}[\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(Z))]$ measures how much of the variation of the log-partition function the conjugation parameters explain: 1 is exact conjugation, 0 no better than a constant.
    """
    var_key, psi_key = jax.random.split(key)
    var_r = model.prior_residual_variances(var_key, params, n_samples)[0]
    zs = model.sample(psi_key, params, n_samples)[:, model.obs_man.data_dim :]
    psi_vals = jax.vmap(
        lambda z: model.obs_man.log_partition_function(model.likelihood_at(params, z))
    )(zs)
    var_psi = jnp.var(psi_vals, ddof=1)
    return var_r, jnp.sqrt(var_r), 1.0 - var_r / var_psi


def reconstruct(
    model: ExactPriorVariational[Any, Any, Any], params: Array, x: Array
) -> Array:
    """Observable means of the likelihood at the recognition model's mean statistics: a mean-field reconstruction."""
    z_means = model.pst_prr_emb.project(
        model.prr_man.to_mean(model.recognition_at(params, x))
    )
    lkl_params = model.gen_hrm.lkl_fun_man(model.likelihood_function(params), z_means)
    return model.obs_man.to_mean(lkl_params)


def reconstruction_error(
    model: ExactPriorVariational[Any, Any, Any], params: Array, xs: Array
) -> Array:
    """Mean squared reconstruction error over a batch."""
    recons = jax.vmap(lambda x: reconstruct(model, params, x))(xs)
    return jnp.mean((xs - recons) ** 2)
