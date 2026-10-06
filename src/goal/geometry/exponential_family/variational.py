"""Variational conjugation: harmoniums with learned conjugation parameters and a recognition model of conjugate form.

A harmonium is conjugated when

$$\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) = \\rho_Z \\cdot \\mathbf s_Z(z) + \\chi$$

holds for all $z$, for some conjugation parameters $\\rho_Z$ and constant $\\chi$ (see :class:`~goal.geometry.exponential_family.harmonium.Conjugated`). When no such $\\rho_Z$ exists, or it cannot be computed, this module learns $\\rho_Z$ and uses the recognition model

$$q(z \\mid x) = p(z; \\hat\\theta_{Z \\mid X}(x)), \\qquad \\hat\\theta_{Z \\mid X}(x) = \\theta_Z - \\rho_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}.$$

$\\rho_Z$ is trained jointly with the generative parameters under the ELBO.

The central quantity is the **conjugation residual**

$$r(z) = \\rho_Z \\cdot \\mathbf s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) + \\psi_X(\\theta_X),$$

the difference between the two sides of the conjugation equation, with $\\psi_X(\\theta_X)$ in place of $\\chi$. $r$ is constant if and only if $\\rho_Z$ are exact conjugation parameters. The ELBO integrand $f(x, z) = \\log p(x, z) - \\log q(z \\mid x)$ splits as

$$f(x, z) = c(x) + r(z), \\qquad \\mathcal{L}(x) = c(x) + \\mathbb{E}_{q(z \\mid x)}[r(Z)],$$

where $c(x)$ does not depend on $z$. This is the **standard form** of the ELBO. The choice of $\\psi_X(\\theta_X)$ for $\\chi$ cancels between $c(x)$ and $r(z)$, so the ELBO does not depend on it.

The residual is used in three ways:

- **ELBO** --- :meth:`VariationalDifferentiable.elbo_at` computes $c(x)$ in closed form and estimates $\\mathbb{E}_q[r]$ by Monte Carlo, with a score-function gradient.
- **Regularizers** --- $\\mathrm{Var}_q[r] = \\mathrm{Var}_q[\\log(q / p(z \\mid x))]$ vanishes if and only if the recognition model is the posterior. Penalizing the variance of the residual under the prior (:meth:`VariationalDifferentiable.prior_residual_variance`) or under the recognition model (:meth:`VariationalDifferentiable.recognition_residual_variance_at`) moves the model toward exact conjugation.
- **Fitting and diagnostics** --- :func:`regress_conjugation_parameters` minimizes the sampled $\\mathrm{Var}_p[r]$ over $\\rho_Z$ by least squares, and :func:`conjugation_metrics` reports it as an $R^2$.

The prior of a variational harmonium may itself be a variational harmonium, which nests the model (see :class:`VariationalPrior`).

The classes mirror the ``Conjugated`` hierarchy: :class:`VariationalConjugated` requires only :class:`Generative` latents, :class:`VariationalDifferentiable` requires closed-form log-partition functions and provides the ELBO, and :class:`VariationalSymmetric` uses one manifold for the posterior and the prior.

$\\beta$-VAE-style KL warmup is left to callers: add $(1 - \\beta) \\cdot \\mathrm{KL}(q \\Vert p)$ with :meth:`VariationalDifferentiable.elbo_divergence`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Manifold
from ..manifold.combinators import Triple
from ..manifold.embedding import IdentityEmbedding, LinearEmbedding
from ..manifold.map import AffineMap
from ..manifold.util import split_by_dims
from .base import Differentiable, Generative
from .harmonium import Harmonium


class VariationalPrior(ABC):
    """A model that can be the prior of a variational harmonium.

    A prior is conditioned on the variables below it by a shift of its natural parameters. An exact family adds the shift to all of its natural parameters. A variational harmonium adds it to the bias of its observable, and passes a shift on to its own prior (see :class:`VariationalDifferentiable`). With ``shift=None`` the methods describe the unconditioned model.

    The data of a prior are split into levels, ordered from the bottom, nearest the observation, to the top. An exact family has one level; a variational harmonium has its observable, then the levels of its prior. The ``conditional_*`` methods return one entry per level, each conditioned on the level above it.
    """

    # Contract

    @property
    @abstractmethod
    def shift_man(self) -> Manifold:
        """The coordinates of a shift: the natural parameters of an exact family, or the observable bias of a variational harmonium. A harmonium with this prior also writes its conjugation parameters in them."""

    @property
    @abstractmethod
    def level_dims(self) -> tuple[int, ...]:
        """The data dimension of each level."""

    @property
    @abstractmethod
    def level_learned(self) -> tuple[tuple[bool, ...], ...]:
        """For each variational harmonium in the prior, bottom first, whether each of its residuals in :meth:`conditional_residuals` uses learned conjugation parameters. Empty for an exact family."""

    @abstractmethod
    def conditional_sample(
        self, key: Array, params: Array, shift: Array | None
    ) -> list[Array]:
        """One sample of each level, drawn from the top down."""

    @abstractmethod
    def conditional_log_densities(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[Array]:
        """The log-density of each level given the level above it."""

    @abstractmethod
    def conditional_parameters(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[Array]:
        """The natural parameters of each level given the level above it."""

    @abstractmethod
    def conditional_residuals(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[tuple[tuple[Array, ...], tuple[Array, ...]]]:
        """For each variational harmonium in the prior, bottom first, its residuals $(r^0, r^X)$ at the level above its observable. $r^X$ is empty when ``shift`` is ``None``."""

    @abstractmethod
    def log_partition_shift(self, params: Array, shift: Array) -> Array:
        """$\\psi(\\theta + \\delta) - \\psi(\\theta)$ of the bottom level under the shift $\\delta$, plus the same for each level above under the shift it receives."""


class VariationalConjugated[
    Observable: Differentiable,
    Posterior: Generative,
    Prior: Generative,
    Conjugation: Manifold,
](
    Generative,
    Triple[AffineMap[Observable, Posterior], Prior, Conjugation],
    ABC,
):
    """Variational harmonium with parameter layout ``[lkl_params, prior_params, rho]``.

    ``lkl_params`` is $(\\theta_X, \\Theta_{XZ})$ and ``prior_params`` is $\\theta_Z$, so the blocks are in the order of a harmonium's ``[obs | int | lat]``, with the prior in place of the latent bias.

    Mathematically, the generative model is a directed harmonium

    $$p(x, z) = p(z; \\theta_Z) \\, p(x \\mid z; \\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)),$$

    and the recognition model retains the conjugate exponential-family form with the learned correction $\\rho_Z$ in place of the analytic conjugation parameters,

    $$q(z \\mid x) = p(z; \\hat\\theta_{Z \\mid X}(x)), \\qquad \\hat\\theta_{Z \\mid X}(x) = \\theta_Z - \\rho_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}.$$

    Training jointly optimizes $(\\theta_Z, \\theta_X, \\Theta_{XZ}, \\rho_Z)$ under the ELBO, moving the model toward a regime in which the conjugate-form posterior is accurate.

    The four type parameters separate four roles:

    - ``Observable`` --- observable family; :class:`Differentiable` because $\\psi_X$ appears in the residual.
    - ``Posterior`` --- the latent family of the underlying harmonium, whose sufficient statistics the likelihood reads; the base class needs only :class:`Generative`.
    - ``Prior`` --- the family of $p(z)$: an exact family, possibly larger than ``Posterior``, or another variational harmonium (see :class:`VariationalPrior`). :attr:`pst_prr_emb` embeds ``Posterior`` into the prior's shift coordinates (:attr:`VariationalPrior.shift_man`).
    - ``Conjugation`` --- storage manifold for the correction. :meth:`conjugation_parameters` maps the stored coordinates to $\\rho_Z$ in the prior's shift coordinates; by default they are $\\rho_Z$ itself.

    The base class requires only sampling on ``Posterior`` and ``Prior``: it provides the model structure, the conjugation residual and joint sampling. The joint density, the ELBO and both residual variances evaluate latent log-densities, so they live on :class:`VariationalDifferentiable`.
    """

    # Contract

    @property
    @abstractmethod
    def gen_hrm(self) -> Harmonium[Observable, Posterior]:
        """The underlying harmonium; supplies the likelihood $p(x \\mid z)$ and the observable/posterior manifolds."""

    @property
    @abstractmethod
    def prr_man(self) -> Prior:
        """The family of $p(z; \\theta_Z)$: an exact family, or another variational harmonium."""

    @property
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[Any, Posterior]:
        """Embedding of the posterior manifold into the prior's shift coordinates: the prior itself for an exact prior, and the prior's observable for a variational one."""

    @property
    @abstractmethod
    def cnj_man(self) -> Conjugation:
        """The manifold in which $\\rho_Z$ is stored."""

    # Triple slots

    @property
    @override
    def fst_man(self) -> AffineMap[Observable, Posterior]:
        return self.gen_hrm.lkl_fun_man

    @property
    @override
    def snd_man(self) -> Prior:
        return self.prr_man

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

    # Parameter access

    def prior_params(self, params: Array) -> Array:
        """Extract the prior natural parameters $\\theta_Z$."""
        _, prior, _ = self.split_coords(params)
        return prior

    def likelihood_function(self, params: Array) -> Array:
        """Extract the affine likelihood parameters $(\\theta_X, \\Theta_{XZ})$."""
        lkl, _, _ = self.split_coords(params)
        return lkl

    # Approximate conjugation

    def conjugation_parameters(self, params: Array, x: Array | None = None) -> Array:
        """Map the stored ``Conjugation`` coordinates to the correction $\\rho_Z$ in the prior's shift coordinates, optionally as a function of the observation.

        The default treats the stored coordinates as $\\rho_Z$ itself and ignores ``x``. Subclasses override to apply a structural completion (e.g. mixture completion zero-pads the categorical interaction slots) or to evaluate a parametric input-dependent correction $\\rho_Z(x)$.
        """
        del x
        _, _, rho = self.split_coords(params)
        return rho

    def posterior_conjugation_parameters(self, params: Array, bias: Array) -> Array:
        """The conjugation parameters $\\rho^X$ of the likelihood with its observable bias replaced by ``bias``.

        Needed only when this harmonium is the prior of another. The base class stores only $\\rho^0$, so a subclass used as a prior must override this method.
        """
        del params, bias
        raise NotImplementedError(
            f"{type(self).__name__} is used as a prior but does not provide posterior conjugation parameters"
        )

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Natural parameters of the likelihood $p(x \\mid z)$ at the given latent state."""
        return self._likelihood_given(self.likelihood_function(params), z)

    def coupling(self, params: Array, x: Array) -> Array:
        """The term $\\mathbf s_X(x) \\cdot \\Theta_{XZ}$ that the observation adds to the posterior bias, in ``Posterior`` coordinates."""
        obs_params, int_params = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        hrm_params = self.gen_hrm.join_coords(
            obs_params, int_params, self.pst_man.zeros()
        )
        return self.gen_hrm.posterior_at(hrm_params, x)

    def observed_shift(self, params: Array, x: Array) -> Array:
        """The shift $\\delta(x) = \\mathbf s_X(x) \\cdot \\Theta_{XZ} - \\rho_Z(x)$ that conditions the prior on the observation."""
        return self.pst_prr_emb.embed(
            self.coupling(params, x)
        ) - self.conjugation_parameters(params, x)

    def recognition_at(self, params: Array, x: Array) -> Array:
        """Natural parameters $\\hat\\theta_{Z \\mid X}(x) = \\theta_Z + \\delta(x)$ of the recognition model, for an exact prior.

        $\\delta$ is the :meth:`observed_shift`. The result is in ``Prior`` coordinates, since the recognition model is the prior at shifted parameters.
        """
        return self.prior_params(params) + self.observed_shift(params, x)

    # Conjugation residual

    def conjugation_offset(self, params: Array) -> Array:
        """Compute the conjugation offset $\\chi$ at the given natural parameters.

        Returns $\\psi_X(\\theta_X)$, the constant that :meth:`conjugation_residual` adds and :meth:`conjugation_baseline` subtracts. Unlike :meth:`~goal.geometry.exponential_family.harmonium.Conjugated.conjugation_offset`, which enters the log-partition function, it only fixes where the ELBO integrand is split into $c(x) + r(z)$: it cancels in the ELBO, and the regularizers use only $\\mathrm{Var}[r]$, so no latent family needs to override it. It is the exact $\\chi$ whenever $\\mathbf s_Z$ has a zero, which makes $r$ vanish at exact conjugation.
        """
        lkl, _, _ = self.split_coords(params)
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(lkl)
        return self.obs_man.log_partition_function(obs_params)

    def conjugation_residual(
        self, params: Array, z: Array, x: Array | None = None
    ) -> Array:
        """Evaluate the conjugation residual $r(z) = \\rho_Z \\cdot \\mathbf s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) + \\psi_X(\\theta_X)$ at the given natural parameters.

        Mathematically, $r$ is the difference between the two sides of the conjugation equation $\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) = \\rho_Z \\cdot \\mathbf s_Z(z) + \\psi_X(\\theta_X)$, so $r$ is constant iff the likelihood is exactly conjugate with conjugation parameters $\\rho_Z$, and vanishes there when $\\mathbf s_Z$ has a zero (see :meth:`conjugation_offset`). It is the per-sample summand of the standard-form ELBO and the integrand of both conjugation regularizers (see module docstring).

        ``x`` is forwarded to :meth:`conjugation_parameters` for input-dependent corrections and ignored otherwise. $\\rho_Z$ and $\\mathbf s_Z(z)$ are paired in the prior's shift coordinates, via :meth:`conjugation_parameters` and :attr:`pst_prr_emb` respectively.
        """
        return self._residual(
            self.likelihood_function(params), self.conjugation_parameters(params, x), z
        )

    # Generative / ExponentialFamily contract on the joint $(x, z)$

    @property
    @override
    def data_dim(self) -> int:
        """Dimension of a joint datapoint: the observable's, plus the prior's."""
        return self.obs_man.data_dim + self.prr_man.data_dim

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

    # Initialization

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize full model parameters with zero conjugation correction.

        The likelihood is taken from an initialization of the underlying harmonium. An exact prior is that harmonium's latent bias, embedded into ``Prior``; a variational prior initializes itself. ``rho`` is zero.
        """
        rho = jnp.zeros(self.cnj_man.dim)
        hrm_params = self.gen_hrm.initialize(key, location, shape)
        obs_params, int_params, lat_params = self.gen_hrm.split_coords(hrm_params)
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        prior_params = self._initial_prior(key, lat_params, location, shape)
        return self.join_coords(lkl_params, prior_params, rho)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize parameters using sample data for observable biases."""
        rho = jnp.zeros(self.cnj_man.dim)
        hrm_params = self.gen_hrm.initialize_from_sample(key, sample, location, shape)
        obs_params, int_params, lat_params = self.gen_hrm.split_coords(hrm_params)
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        prior_params = self._initial_prior(key, lat_params, location, shape)
        return self.join_coords(lkl_params, prior_params, rho)

    # Internals

    def _latent_stats(self, z: Array) -> Array:
        """Sufficient statistics of the posterior variables, read as the leading slice of a datapoint of the prior."""
        return self.pst_man.sufficient_statistic(z[..., : self.pst_man.data_dim])

    def _likelihood_given(self, lkl_params: Array, z: Array) -> Array:
        return self.gen_hrm.lkl_fun_man(lkl_params, self._latent_stats(z))

    def _residual(self, lkl_params: Array, rho: Array, z: Array) -> Array:
        """The residual of the given likelihood with conjugation parameters ``rho``, at the prior datapoint ``z``."""
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(lkl_params)
        rho_term = jnp.dot(rho, self.pst_prr_emb.embed(self._latent_stats(z)))
        return (
            rho_term
            - self.obs_man.log_partition_function(self._likelihood_given(lkl_params, z))
            + self.obs_man.log_partition_function(obs_params)
        )

    def _initial_prior(
        self, key: Array, lat_params: Array, location: float, shape: float
    ) -> Array:
        if isinstance(self.prr_man, VariationalPrior):
            return self.prr_man.initialize(jax.random.fold_in(key, 1), location, shape)
        return self.pst_prr_emb.embed(lat_params)


class VariationalDifferentiable[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable | VariationalDifferentiable[Any, Any, Any, Any],
    Conjugation: Manifold,
](
    VariationalConjugated[Observable, Posterior, Prior, Conjugation],
    VariationalPrior,
    ABC,
):
    """Variational conjugation with closed-form log-partition functions, enabling the standard-form ELBO.

    ``Posterior`` is :class:`Differentiable`. ``Prior`` is an exact :class:`Differentiable` family or another :class:`VariationalDifferentiable`.

    Mathematically, the ELBO integrand splits as $f(x, z) = c(x) + r(z)$ (see module docstring); with $\\psi_Z$ available, the $x$-dependent piece $c(x)$ is computed analytically (:meth:`conjugation_baseline`) and only the residual is left to Monte Carlo (:meth:`elbo_at`). The same machinery yields the closed-form KL between recognition model and prior (:meth:`elbo_divergence`) and the recognition-side conjugation regularizer (:meth:`recognition_residual_variance_at`).

    With a variational prior the model is nested, and the latent data $w$ are all levels of the prior. The recognition model $q(w \\mid x)$ is the prior conditioned by the :meth:`observed_shift`. A nested harmonium conditioned by a shift $\\delta$ has the posterior bias $\\hat\\theta_X = \\theta_X + \\delta$, and passes the shift $\\rho^X - \\rho^0$ to its own prior, where $\\rho^X$ and $\\rho^0$ are its conjugation parameters at $\\hat\\theta_X$ and $\\theta_X$. The integrand then splits as

    $$\\log p(x, w) - \\log q(w \\mid x) = c(x) + \\sum r^0 - \\sum r^X,$$

    where $r^0$ is the residual of each harmonium at $\\theta_X$, $r^X$ the residual of each nested harmonium at $\\hat\\theta_X$, and $c(x)$ the log-marginal the model would have if every conjugation were exact.

    Mirrors :class:`DifferentiableConjugated` on the analytic side.
    """

    # Overrides

    @property
    @override
    def shift_man(self) -> Observable:
        """A harmonium used as a prior is shifted in its observable bias."""
        return self.obs_man

    @property
    @override
    def level_dims(self) -> tuple[int, ...]:
        return (self.obs_man.data_dim, *self._prior.level_dims)

    @property
    @override
    def level_learned(self) -> tuple[tuple[bool, ...], ...]:
        return ((self.learned_conjugation,), *self._prior.level_learned)

    @override
    def conditional_sample(
        self, key: Array, params: Array, shift: Array | None
    ) -> list[Array]:
        key_prior, key_root = jax.random.split(key)
        levels = self._prior.conditional_sample(
            key_prior, self.prior_params(params), self._passed_shift(params, shift)
        )
        lkl_params = self._likelihood_given(
            self._shifted_likelihood(params, shift), levels[0]
        )
        return [self.obs_man.sample(key_root, lkl_params, 1)[0], *levels]

    @override
    def conditional_log_densities(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[Array]:
        lkl_params = self._likelihood_given(
            self._shifted_likelihood(params, shift), levels[1]
        )
        deep = self._prior.conditional_log_densities(
            self.prior_params(params), self._passed_shift(params, shift), levels[1:]
        )
        return [self.obs_man.log_density(lkl_params, levels[0]), *deep]

    @override
    def conditional_parameters(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[Array]:
        lkl_params = self._likelihood_given(
            self._shifted_likelihood(params, shift), levels[1]
        )
        deep = self._prior.conditional_parameters(
            self.prior_params(params), self._passed_shift(params, shift), levels[1:]
        )
        return [lkl_params, *deep]

    @override
    def conditional_residuals(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[tuple[tuple[Array, ...], tuple[Array, ...]]]:
        z = levels[1]
        r0 = (self.conjugation_residual(params, z),)
        rx: tuple[Array, ...] = ()
        if shift is not None:
            rx = (
                self._residual(
                    self._shifted_likelihood(params, shift),
                    self._posterior_conjugation(params, shift),
                    z,
                ),
            )
        deep = self._prior.conditional_residuals(
            self.prior_params(params), self._passed_shift(params, shift), levels[1:]
        )
        return [(r0, rx), *deep]

    @override
    def log_partition_shift(self, params: Array, shift: Array) -> Array:
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        deep = self._prior.log_partition_shift(
            self.prior_params(params),
            self._posterior_conjugation(params, shift)
            - self.conjugation_parameters(params),
        )
        return (
            self.obs_man.log_partition_function(obs_params + shift)
            - self.obs_man.log_partition_function(obs_params)
            + deep
        )

    # Methods

    @property
    def learned_conjugation(self) -> bool:
        """Whether the conjugation parameters are learned. A subclass that computes them exactly returns ``False``; its residual is then zero, and :meth:`elbo_at` leaves it out of the score term."""
        return True

    # Joint density

    def log_density(self, params: Array, xz: Array) -> Array:
        """Compute joint log density $\\log p(x, z) = \\log p(z) + \\log p(x|z)$ for a joint data point."""
        x = xz[..., : self.obs_man.data_dim]
        z = xz[..., self.obs_man.data_dim :]
        prior = self.prior_params(params)
        log_pz = self.prr_man.log_density(prior, z)
        lkl_params = self.likelihood_at(params, z)
        log_px_given_z = self.obs_man.log_density(lkl_params, x)
        return log_pz + log_px_given_z

    # Recognition model

    def recognition_shift(self, params: Array, x: Array) -> Array:
        """The shift that conditions the prior on ``x``.

        For a variational prior this is the :meth:`observed_shift`. For an exact prior it is computed from :meth:`recognition_at`, so that a subclass overriding :meth:`recognition_at` (e.g. with a clamp) changes every estimator.
        """
        if isinstance(self.prr_man, VariationalPrior):
            return self.observed_shift(params, x)
        return self.recognition_at(params, x) - self.prior_params(params)

    def sample_recognition(self, key: Array, params: Array, x: Array, n: int) -> Array:
        """Samples of every latent level from $q(w \\mid x)$, as rows, bottom level first."""
        prior, shift = self.prior_params(params), self.recognition_shift(params, x)

        def draw(k: Array) -> Array:
            return jnp.concatenate(self._prior.conditional_sample(k, prior, shift))

        return jax.vmap(draw)(jax.random.split(key, n))

    def recognition_conditionals(
        self, params: Array, x: Array, w: Array
    ) -> list[Array]:
        """Natural parameters of each latent level's recognition conditional, bottom first, given the levels above it in ``w``. The last entry depends on $x$ alone."""
        levels = split_by_dims(w, self._prior.level_dims)
        return self._prior.conditional_parameters(
            self.prior_params(params), self.recognition_shift(params, x), levels
        )

    def recognition_log_density(self, params: Array, x: Array, w: Array) -> Array:
        """$\\log q(w \\mid x)$, summed over the latent levels."""
        levels = split_by_dims(w, self._prior.level_dims)
        return _total(
            self._prior.conditional_log_densities(
                self.prior_params(params), self.recognition_shift(params, x), levels
            )
        )

    # ELBO

    def conjugation_baseline(self, params: Array, x: Array) -> Array:
        """The latent-independent ELBO term $c(x) = \\mathbf s_X(x) \\cdot \\theta_X + \\psi_Z(\\hat\\theta_{Z \\mid X}(x)) - \\psi_Z(\\theta_Z) - \\psi_X(\\theta_X) + \\log h_X(x)$ at the given natural parameters; with a variational prior, $\\psi_Z(\\hat\\theta_{Z \\mid X}(x)) - \\psi_Z(\\theta_Z)$ is the prior's :meth:`~VariationalPrior.log_partition_shift`.

        Mathematically, $c(x)$ is what remains of the ELBO integrand $f(x, z) = \\log p(x, z) - \\log q(z \\mid x)$ after the $z$-dependent terms are collected into the residual: substituting the exponential-family forms cancels the shared $\\mathbf s_Z \\cdot \\theta_Z$ and $\\mathbf s_X \\cdot \\Theta_{XZ} \\cdot \\mathbf s_Z$ terms. The $\\mp \\psi_X(\\theta_X)$ terms here and in :meth:`VariationalConjugated.conjugation_residual` are a matched pair, so the $\\chi$ convention cancels in $c(x) + r(z)$ and the ELBO does not depend on it.

        $c(x)$ is also the formula of :meth:`DifferentiableConjugated.log_observable_density` with the learned $\\rho_Z$ in place of the analytic conjugation parameters --- the log-marginal the model would have if conjugation were exact. The ELBO $\\mathcal{L}(x) = c(x) + \\mathbb{E}_q[r]$ accordingly collapses to $\\log p(x)$ when the model is exactly conjugated; in general $\\log p(x) = c(x) + \\mathbb{E}_q[r] + \\mathrm{KL}(q \\Vert p(z \\mid x))$.
        """
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        return (
            jnp.dot(self.obs_man.sufficient_statistic(x), obs_params)
            - self.conjugation_offset(params)
            + self.obs_man.log_base_measure(x)
            + self._prior.log_partition_shift(
                self.prior_params(params), self.recognition_shift(params, x)
            )
        )

    def conjugation_residuals(
        self, params: Array, w: Array, x: Array
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        """The residuals at the latent data ``w``, given the observation ``x``: $r^0$ of each harmonium in the model, and $r^X$ of each harmonium nested in the prior, bottom first.

        The ELBO integrand is :meth:`conjugation_baseline` plus $\\sum r^0 - \\sum r^X$.
        """
        levels = split_by_dims(w, self._prior.level_dims)
        per_level = self._recognition_residuals(
            params,
            x,
            self.prior_params(params),
            self.recognition_shift(params, x),
            levels,
        )
        r0 = tuple(r for level_r0, _ in per_level for r in level_r0)
        rx = tuple(r for _, level_rx in per_level for r in level_rx)
        return r0, rx

    def elbo_at(
        self,
        key: Array,
        params: Array,
        x: Array,
        n_samples: int,
    ) -> Array:
        """Estimate the standard-form ELBO $\\mathcal{L}(x) = c(x) + \\mathbb{E}_{q}[\\sum r^0 - \\sum r^X]$: the baseline analytically, the residual term by Monte Carlo.

        The split absorbs $c(x)$ as an exact $x$-dependent baseline, so only the residual is sampled and only it drives the score-function correction,

        $$\\nabla \\mathcal{L}(x) = \\mathbb{E}_q[\\nabla \\log p(x, W)] + \\mathbb{E}_q[(r - b) \\nabla \\log q],$$

        where the score term accounts for the dependence of the sampling distribution $q$ on the parameters and the leave-one-out baseline $b$ reduces finite-sample variance without biasing the gradient (:func:`score_mean_estimate`). The score term includes only the levels at and above the lowest level with a learned residual, since the levels below do not affect any learned residual. With every conjugation exact the estimate is $c(x)$, and nothing is sampled. The samples are stop-gradiented: the sampler may be non-differentiable (e.g. VonMises rejection), so they are evaluation points, not gradient carriers.
        """
        baseline = self.conjugation_baseline(params, x)
        learned = [any(flags) for flags in self.level_learned]
        if not any(learned):
            return baseline
        start = learned.index(True)
        prior, shift = self.prior_params(params), self.recognition_shift(params, x)

        def draw(k: Array) -> list[Array]:
            return [
                jax.lax.stop_gradient(lv)
                for lv in self._prior.conditional_sample(k, prior, shift)
            ]

        def terms(w: list[Array]) -> tuple[Array, Array]:
            per_level = self._recognition_residuals(params, x, prior, shift, w)
            r = _total([_total(r0) - _total(rx) for r0, rx in per_level])
            log_q = self._prior.conditional_log_densities(prior, shift, w)
            return r, _total(log_q[start:])

        samples = jax.vmap(draw)(jax.random.split(key, n_samples))
        r_vals, log_q_vals = jax.vmap(terms)(samples)
        return baseline + score_mean_estimate(r_vals, log_q_vals)

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
        """Closed-form $\\mathrm{KL}(q(z \\mid x) \\Vert p(z))$ at the given natural parameters, for an exact prior.

        Not used by :meth:`elbo_at` --- the standard form bundles the KL into $c(x)$ and $\\mathbb{E}_q[r]$. Exposed for $\\beta$-VAE-style warmup (callers add $(1 - \\beta) \\cdot \\mathrm{KL}$ to the ELBO) and as a diagnostic in its own right.
        """
        prr_man = cast(Differentiable, self.prr_man)
        return prr_man.relative_entropy(
            self.recognition_at(params, x), self.prior_params(params)
        )

    # Conjugation regularizers

    def prior_residual_variance(
        self, key: Array, params: Array, n_samples: int
    ) -> Array:
        """Prior conjugation regularizer $\\mathcal{R}_p = \\mathrm{Var}_{p(z)}[r(Z)]$ of this harmonium's own residual, estimated over prior samples.

        $x$-independent, matching the intuition that conjugation is a property of the likelihood and prior alone --- though the penalty may therefore act on regions of latent space that inference never visits.

        The sampling distribution $p(z)$ depends on $\\theta_Z$, so the gradient has a score-function piece in $\\theta_Z$ besides the direct piece through $r$; both are estimated without bias by :func:`score_variance_estimate`, which needs ``n_samples >= 2``.
        """
        prior = self.prior_params(params)

        def draw(k: Array) -> tuple[list[Array], Array]:
            w = [
                jax.lax.stop_gradient(lv)
                for lv in self._prior.conditional_sample(k, prior, None)
            ]
            return w, _total(self._prior.conditional_log_densities(prior, None, w))

        w, log_p_vals = jax.vmap(draw)(jax.random.split(key, n_samples))
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z))(w[0])
        return score_variance_estimate(r_vals, log_p_vals)

    def prior_residual_variances(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """$\\mathrm{Var}_p[r^0]$ over ancestral samples, for each harmonium in the model, bottom first; zero where the conjugation parameters are exact.

        The distribution of each residual's latent level depends on the parameters of every level above it, so the gradient includes a score-function term through their log-densities (:func:`score_variance_estimate`). Needs ``n_samples >= 2``.
        """

        def draw(k: Array) -> tuple[list[Array], list[Array]]:
            levels = [
                jax.lax.stop_gradient(lv)
                for lv in self.conditional_sample(k, params, None)
            ]
            return levels, self.conditional_log_densities(params, None, levels)

        levels, log_p = jax.vmap(draw)(jax.random.split(key, n_samples))
        residuals = jax.vmap(lambda lv: self.conditional_residuals(params, None, lv))(
            levels
        )
        return _residual_variances(
            self.level_learned, [r0 for r0, _ in residuals], log_p[1:]
        )

    def recognition_residual_variance_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """Recognition conjugation regularizer $\\mathcal{R}_q(x) = \\mathrm{Var}_{q(z \\mid x)}[r(Z)]$ of this harmonium's own residual, estimated over recognition samples.

        Mathematically, for an exact prior $\\mathrm{Var}_q[r(Z)] = \\mathrm{Var}_q[\\log (q(Z \\mid x) / p(Z \\mid x))]$, so $\\mathcal{R}_q(x)$ measures the inference gap at $x$: it vanishes iff the recognition model matches the true posterior. It focuses the penalty on the latents inference actually visits, and can reuse the samples drawn for the ELBO.

        As for :meth:`prior_residual_variance`, the gradient has a direct piece through $r$ and a score-function piece through the sampling distribution, here $q$ --- see :func:`score_variance_estimate`.
        """
        prior, shift = self.prior_params(params), self.recognition_shift(params, x)

        def draw(k: Array) -> tuple[list[Array], Array]:
            w = [
                jax.lax.stop_gradient(lv)
                for lv in self._prior.conditional_sample(k, prior, shift)
            ]
            return w, _total(self._prior.conditional_log_densities(prior, shift, w))

        w, log_q_vals = jax.vmap(draw)(jax.random.split(key, n_samples))
        r_vals = jax.vmap(lambda z: self.conjugation_residual(params, z, x))(w[0])
        return score_variance_estimate(r_vals, log_q_vals)

    def mean_recognition_residual_variance(
        self, key: Array, params: Array, xs: Array, n_samples: int
    ) -> Array:
        """Mean of :meth:`recognition_residual_variance_at` over a batch."""
        batch_size = xs.shape[0]
        keys = jax.random.split(key, batch_size)
        losses = jax.vmap(
            lambda k, x: self.recognition_residual_variance_at(k, params, x, n_samples)
        )(keys, xs)
        return jnp.mean(losses)

    def inner_residual_variances_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """$\\mathrm{Var}_q[r^X]$ over recognition samples, for each harmonium nested in the prior; zero where the conjugation parameters are exact, and empty for an exact prior.

        These variances train the posterior conjugation parameters $\\rho^X$. As in :meth:`prior_residual_variances`, the gradient includes a score-function term. Needs ``n_samples >= 2``.
        """
        prior, shift = self.prior_params(params), self.recognition_shift(params, x)

        def draw(k: Array) -> tuple[list[Array], list[Array]]:
            levels = [
                jax.lax.stop_gradient(lv)
                for lv in self._prior.conditional_sample(k, prior, shift)
            ]
            return levels, self._prior.conditional_log_densities(prior, shift, levels)

        levels, log_q = jax.vmap(draw)(jax.random.split(key, n_samples))
        residuals = jax.vmap(
            lambda lv: self._prior.conditional_residuals(prior, shift, lv)
        )(levels)
        return _residual_variances(
            self._prior.level_learned, [rx for _, rx in residuals], log_q[1:]
        )

    def mean_inner_residual_variances(
        self, key: Array, params: Array, xs: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """Mean of :meth:`inner_residual_variances_at` over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        vals = jax.vmap(
            lambda k, x: self.inner_residual_variances_at(k, params, x, n_samples)
        )(keys, xs)
        return tuple(jnp.mean(v) for v in vals)

    # Internals

    @property
    def _prior(self) -> VariationalPrior:
        """The prior as a :class:`VariationalPrior`; an exact prior is wrapped."""
        prr_man = self.prr_man
        if isinstance(prr_man, VariationalPrior):
            return prr_man
        return _ExactPrior(prr_man)

    def _shifted_likelihood(self, params: Array, shift: Array | None) -> Array:
        """The likelihood with its observable bias shifted by ``shift``."""
        lkl_params = self.likelihood_function(params)
        if shift is None:
            return lkl_params
        obs_params, int_params = self.gen_hrm.lkl_fun_man.split_coords(lkl_params)
        return self.gen_hrm.lkl_fun_man.join_coords(obs_params + shift, int_params)

    def _posterior_conjugation(self, params: Array, shift: Array) -> Array:
        """$\\rho^X$ at the observable bias shifted by ``shift``."""
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        return self.posterior_conjugation_parameters(params, obs_params + shift)

    def _passed_shift(self, params: Array, shift: Array | None) -> Array | None:
        """The shift $\\rho^X - \\rho^0$ passed to the prior when this harmonium is conditioned by ``shift``."""
        if shift is None:
            return None
        return self._posterior_conjugation(params, shift) - self.conjugation_parameters(
            params
        )

    def _recognition_residuals(
        self, params: Array, x: Array, prior: Array, shift: Array, w: Sequence[Array]
    ) -> list[tuple[tuple[Array, ...], tuple[Array, ...]]]:
        """This harmonium's $r^0$ given the observation, then the prior's residuals under the recognition shift."""
        deep = self._prior.conditional_residuals(prior, shift, w)
        return [((self.conjugation_residual(params, w[0], x),), ()), *deep]


class VariationalSymmetric[
    Observable: Differentiable,
    Latent: Differentiable,
    Conjugation: Manifold,
](
    VariationalDifferentiable[Observable, Latent, Latent, Conjugation],
    ABC,
):
    """Variational conjugation in the symmetric case where Posterior and Prior share the same manifold (``pst_man == prr_man``).

    Mirrors :class:`SymmetricConjugated` on the analytic side: extends :class:`VariationalDifferentiable` and provides ``prr_man`` and ``pst_prr_emb`` (an :class:`IdentityEmbedding`) so subclasses only need to supply ``gen_hrm``, ``lat_man``, ``cnj_man``, and ``conjugation_parameters``.
    """

    # Contract

    @property
    @abstractmethod
    def lat_man(self) -> Latent:
        """The shared posterior/prior manifold."""

    # Overrides

    @property
    @override
    def prr_man(self) -> Latent:
        return self.lat_man

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Latent]:
        return IdentityEmbedding(self.lat_man)


@dataclass(frozen=True)
class _ExactPrior(VariationalPrior):
    """An exact family as a prior: one level, conditioned by adding the shift to its natural parameters."""

    model: Differentiable

    @property
    @override
    def shift_man(self) -> Differentiable:
        return self.model

    @property
    @override
    def level_dims(self) -> tuple[int, ...]:
        return (self.model.data_dim,)

    @property
    @override
    def level_learned(self) -> tuple[tuple[bool, ...], ...]:
        return ()

    @override
    def conditional_sample(
        self, key: Array, params: Array, shift: Array | None
    ) -> list[Array]:
        return [self.model.sample(key, _shifted(params, shift), 1)[0]]

    @override
    def conditional_log_densities(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[Array]:
        return [self.model.log_density(_shifted(params, shift), levels[0])]

    @override
    def conditional_parameters(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[Array]:
        del levels
        return [_shifted(params, shift)]

    @override
    def conditional_residuals(
        self, params: Array, shift: Array | None, levels: Sequence[Array]
    ) -> list[tuple[tuple[Array, ...], tuple[Array, ...]]]:
        del params, shift, levels
        return []

    @override
    def log_partition_shift(self, params: Array, shift: Array) -> Array:
        return self.model.log_partition_function(
            params + shift
        ) - self.model.log_partition_function(params)


def _shifted(params: Array, shift: Array | None) -> Array:
    return params if shift is None else params + shift


def _total(vals: Sequence[Array]) -> Array:
    return sum(vals, start=jnp.zeros(()))


def _residual_variances(
    level_learned: tuple[tuple[bool, ...], ...],
    residuals: Sequence[tuple[Array, ...]],
    log_densities: Sequence[Array],
) -> tuple[Array, ...]:
    """The variance of each harmonium's sampled residuals, with a score-function term through the log-densities of its latent level and every level above; zero where the conjugation parameters are exact.

    The latent level of harmonium $k$ is entry $k$ of ``log_densities``.
    """
    out: list[Array] = []
    for k, (flags, rs) in enumerate(zip(level_learned, residuals)):
        above = _total(log_densities[k:])
        out.extend(
            score_variance_estimate(r, above) if learned else jnp.zeros(())
            for learned, r in zip(flags, rs)
        )
    return tuple(out)


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
    lkl, prior_params, _ = model.split_coords(params)
    cnj_dim = model.cnj_man.dim
    zero_rho = jnp.zeros(cnj_dim)
    params_zero_rho = model.join_coords(lkl, prior_params, zero_rho)

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
            trial_params = model.join_coords(lkl, prior_params, r)
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
    r_squared = 1.0 - ss_res / ss_tot

    return rho, r_squared, chi, jnp.var(residuals)


def conjugation_metrics[
    Observable: Differentiable,
    Posterior: Differentiable,
    Prior: Differentiable,
    Conjugation: Manifold,
](
    model: VariationalDifferentiable[Observable, Posterior, Prior, Conjugation],
    key: Array,
    params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array]:
    """Compute conjugation quality metrics ``(var_f, std_f, r_squared)`` under the prior.

    $R^2 = 1 - \\mathrm{Var}_p[r] / \\mathrm{Var}_p[\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(Z))]$ measures how much of the variation of the log-partition the affine correction explains: 1 = exact conjugation, 0 = no better than a constant, < 0 = worse. Both variances are shift-invariant, so the $\\psi_X(\\theta_X)$ convention in :meth:`conjugation_residual` does not affect the metric.
    """
    var_key, psi_key = jax.random.split(key)
    var_f = model.prior_residual_variance(var_key, params, n_samples)
    std_f = jnp.sqrt(var_f)

    lkl, prior_params, _ = model.split_coords(params)
    z_samples = model.prr_man.sample(psi_key, prior_params, n_samples)

    def psi_at(z: Array) -> Array:
        s_z = model.pst_man.sufficient_statistic(z)
        lkl_params = model.gen_hrm.lkl_fun_man(lkl, s_z)
        return model.obs_man.log_partition_function(lkl_params)

    psi_vals = jax.vmap(psi_at)(z_samples)
    var_psi = jnp.var(psi_vals, ddof=1)
    return var_f, std_f, 1.0 - var_f / var_psi


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

    Requires an exact ``Prior`` with a closed-form ``to_mean``; the recognition model's mean parameters are projected onto ``Posterior`` through :attr:`~VariationalConjugated.pst_prr_emb`.
    """
    q_params = model.recognition_at(params, x)
    z_mean_stats = model.pst_prr_emb.project(model.prr_man.to_mean(q_params))
    lkl, _, _ = model.split_coords(params)
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
