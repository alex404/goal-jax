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
function of the likelihood alone. The model and its recognition model are

$$\\tilde p(x, z) = \\tilde p_Z(z; \\theta_Z + \\rho) \\, p(x \\mid z; \\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)), \\qquad q(z \\mid x) = \\tilde p_Z(z; \\theta_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}),$$

where the deep model $\\tilde p_Z$ is an exact family or again a variational model. When
$\\rho$ is exact, $\\tilde p$ is the density of the graphical harmonium. The model is a
directed variational autoencoder in other coordinates: the directed prior bias of each
level is $\\theta_Y + \\rho_Y$, with likelihoods and interactions shared.

The central quantity is the **conjugation residual** of each level,

$$r(z) = \\delta \\cdot \\mathbf s_Z(z) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) + \\psi_X(\\theta_X),$$

with the slope $\\delta = \\rho$ in the model. $r$ is constant, and its variance zero, if and
only if $\\rho$ is exact. Summed over the levels, it gives

$$\\log \\tilde p(x, z) - \\log q(z \\mid x) = c(x) + r(z) + \\sum r^0(z) - \\sum r^X(z),$$

where $c(x)$ is the log-marginal under exact conjugation
(:meth:`VariationalConjugated.conjugation_baseline`), $r$ is the residual of this level, and
$r^0$ and $r^X$ are the residuals of the deep model at the prior and at the posterior.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
from jax import Array

from ..manifold.base import Manifold
from ..manifold.combinators import Pair, Tuple
from ..manifold.embedding import LinearEmbedding
from ..manifold.util import split_by_dims
from .base import Differentiable
from .combinators import DifferentiableTuple
from .graphical import GraphicalHarmonium


@dataclass(frozen=True)
class ConjugationTuple(Tuple):
    """The conjugation function parameters of a variational model, one element per level, bottom first: a plain product of the levels' :attr:`~VariationalConjugated.cnj_fun_man`.

    A level whose conjugation parameters are computed rather than stored has a
    zero-dimensional element, so element $k$ is always level $k$.
    """

    # Fields

    elm_mans: tuple[Manifold, ...]
    """The conjugation function parameters of each level, bottom first."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return sum(elm.dim for elm in self.elm_mans)

    @override
    def split_coords(self, coords: Array) -> tuple[Array, ...]:
        """Split into the conjugation function parameters of each level."""
        return split_by_dims(coords, tuple(elm.dim for elm in self.elm_mans))

    @override
    def join_coords(self, *components: Array) -> Array:
        """Concatenate the conjugation function parameters of each level."""
        return jnp.concatenate(components)


class DeepModel[Prior: Manifold](Pair[Prior, ConjugationTuple], ABC):
    """A model that can be the deep model of a variational model, the model of the latent above a level.

    Its parameters are ``[prr_params | cnj_fun_tup_params]``: natural parameters of the prior
    family, and the conjugation function parameters of its levels. It is a variational model
    (:class:`VariationalConjugated`), or an exponential family with no levels
    (:class:`DifferentiableDeepModel`).
    """

    # Contract

    @property
    @abstractmethod
    def data_dim(self) -> int:
        """Dimension of a datapoint."""

    @abstractmethod
    def conjugated_log_partition_function(self, params: Array) -> Array:
        """$\\tilde\\Psi$: the log-partition function under exact conjugation, the exact one for an exponential family."""

    @abstractmethod
    def sample(self, key: Array, params: Array, n: int) -> Array:
        """Ancestral samples."""

    @abstractmethod
    def log_density(self, params: Array, xz: Array) -> Array:
        """Log-density of a datapoint."""

    @abstractmethod
    def conjugation_residuals(self, params: Array, xz: Array) -> tuple[Array, ...]:
        """The residual of each level at a datapoint, bottom first."""

    @abstractmethod
    def conjugation_residual_variances(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """Monte Carlo estimate of the variance of the residual of each level under the model's own samples, bottom first."""

    # Methods

    def score_mean(
        self,
        key: Array,
        params: Array,
        f: Callable[[Array], Array],
        n_samples: int,
    ) -> Array:
        """Monte Carlo estimate of $\\mathbb E[f(Z)]$ under the model, with an unbiased gradient.

        ``f`` is a function of a datapoint that may depend on the parameters, e.g. a
        residual. The samples carry no gradient, since the sampler may be
        non-differentiable, so a score-function term $\\frac{1}{K} \\sum_k (f_k - b_k) \\nabla
        \\log p(z_k)$ with the leave-one-out baseline $b_k$ supplies that part of the gradient.
        """
        zs = jax.lax.stop_gradient(
            self.sample(key, jax.lax.stop_gradient(params), n_samples)
        )
        vals = jax.vmap(f)(zs)
        log_dens = jax.vmap(lambda z: self.log_density(params, z))(zs)
        vals_sg = jax.lax.stop_gradient(vals)
        centered = (
            vals_sg
            if n_samples == 1
            else n_samples / (n_samples - 1) * (vals_sg - jnp.mean(vals_sg))
        )
        score = jnp.mean(centered * log_dens)
        return jnp.mean(vals) + score - jax.lax.stop_gradient(score)

    def score_variance(
        self,
        key: Array,
        params: Array,
        f: Callable[[Array], Array],
        n_samples: int,
    ) -> Array:
        """Monte Carlo estimate of $\\mathrm{Var}[f(Z)]$ under the model, with an unbiased gradient; as :meth:`score_mean`, and needs ``n_samples >= 2``.

        Mathematically, $\\mathrm{Var}[f] = \\frac{1}{2} \\mathbb E[(f(Z) - f(Z'))^2]$, estimated by
        the mean over pairs of samples, with the score term of the pair.
        """
        zs = jax.lax.stop_gradient(
            self.sample(key, jax.lax.stop_gradient(params), n_samples)
        )
        vals = jax.vmap(f)(zs)
        log_dens = jax.vmap(lambda z: self.log_density(params, z))(zs)
        pairs = 0.5 * (vals[:, None] - vals[None, :]) ** 2
        score = jnp.sum(
            jax.lax.stop_gradient(pairs) * (log_dens[:, None] + log_dens[None, :])
        )
        return (jnp.sum(pairs) + score - jax.lax.stop_gradient(score)) / (
            n_samples * (n_samples - 1)
        )


@dataclass(frozen=True)
class DifferentiableDeepModel[Prior: Differentiable](DeepModel[Prior]):
    """A differentiable exponential family as a deep model. Nothing is attached above it, so it has no levels: no conjugation function parameters and no residuals."""

    # Fields

    _fst_man: Prior

    # Overrides

    @property
    @override
    def fst_man(self) -> Prior:
        return self._fst_man

    @property
    @override
    def snd_man(self) -> ConjugationTuple:
        return ConjugationTuple(())

    @property
    @override
    def data_dim(self) -> int:
        return self.fst_man.data_dim

    @override
    def conjugated_log_partition_function(self, params: Array) -> Array:
        return self.fst_man.log_partition_function(params)

    @override
    def sample(self, key: Array, params: Array, n: int) -> Array:
        return self.fst_man.sample(key, params, n)

    @override
    def log_density(self, params: Array, xz: Array) -> Array:
        return self.fst_man.log_density(params, xz)

    @override
    def conjugation_residuals(self, params: Array, xz: Array) -> tuple[Array, ...]:
        return ()

    @override
    def conjugation_residual_variances(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, ...]:
        return ()


class VariationalConjugated[
    Graphical: GraphicalHarmonium[Any],
    Conjugation: Manifold,
](
    DeepModel[Graphical],
    ABC,
):
    """A graphical harmonium with conjugation parameters, approximating the operations of a conjugated harmonium.

    Its parameters are ``[hrm_params | cnj_fun_tup_params]``: the natural parameters of
    :attr:`gen_hrm`, and the conjugation function parameters of every level
    (:class:`ConjugationTuple`), this level's (:attr:`cnj_fun_man`) first and the deep model's
    after them. It is not an exponential family: the conjugation function parameters parameterize
    the approximation, not the joint density.

    A subclass declares the graphical harmonium, the conjugation function parameters and how they
    give $\\rho$ (:meth:`conjugation_parameters`), and the deep model, the model of the
    latent above this level (:attr:`dep_man`): an exponential family in
    :class:`DifferentiableVariationalConjugated`, or another variational model, whose graphical
    harmonium is the prior family. A variational model is itself a :class:`DeepModel`, so
    levels nest to any depth.
    """

    # Contract

    @property
    @abstractmethod
    def gen_hrm(self) -> Graphical:
        """The graphical harmonium."""

    @property
    @abstractmethod
    def cnj_fun_man(self) -> Conjugation:
        """The manifold of the parameters of this level's conjugation function; those of the levels above are the rest of :attr:`snd_man`."""

    @property
    @abstractmethod
    def pst_prr_emb(self) -> LinearEmbedding[Any, Any]:
        """Embedding of the posterior of :attr:`gen_hrm` into the prior family."""

    @abstractmethod
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        """The conjugation function: the conjugation parameters $\\rho$ of the given likelihood natural parameters, in natural coordinates of the prior family.

        A function of the likelihood alone, so the prior and the recognition model of a
        nested model use the same one. Its parameters may be none (exact conjugation
        parameters, where an attached harmonium is conjugated), $\\rho$ itself (a constant),
        or those of a learned approximation.
        """

    @property
    @abstractmethod
    def dep_man(self) -> DeepModel[Any]:
        """The deep model, whose prior family is the ambient of :attr:`pst_prr_emb`."""

    # Overrides

    @property
    @override
    def fst_man(self) -> Graphical:
        return self.gen_hrm

    @property
    @override
    def snd_man(self) -> ConjugationTuple:
        return ConjugationTuple((self.cnj_fun_man, *self.dep_man.snd_man.elm_mans))

    @property
    @override
    def data_dim(self) -> int:
        """Dimension of a joint datapoint $(x, z)$."""
        return self.obs_man.data_dim + self.dep_man.data_dim

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

    def likelihood_function(self, params: Array) -> Array:
        """The likelihood natural parameters $(\\theta_X, \\Theta_{XZ})$."""
        hrm_params, _ = self.split_coords(params)
        return self.gen_hrm.likelihood_function(hrm_params)

    def likelihood_at(self, params: Array, z: Array) -> Array:
        """Natural parameters of $p(x \\mid z)$ at a datapoint of the prior family, whose leading slice is the posterior's datapoint."""
        s_z = self.pst_man.sufficient_statistic(z[..., : self.pst_man.data_dim])
        return self.gen_hrm.lkl_fun_man(self.likelihood_function(params), s_z)

    def conjugated_prior_params(self, params: Array) -> Array:
        """Parameters of the deep model at the prior, whose prior-family part is $\\theta_Z + \\rho$: the prior of :attr:`gen_hrm` under conjugation by $\\rho$, exact only when $\\rho$ is.

        This level's conjugation function is evaluated here, and only here. The deep model
        evaluates its own, at the parameters it is given: these, or those of
        :meth:`recognition_at`.
        """
        hrm_params, cnj_fun_tup_params = self.split_coords(params)
        cnj_fun_params, *_ = self.snd_man.split_coords(cnj_fun_tup_params)
        obs_params, int_params, lat_params = self.gen_hrm.split_coords(hrm_params)
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        rho = self.conjugation_parameters(lkl_params, cnj_fun_params)
        return self.dep_man.join_coords(
            self.pst_prr_emb.translate(rho, lat_params),
            cnj_fun_tup_params[self.cnj_fun_man.dim :],
        )

    def posterior_at(self, params: Array, x: Array) -> Array:
        """Posterior natural parameters $\\theta_Z + \\mathbf s_X(x) \\cdot \\Theta_{XZ}$ of :attr:`gen_hrm`."""
        hrm_params, _ = self.split_coords(params)
        return self.gen_hrm.posterior_at(hrm_params, x)

    def recognition_at(self, params: Array, x: Array) -> Array:
        """Parameters of the deep model at the posterior: the recognition model $q(z \\mid x)$."""
        _, cnj_fun_tup_params = self.split_coords(params)
        return self.dep_man.join_coords(
            self.pst_prr_emb.embed(self.posterior_at(params, x)),
            cnj_fun_tup_params[self.cnj_fun_man.dim :],
        )

    @override
    def conjugated_log_partition_function(self, params: Array) -> Array:
        """$\\tilde\\Psi(\\theta) = \\tilde\\Psi_Z(\\theta_Z + \\rho) + \\psi_X(\\theta_X)$: the log-partition function of :attr:`gen_hrm` under exact conjugation.

        It differs from the exact one by $\\psi(\\theta) - \\tilde\\Psi(\\theta) = \\log \\mathbb E_{\\tilde p}[e^{-\\sum r(Z)}]$.
        """
        obs_params, _ = self.gen_hrm.lkl_fun_man.split_coords(
            self.likelihood_function(params)
        )
        return self.dep_man.conjugated_log_partition_function(
            self.conjugated_prior_params(params)
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
        dep_prior = self.conjugated_prior_params(params)
        dep_post = self.recognition_at(params, x)
        return (
            jnp.dot(self.obs_man.sufficient_statistic(x), obs_params)
            + self.obs_man.log_base_measure(x)
            - self.obs_man.log_partition_function(obs_params)
            + self.dep_man.conjugated_log_partition_function(dep_post)
            - self.dep_man.conjugated_log_partition_function(dep_prior)
        )

    @override
    def log_density(self, params: Array, xz: Array) -> Array:
        """$\\log \\tilde p(x, z) = \\log \\tilde p_Z(z) + \\log p(x \\mid z)$ of a joint datapoint."""
        x = xz[..., : self.obs_man.data_dim]
        z = xz[..., self.obs_man.data_dim :]
        dep_params = self.conjugated_prior_params(params)
        return self.dep_man.log_density(dep_params, z) + self.obs_man.log_density(
            self.likelihood_at(params, z), x
        )

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Ancestral samples of $\\tilde p(x, z)$: $z$ from the deep model at the prior, then $x \\sim p(x \\mid z)$."""
        key_z, key_x = jax.random.split(key)
        dep_params = self.conjugated_prior_params(params)
        zs = self.dep_man.sample(key_z, dep_params, n)

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
        """The initialization of :attr:`gen_hrm`, with zero conjugation function parameters at every level."""
        hrm_params = self.gen_hrm.initialize(key, location, shape)
        return self.join_coords(hrm_params, self.snd_man.zeros())

    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """As :meth:`initialize`, with the observable biases from the sample."""
        hrm_params = self.gen_hrm.initialize_from_sample(key, sample, location, shape)
        return self.join_coords(hrm_params, self.snd_man.zeros())

    def conjugation_residual(self, params: Array, z: Array) -> Array:
        """The residual $r(z) = \\rho \\cdot \\mathbf s(z) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z)) + \\psi_X(\\theta_X)$ of this level at the datapoint ``z`` of the prior family, with $\\rho$ in natural coordinates of the prior family."""
        hrm_params, _ = self.split_coords(params)
        obs_params, _, lat_params = self.gen_hrm.split_coords(hrm_params)
        prr_params, _ = self.dep_man.split_coords(self.conjugated_prior_params(params))
        rho = prr_params - self.pst_prr_emb.embed(lat_params)
        return (
            jnp.dot(rho, self.prr_man.sufficient_statistic(z))
            - self.obs_man.log_partition_function(self.likelihood_at(params, z))
            + self.obs_man.log_partition_function(obs_params)
        )

    @override
    def conjugation_residuals(self, params: Array, xz: Array) -> tuple[Array, ...]:
        """The residual of each level of $\\tilde p$ at a joint datapoint, bottom first."""
        z = xz[..., self.obs_man.data_dim :]
        return (
            self.conjugation_residual(params, z),
            *self.dep_man.conjugation_residuals(
                self.conjugated_prior_params(params), z
            ),
        )

    def elbo_residual(self, params: Array, x: Array, z: Array) -> Array:
        """The ELBO integrand minus :meth:`conjugation_baseline`: $r(z) + \\sum r^0(z) - \\sum r^X(z)$, with $r^0$ and $r^X$ the residuals of the deep model at the prior and at the posterior."""
        dep_prior = self.conjugated_prior_params(params)
        dep_post = self.recognition_at(params, x)
        return (
            self.conjugation_residual(params, z)
            + sum(self.dep_man.conjugation_residuals(dep_prior, z), start=jnp.zeros(()))
            - sum(self.dep_man.conjugation_residuals(dep_post, z), start=jnp.zeros(()))
        )

    def sample_recognition(self, key: Array, params: Array, x: Array, n: int) -> Array:
        """Samples of $q(z \\mid x)$, the deep model at the posterior, as rows."""
        return self.dep_man.sample(key, self.recognition_at(params, x), n)

    def recognition_log_density(self, params: Array, x: Array, z: Array) -> Array:
        """$\\log q(z \\mid x)$."""
        return self.dep_man.log_density(self.recognition_at(params, x), z)

    def elbo_at(self, key: Array, params: Array, x: Array, n_samples: int) -> Array:
        """Estimate the ELBO $\\mathcal L(x) = c(x) + \\mathbb E_q[r + \\sum r^0 - \\sum r^X]$: the baseline in closed form, the residuals by Monte Carlo.

        Only the residuals are sampled, and only they drive the score-function correction:
        with $r$ the summed residual (:meth:`elbo_residual`),

        $$\\nabla \\mathcal L(x) = \\nabla c(x) + \\mathbb E_q[\\nabla r] + \\mathbb E_q[(r - b) \\nabla \\log q],$$

        with a leave-one-out baseline $b$ (:meth:`DeepModel.score_mean`).
        """
        mean = self.dep_man.score_mean(
            key,
            self.recognition_at(params, x),
            lambda z: self.elbo_residual(params, x, z),
            n_samples,
        )
        return self.conjugation_baseline(params, x) + mean

    def mean_elbo(self, key: Array, params: Array, xs: Array, n_samples: int) -> Array:
        """Mean of :meth:`elbo_at` over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        return jnp.mean(
            jax.vmap(lambda k, x: self.elbo_at(k, params, x, n_samples))(keys, xs)
        )

    @override
    def conjugation_residual_variances(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """The variance under $\\tilde p$ of the residual of each level, bottom first.

        The variance of each level is under samples of its own deep model, with a
        score-function term for the dependence of their distribution on the parameters
        (:meth:`DeepModel.score_variance`). Needs ``n_samples >= 2``.
        """
        key_own, key_deep = jax.random.split(key)
        dep_prior = self.conjugated_prior_params(params)
        var = self.dep_man.score_variance(
            key_own,
            dep_prior,
            lambda z: self.conjugation_residual(params, z),
            n_samples,
        )
        return (
            var,
            *self.dep_man.conjugation_residual_variances(
                key_deep, dep_prior, n_samples
            ),
        )

    def recognition_residual_variances_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> tuple[Array, ...]:
        """The variance under $q(z \\mid x)$ of the residual $r$ of this level, then of each residual $r^X$ of the deep model at the posterior.

        For an exact deep model, $\\mathrm{Var}_q[r] = \\mathrm{Var}_q[\\log(q(Z \\mid x) /
        p(Z \\mid x))]$, which vanishes if and only if the recognition model is the posterior.
        Gradients include the score-function term (:meth:`DeepModel.score_variance`). Needs
        ``n_samples >= 2``.
        """
        key_own, key_deep = jax.random.split(key)
        dep_post = self.recognition_at(params, x)
        var = self.dep_man.score_variance(
            key_own, dep_post, lambda z: self.conjugation_residual(params, z), n_samples
        )
        return (
            var,
            *self.dep_man.conjugation_residual_variances(key_deep, dep_post, n_samples),
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


class DifferentiableVariationalConjugated[
    Graphical: GraphicalHarmonium[Any],
    Prior: Differentiable,
    Conjugation: Manifold,
](
    VariationalConjugated[Graphical, Conjugation],
    ABC,
):
    """A variational model whose prior family is :class:`Differentiable`, used as its deep model as it is."""

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
    def dep_man(self) -> DifferentiableDeepModel[Prior]:
        return DifferentiableDeepModel(self.prr_man)

    # Methods

    def elbo_divergence(self, params: Array, x: Array) -> Array:
        """Closed-form $\\mathrm{KL}(q(z \\mid x) \\Vert \\tilde p(z))$, for $\\beta$-VAE-style warm-up and diagnostics."""
        return self.prr_man.relative_entropy(
            self.recognition_at(params, x), self.conjugated_prior_params(params)
        )


def regress_conjugation_parameters(
    model: VariationalConjugated[Any, Any],
    key: Array,
    params: Array,
    n_samples: int,
) -> tuple[Array, Array, Array, Array]:
    """Fit this level's conjugation function parameters by least squares against samples of the deep model at the prior; they minimize the sampled variance of the residual.

    Solves $\\min_{\\chi, c} \\sum_k (\\chi + \\rho(c) \\cdot \\mathbf s(z_k) - \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z_k)))^2$ over the conjugation function parameters $c$, with $\\rho$ from :meth:`~VariationalConjugated.conjugation_parameters` linearized at $c = 0$, so it is exact when $\\rho$ is affine in $c$. The intercept is free and the variance is shift-invariant, so the fit minimizes the sampled variance of the residual. The samples are drawn at the given parameters and held fixed, although with $\\rho$ in the prior they depend on $c$. They carry no gradient, since the sampler may be non-differentiable; gradients flow through the targets $\\psi_X$.

    Returns ``(cnj_fun_params, r_squared, chi, residual_var)``. A fitting heuristic, not part of the variational objective.
    """
    lkl_params = model.likelihood_function(params)
    zero_cnj_fun = model.cnj_fun_man.zeros()
    obs_dim = model.obs_man.data_dim
    zs = jax.lax.stop_gradient(model.sample(key, params, n_samples)[:, obs_dim:])

    def design_and_target(z: Array) -> tuple[Array, Array, Array]:
        s_z = model.prr_man.sufficient_statistic(z)

        def rho_dot_s(c: Array) -> Array:
            return jnp.dot(model.conjugation_parameters(lkl_params, c), s_z)

        s_proj = jax.grad(rho_dot_s)(zero_cnj_fun)
        offset = rho_dot_s(zero_cnj_fun)
        psi_x = model.obs_man.log_partition_function(model.likelihood_at(params, z))
        return s_proj, psi_x, offset

    s_proj_all, psi_all, offset_all = jax.vmap(design_and_target)(zs)
    adj_psi = psi_all - offset_all

    design = jnp.concatenate([jnp.ones((n_samples, 1)), s_proj_all], axis=1)
    coeffs = jnp.linalg.lstsq(design, adj_psi, rcond=None)[0]
    chi = coeffs[0]
    cnj_fun_params = coeffs[1:]

    residuals = adj_psi - design @ coeffs
    ss_res = jnp.sum(residuals**2)
    ss_tot = jnp.sum((adj_psi - jnp.mean(adj_psi)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    return cnj_fun_params, r_squared, chi, jnp.var(residuals)


def conjugation_metrics(
    model: VariationalConjugated[Any, Any],
    key: Array,
    params: Array,
    n_samples: int = 100,
) -> tuple[Array, Array, Array]:
    """Conjugation quality ``(var_r, std_r, r_squared)`` of this level under the model.

    $R^2 = 1 - \\mathrm{Var}[r] / \\mathrm{Var}[\\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(Z))]$ measures how much of the variation of the log-partition function the conjugation parameters explain: 1 is exact conjugation, 0 no better than a constant.
    """
    var_key, psi_key = jax.random.split(key)
    var_r = model.conjugation_residual_variances(var_key, params, n_samples)[0]
    var_psi = model.dep_man.score_variance(
        psi_key,
        model.conjugated_prior_params(params),
        lambda z: model.obs_man.log_partition_function(model.likelihood_at(params, z)),
        n_samples,
    )
    return var_r, jnp.sqrt(var_r), 1.0 - var_r / var_psi


def reconstruct(
    model: DifferentiableVariationalConjugated[Any, Any, Any], params: Array, x: Array
) -> Array:
    """Observable means of the likelihood at the recognition model's mean statistics: a mean-field reconstruction."""
    z_means = model.pst_prr_emb.project(
        model.prr_man.to_mean(model.recognition_at(params, x))
    )
    lkl_params = model.gen_hrm.lkl_fun_man(model.likelihood_function(params), z_means)
    return model.obs_man.to_mean(lkl_params)


def reconstruction_error(
    model: DifferentiableVariationalConjugated[Any, Any, Any], params: Array, xs: Array
) -> Array:
    """Mean squared reconstruction error over a batch."""
    recons = jax.vmap(lambda x: reconstruct(model, params, x))(xs)
    return jnp.mean((xs - recons) ** 2)
