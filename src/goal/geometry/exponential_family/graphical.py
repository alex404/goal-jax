"""Graphical harmoniums: a deep model with harmoniums attached to its cliques.

A graphical harmonium is built from a deep latent model and one or more **attachments**:
harmoniums whose posterior sits on a clique of the deep model. The observable is the
attachments' observables, the interaction is their interactions, and the deep model is the
posterior, so the layout, the crossing cliques and their maps are all read off the
attachments (see :class:`~goal.geometry.manifold.clique.RecursiveLinearCliques`). A
hierarchical mixture of Gaussians is one linear Gaussian model attached to the observable
node of a mixture.

Conjugation is derived in the same way. Each attachment is conjugated on its own, and its
conjugation parameters are placed on its clique of the prior deep model; with several
attachments, which are conditionally independent given the deep model, the placed
parameters are summed.

Mathematically, writing $h_i$ for the attachments, $C_i$ for their cliques and $\\iota_{C_i}$
for the inclusion of a clique's block into the prior deep model, the conjugation equation
of the composite holds with

.. math::
    \\rho = \\sum_i \\iota_{C_i}(\\rho_i), \\qquad \\chi = \\sum_i \\chi_i,

where $\\rho_i, \\chi_i$ are the conjugation parameters and offset of $h_i$. This requires
each $\\rho_i$ to have a block in the prior deep model, i.e. the cliques of $h_i$'s prior,
placed on $C_i$, must be cliques of the prior deep model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, override

import jax
import jax.numpy as jnp
from jax import Array

from ..algebra.matrix import MatrixRep
from ..manifold.base import Manifold
from ..manifold.clique import CliqueEmbedding, LinearCliques
from ..manifold.combinators import Null, Tuple
from ..manifold.embedding import LinearEmbedding
from ..manifold.map import MultilayerPerceptron
from ..manifold.util import split_by_dims
from .base import Analytic, Differentiable, Generative, Gibbs
from .combinators import DifferentiablePair
from .harmonium import (
    AnalyticConjugated,
    Conjugated,
    DifferentiableConjugated,
    Harmonium,
)
from .variational import score_surrogate, variance_with_score_correction

### Graphical Harmoniums ###


@dataclass(frozen=True)
class GraphicalHarmonium[Deep: Gibbs](Harmonium[Any, Deep], ABC):
    """A harmonium whose posterior is a deep model with harmoniums attached to its cliques.

    A model declares the deep model (:attr:`pst_man`) and its :attr:`attachments`; the
    observable, the crossing cliques and their maps are those of the attachments. An
    attachment's crossings are carried over with their root part shifted past the earlier
    attachments' observables, and their deep part relabelled from the attachment's
    posterior numbering into the deep model's through its clique.
    """

    # Contract

    @property
    @abstractmethod
    def attachments(self) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        """Each attached harmonium, with the clique of the deep model its posterior sits on.

        Node $j$ of the attachment's posterior is node ``clique[j]`` of the deep model.
        """

    # Overrides

    @property
    @override
    def obs_man(self) -> Any:
        """The attachments' observables side by side, in attachment order.

        One attachment gives its observable, two an :class:`ObservablePair`.
        """
        obs_mans = [hrm.obs_man for hrm, _ in self.attachments]
        if len(obs_mans) == 1:
            return obs_mans[0]
        fst, snd = obs_mans
        return ObservablePair(fst, snd)

    @property
    @override
    def crs_cliques(self) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        """Every attachment's crossings, in attachment order, relabelled into the composite."""
        return tuple(self._crossing_sources)

    @override
    def crs_rep(self, crossing: tuple[tuple[int, ...], tuple[int, ...]]) -> MatrixRep:
        hrm, own = self._crossing_sources[crossing]
        return hrm.crs_rep(own)

    @override
    def crs_emb_constructors(
        self, crossing: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> tuple[
        tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...],
        tuple[Callable[[Manifold], LinearEmbedding[Any, Any]], ...],
    ]:
        hrm, own = self._crossing_sources[crossing]
        return hrm.crs_emb_constructors(own)

    # Methods

    def attachment_likelihoods(self, lkl_params: Array) -> tuple[Array, ...]:
        """Split the composite's likelihood natural parameters into each attachment's."""
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)
        obs_blocks = split_by_dims(
            obs_params, tuple(hrm.obs_man.dim for hrm, _ in self.attachments)
        )
        int_blocks = iter(self.crs_man.coord_blocks(int_params))
        lkls: list[Array] = []
        for (hrm, _), obs_block in zip(self.attachments, obs_blocks, strict=True):
            int_block = jnp.concatenate([next(int_blocks) for _ in hrm.crs_cliques])
            lkls.append(hrm.lkl_fun_man.join_coords(obs_block, int_block))
        return tuple(lkls)

    def attachment_posterior(self, index: int, coords: Array) -> Array:
        """Read one attachment's posterior coordinates from the deep model's.

        The coordinates may be natural or mean: the blocks of the attachment's posterior
        cliques are read where its clique places them.
        """
        hrm, clique = self.attachments[index]
        blocks = [
            CliqueEmbedding(placed, self.pst_man).project(coords)
            for _, placed in _placed_cliques(hrm.pst_man, clique)
        ]
        return jnp.concatenate(blocks)

    @property
    def _crossing_sources(
        self,
    ) -> dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[Harmonium[Any, Any], tuple[tuple[int, ...], tuple[int, ...]]],
    ]:
        """Each composite crossing, in order, with its attachment and its crossing there."""
        sources: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[Harmonium[Any, Any], tuple[tuple[int, ...], tuple[int, ...]]],
        ] = {}
        n_rot = 0
        for hrm, clique in self.attachments:
            for near, far in hrm.crs_cliques:
                placed = (
                    tuple(i + n_rot for i in near),
                    tuple(clique[j] for j in far),
                )
                sources[placed] = (hrm, (near, far))
            n_rot += hrm.obs_man.n_nodes
        return sources


class DifferentiableGraphical[Deep: Differentiable, PriorDeep: Differentiable](
    GraphicalHarmonium[Deep],
    DifferentiableConjugated[Any, Deep, PriorDeep],
    ABC,
):
    """A graphical harmonium whose attachments are conjugated, and so is the composite.

    Its conjugation parameters are each attachment's, placed on the attachment's clique of
    the prior deep model (:meth:`place_conjugation`) and summed. The prior deep model is
    :attr:`~goal.geometry.exponential_family.harmonium.Conjugated.prr_man`, given through
    :attr:`~goal.geometry.exponential_family.harmonium.Conjugated.pst_prr_emb`; it must
    hold each attachment's prior on that attachment's clique.
    """

    # Overrides

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Place each attachment's conjugation parameters on its clique of the prior deep model, summed."""
        rhos = tuple(
            hrm.conjugation_parameters(lkl)
            for (hrm, _), lkl in zip(
                self.cnj_attachments,
                self.attachment_likelihoods(lkl_params),
                strict=True,
            )
        )
        return self.place_conjugation(rhos)

    @override
    def conjugation_offset(self, lkl_params: Array) -> Array:
        """Sum the attachments' conjugation offsets."""
        chis = [
            hrm.conjugation_offset(lkl)
            for (hrm, _), lkl in zip(
                self.cnj_attachments,
                self.attachment_likelihoods(lkl_params),
                strict=True,
            )
        ]
        return jnp.sum(jnp.stack(chis))

    # Methods

    @property
    def cnj_attachments(
        self,
    ) -> tuple[tuple[Conjugated[Any, Any, Any], tuple[int, ...]], ...]:
        """The attachments, typed as conjugated."""
        return cast(
            tuple[tuple[Conjugated[Any, Any, Any], tuple[int, ...]], ...],
            self.attachments,
        )

    def place_conjugation(self, rhos: tuple[Array, ...]) -> Array:
        """Place each attachment's conjugation parameters on its clique of the prior deep model, and sum.

        Each entry of ``rhos`` is in the coordinates of its attachment's prior; the result is
        in the prior deep model's.
        """
        out = self.prr_man.zeros()
        for (hrm, clique), rho in zip(self.cnj_attachments, rhos, strict=True):
            att_prr: LinearCliques = hrm.prr_man
            for (_, placed), block in zip(
                _placed_cliques(att_prr, clique),
                att_prr.coord_blocks(rho),
                strict=True,
            ):
                out = out + CliqueEmbedding(placed, self.prr_man).embed(block)
        return out


class AnalyticGraphical[Deep: Analytic](
    AnalyticConjugated[Any, Deep],
    DifferentiableGraphical[Deep, Deep],
    ABC,
):
    """A graphical harmonium with one analytic attachment and an analytic deep model.

    The posterior and prior deep models coincide, and the likelihood is recovered from mean
    parameters by projecting onto the attachment and inverting there.
    """

    # Overrides

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Project the mean parameters onto the attachment and convert there."""
        ((hrm, _),) = self.attachments
        hrm = cast(AnalyticConjugated[Any, Any], hrm)
        obs_means, int_means, lat_means = self.split_coords(means)
        att_means = hrm.join_coords(
            obs_means, int_means, self.attachment_posterior(0, lat_means)
        )
        return hrm.to_natural_likelihood(att_means)


### Variational Graphical Harmoniums ###


@dataclass(frozen=True)
class VariationalGraphical(Generative, Tuple, ABC):
    """A three-level directed model $p(x, y, z) = p(x \\mid y) p(y \\mid z) p(z)$ with a chain recognition model, built on a graphical harmonium.

    The generative structure is that of :attr:`gen_hrm`: a graphical harmonium with one attachment, the lower harmonium over $(x, y)$, on the observable node of a deep harmonium over $(y, z)$, the upper harmonium. Each of the two edges is conjugated exactly when its harmonium is :class:`~goal.geometry.exponential_family.harmonium.Conjugated`; otherwise its conjugation parameters are learned. Exact and learned edges mix freely.

    Parameters are stored in directed coordinates ``[top prior | upper lkl | lower lkl | rho_Y | rho_Z | inner map]``: the prior $\\theta^*_Z$ over $z$, the likelihoods $(\\theta_Y, \\Theta_{YZ})$ and $(\\theta_X, \\Theta_{XY})$, and the learned conjugation parameters of the edges that are not exact. The slots of exact edges are empty: a learned lower edge stores $\\rho_Y$; a learned upper edge stores the input-independent $\\rho^0_Z$ and the weights of :attr:`inr_map`, which amortizes the upper edge's conjugation parameters at the posterior bias of $y$.

    Mathematically, the recognition model factors as $q(y, z \\mid x) = q(y \\mid z, x) q(z \\mid x)$ with

    .. math::
        \\hat\\theta_{Y \\mid X}(x) = \\theta_Y - \\rho_Y + \\mathbf s_X(x) \\cdot \\Theta_{XY}, \\qquad
        \\hat\\theta_{Z \\mid X}(x) = \\theta^*_Z - \\rho^0_Z + \\rho^X_Z(x), \\qquad
        \\hat\\theta_{Y \\mid Z, X} = \\hat\\theta_{Y \\mid X}(x) + \\Theta_{YZ} \\cdot \\mathbf s_Z(z),

    where $\\rho^0_Z$ and $\\rho^X_Z(x)$ are the upper edge's conjugation parameters at the biases $\\theta_Y$ and $\\hat\\theta_{Y \\mid X}(x)$: computed when the edge is exact, and learned (respectively amortized) when it is not. The ELBO integrand splits as

    .. math::
        \\log p(x, y, z) - \\log q(y, z \\mid x) = c(x) + r_Y(y) + r^*_Z(z) - r^X_Z(z; x),

    with $r_Y$ the lower edge's conjugation residual, $r^*_Z$ and $r^X_Z$ the upper edge's at the generative and the posterior bias, and $c(x)$ collecting everything else (:meth:`conjugation_baseline`). An exact edge has a vanishing residual, and with both edges exact the ELBO is $\\log p(x)$.

    The lower harmonium's posterior and prior are the upper harmonium's observable, so $\\rho_Y$ and $\\theta_Y$ share coordinates.
    """

    # Contract

    @property
    @abstractmethod
    def gen_hrm(self) -> GraphicalHarmonium[Any]:
        """The generative structure: one attachment, on the observable node of a harmonium."""

    @property
    @abstractmethod
    def inr_map(self) -> MultilayerPerceptron[Any, Any] | None:
        """The amortized map from the posterior bias of $y$ to the upper edge's conjugation parameters; ``None`` when the upper edge is exact."""

    # Overrides

    @property
    @override
    def dim(self) -> int:
        return sum(man.dim for man in self.cmp_mans)

    @override
    def split_coords(
        self, coords: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """``(top prior, upper lkl, lower lkl, rho_Y, rho_Z, inner map)``."""
        prr, upr, lwr, rho_y, rho_z, inr = split_by_dims(
            coords, tuple(man.dim for man in self.cmp_mans)
        )
        return prr, upr, lwr, rho_y, rho_z, inr

    @override
    def join_coords(self, *components: Array) -> Array:
        return jnp.concatenate(components)

    @property
    @override
    def data_dim(self) -> int:
        """Dimension of a joint $(x, y, z)$ datapoint."""
        return self.gen_hrm.data_dim

    @override
    def sufficient_statistic(self, x: Array) -> Array:
        """Sufficient statistic of a joint $(x, y, z)$ datapoint, in the generative harmonium's coordinates."""
        return self.gen_hrm.sufficient_statistic(x)

    @override
    def log_base_measure(self, x: Array) -> Array:
        return self.gen_hrm.log_base_measure(x)

    @override
    def sample(self, key: Array, params: Array, n: int = 1) -> Array:
        """Ancestral samples $z \\sim p(z)$, $y \\sim p(y \\mid z)$, $x \\sim p(x \\mid y)$, as rows $[x, y, z]$."""
        key_z, key_y, key_x = jax.random.split(key, 3)
        prr, upr_lkl, lwr_lkl, _, _, _ = self.split_coords(params)
        zs = self.top_man.sample(key_z, prr, n)

        def sample_y(subkey: Array, z: Array) -> Array:
            y_params = self.upr_hrm.lkl_fun_man(
                upr_lkl, self.upr_hrm.pst_man.sufficient_statistic(z)
            )
            return self.mid_man.sample(subkey, y_params, 1)[0]

        ys = jax.vmap(sample_y)(jax.random.split(key_y, n), zs)

        def sample_x(subkey: Array, y: Array) -> Array:
            x_params = self.lwr_hrm.lkl_fun_man(
                lwr_lkl, self.lwr_hrm.pst_man.sufficient_statistic(y)
            )
            return self.obs_man.sample(subkey, x_params, 1)[0]

        xs = jax.vmap(sample_x)(jax.random.split(key_x, n), ys)
        return jnp.concatenate([xs, ys, zs], axis=-1)

    @override
    def initialize(
        self, key: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """Initialize the likelihoods and prior from the two harmoniums, and the learned conjugation parameters at zero.

        The inner map's hidden layers are Glorot-initialized and its output layer is zero, so with a learned upper edge the recognition model starts at $q(z \\mid x) = p(z)$.
        """
        key_lwr, key_upr, key_inr = jax.random.split(key, 3)
        lwr_params = self.lwr_hrm.initialize(key_lwr, location, shape)
        return self._initial_params(lwr_params, key_upr, key_inr, location, shape)

    @override
    def initialize_from_sample(
        self, key: Array, sample: Array, location: float = 0.0, shape: float = 0.1
    ) -> Array:
        """As :meth:`initialize`, with the lower harmonium initialized from the sample."""
        key_lwr, key_upr, key_inr = jax.random.split(key, 3)
        lwr_params = self.lwr_hrm.initialize_from_sample(
            key_lwr, sample, location, shape
        )
        return self._initial_params(lwr_params, key_upr, key_inr, location, shape)

    # Methods

    @property
    def lwr_hrm(self) -> Harmonium[Any, Any]:
        """The lower harmonium over $(x, y)$."""
        ((hrm, _),) = self.gen_hrm.attachments
        return hrm

    @property
    def upr_hrm(self) -> Harmonium[Any, Any]:
        """The upper harmonium over $(y, z)$."""
        return self.gen_hrm.pst_man

    @property
    def obs_man(self) -> Any:
        """The observable family, of $x$."""
        return self.lwr_hrm.obs_man

    @property
    def mid_man(self) -> Any:
        """The middle family, of $y$."""
        return self.upr_hrm.obs_man

    @property
    def top_man(self) -> Any:
        """The top family, of $z$: the upper harmonium's prior when it is conjugated, its posterior otherwise."""
        upr = self.upr_hrm
        return upr.prr_man if isinstance(upr, Conjugated) else upr.pst_man

    @property
    def lower_exact(self) -> bool:
        """Whether the lower edge is conjugated, so that its residual $r_Y$ vanishes."""
        return isinstance(self.lwr_hrm, Conjugated)

    @property
    def upper_exact(self) -> bool:
        """Whether the upper edge is conjugated, so that its residuals vanish."""
        return isinstance(self.upr_hrm, Conjugated)

    @property
    def cmp_mans(self) -> tuple[Manifold, ...]:
        """The manifold of each parameter slot; the slots of exact edges are :class:`~goal.geometry.manifold.combinators.Null`."""
        inr_map = self.inr_map
        return (
            self.top_man,
            self.upr_hrm.lkl_fun_man,
            self.lwr_hrm.lkl_fun_man,
            Null() if self.lower_exact else self.mid_man,
            Null() if self.upper_exact else self.top_man,
            inr_map if not self.upper_exact and inr_map is not None else Null(),
        )

    # Conjugation parameters

    def lower_conjugation(self, params: Array) -> Array:
        """The lower edge's conjugation parameters $\\rho_Y$: computed when the edge is exact, stored otherwise."""
        _, _, lwr_lkl, rho_y, _, _ = self.split_coords(params)
        lwr = self.lwr_hrm
        if isinstance(lwr, Conjugated):
            return lwr.conjugation_parameters(lwr_lkl)
        return rho_y

    def upper_conjugation(self, params: Array) -> Array:
        """The upper edge's conjugation parameters $\\rho^0_Z$ at the generative bias $\\theta_Y$: computed when the edge is exact, stored otherwise."""
        _, upr_lkl, _, _, rho_z, _ = self.split_coords(params)
        upr = self.upr_hrm
        if isinstance(upr, Conjugated):
            return upr.conjugation_parameters(upr_lkl)
        return rho_z

    def inner_conjugation(self, params: Array, x: Array) -> Array:
        """The upper edge's conjugation parameters $\\rho^X_Z(x)$ at the posterior bias $\\hat\\theta_{Y \\mid X}(x)$: computed when the edge is exact, amortized by :attr:`inr_map` otherwise."""
        *_, inr = self.split_coords(params)
        mid_bias = self.posterior_mid_bias(params, x)
        upr = self.upr_hrm
        if isinstance(upr, Conjugated):
            return upr.conjugation_parameters(self._upper_lkl_at(params, mid_bias))
        return cast(MultilayerPerceptron[Any, Any], self.inr_map)(inr, mid_bias)

    # Recognition model

    def posterior_mid_bias(self, params: Array, x: Array) -> Array:
        """Natural parameters $\\hat\\theta_{Y \\mid X}(x) = \\theta_Y - \\rho_Y + \\mathbf s_X(x) \\cdot \\Theta_{XY}$ of $q(y \\mid x)$ before the coupling to $z$."""
        _, upr_lkl, lwr_lkl, _, _, _ = self.split_coords(params)
        theta_y, _ = self.upr_hrm.lkl_fun_man.split_coords(upr_lkl)
        theta_x, theta_xy = self.lwr_hrm.lkl_fun_man.split_coords(lwr_lkl)
        lwr_params = self.lwr_hrm.join_coords(
            theta_x, theta_xy, theta_y - self.lower_conjugation(params)
        )
        return self.lwr_hrm.posterior_at(lwr_params, x)

    def approximate_posterior_top(self, params: Array, x: Array) -> Array:
        """Natural parameters $\\hat\\theta_{Z \\mid X}(x) = \\theta^*_Z - \\rho^0_Z + \\rho^X_Z(x)$ of $q(z \\mid x)$."""
        prr, *_ = self.split_coords(params)
        return prr - self.upper_conjugation(params) + self.inner_conjugation(params, x)

    def posterior_mid_at(self, params: Array, x: Array, z: Array) -> Array:
        """Natural parameters $\\hat\\theta_{Y \\mid X}(x) + \\Theta_{YZ} \\cdot \\mathbf s_Z(z)$ of $q(y \\mid z, x)$."""
        upr_lkl = self._upper_lkl_at(params, self.posterior_mid_bias(params, x))
        return self.upr_hrm.lkl_fun_man(
            upr_lkl, self.upr_hrm.pst_man.sufficient_statistic(z)
        )

    def sample_posterior(
        self, key: Array, params: Array, x: Array, n: int
    ) -> tuple[Array, Array]:
        """Chain samples $z \\sim q(z \\mid x)$, then $y \\sim q(y \\mid z, x)$; returns ``(ys, zs)``."""
        key_z, key_y = jax.random.split(key)
        zs = self.top_man.sample(key_z, self.approximate_posterior_top(params, x), n)
        return self._sample_mid(key_y, params, x, zs), zs

    def log_q(self, params: Array, x: Array, y: Array, z: Array) -> Array:
        """$\\log q(y, z \\mid x) = \\log q(z \\mid x) + \\log q(y \\mid z, x)$."""
        log_qz = self.top_man.log_density(self.approximate_posterior_top(params, x), z)
        log_qy = self.mid_man.log_density(self.posterior_mid_at(params, x, z), y)
        return log_qz + log_qy

    def log_density_joint(self, params: Array, x: Array, y: Array, z: Array) -> Array:
        """$\\log p(x, y, z) = \\log p(z) + \\log p(y \\mid z) + \\log p(x \\mid y)$."""
        prr, upr_lkl, lwr_lkl, _, _, _ = self.split_coords(params)
        y_params = self.upr_hrm.lkl_fun_man(
            upr_lkl, self.upr_hrm.pst_man.sufficient_statistic(z)
        )
        x_params = self.lwr_hrm.lkl_fun_man(
            lwr_lkl, self.lwr_hrm.pst_man.sufficient_statistic(y)
        )
        return (
            self.top_man.log_density(prr, z)
            + self.mid_man.log_density(y_params, y)
            + self.obs_man.log_density(x_params, x)
        )

    # Residuals

    def lower_residual(self, params: Array, y: Array) -> Array:
        """$r_Y(y) = \\rho_Y \\cdot \\mathbf s_Y(y) - \\psi_X(\\theta_X + \\Theta_{XY} \\cdot \\mathbf s_Y(y)) + \\psi_X(\\theta_X)$."""
        _, _, lwr_lkl, _, _, _ = self.split_coords(params)
        theta_x, _ = self.lwr_hrm.lkl_fun_man.split_coords(lwr_lkl)
        s_y = self.lwr_hrm.pst_man.sufficient_statistic(y)
        x_params = self.lwr_hrm.lkl_fun_man(lwr_lkl, s_y)
        return (
            jnp.dot(self.lower_conjugation(params), s_y)
            - self.obs_man.log_partition_function(x_params)
            + self.obs_man.log_partition_function(theta_x)
        )

    def upper_residual(self, params: Array, z: Array) -> Array:
        """$r^*_Z(z)$: the upper edge's residual at the generative bias $\\theta_Y$ with $\\rho^0_Z$."""
        _, upr_lkl, _, _, _, _ = self.split_coords(params)
        return self._upper_residual_at(upr_lkl, self.upper_conjugation(params), z)

    def inner_residual(self, params: Array, x: Array, z: Array) -> Array:
        """$r^X_Z(z; x)$: the upper edge's residual at the posterior bias $\\hat\\theta_{Y \\mid X}(x)$ with $\\rho^X_Z(x)$."""
        upr_lkl = self._upper_lkl_at(params, self.posterior_mid_bias(params, x))
        return self._upper_residual_at(upr_lkl, self.inner_conjugation(params, x), z)

    def learning_signal(self, params: Array, x: Array, y: Array, z: Array) -> Array:
        """$r_Y(y) + r^*_Z(z) - r^X_Z(z; x)$, the part of the ELBO integrand that depends on the latents."""
        return (
            self.lower_residual(params, y)
            + self.upper_residual(params, z)
            - self.inner_residual(params, x, z)
        )

    # ELBO

    def conjugation_baseline(self, params: Array, x: Array) -> Array:
        """The latent-independent ELBO term.

        Mathematically, $c(x) = \\mathbf s_X(x) \\cdot \\theta_X - \\psi_X(\\theta_X) - \\psi_Y(\\theta_Y) - \\psi_Z(\\theta^*_Z) + \\psi_Y(\\hat\\theta_{Y \\mid X}(x)) + \\psi_Z(\\hat\\theta_{Z \\mid X}(x)) + \\log \\mu_X(x)$, the log-marginal the model would have if both edges were exact.
        """
        prr, upr_lkl, lwr_lkl, _, _, _ = self.split_coords(params)
        theta_y, _ = self.upr_hrm.lkl_fun_man.split_coords(upr_lkl)
        theta_x, _ = self.lwr_hrm.lkl_fun_man.split_coords(lwr_lkl)
        s_x = self.obs_man.sufficient_statistic(x)
        return (
            jnp.dot(s_x, theta_x)
            - self.obs_man.log_partition_function(theta_x)
            - self.mid_man.log_partition_function(theta_y)
            - self.top_man.log_partition_function(prr)
            + self.mid_man.log_partition_function(self.posterior_mid_bias(params, x))
            + self.top_man.log_partition_function(
                self.approximate_posterior_top(params, x)
            )
            + self.obs_man.log_base_measure(x)
        )

    def elbo_at(
        self, key: Array, params: Array, x: Array, n_samples: int, pathwise_z: bool
    ) -> Array:
        """Estimate the ELBO $c(x) + \\mathbb E_q[r_Y + r^*_Z - r^X_Z]$ by Monte Carlo over chain samples.

        The value is the same either way; ``pathwise_z`` selects the gradient estimator for $z$. With ``pathwise_z`` the gradient flows through the samples of $z$, which requires a differentiable sampler (a normal), and the score-function correction covers $y$ alone; otherwise both layers use the score function.
        """
        key_z, key_y = jax.random.split(key)
        q_top = self.approximate_posterior_top(params, x)
        zs = self.top_man.sample(key_z, q_top, n_samples)
        if not pathwise_z:
            zs = jax.lax.stop_gradient(zs)
        ys = jax.lax.stop_gradient(self._sample_mid(key_y, params, x, zs))

        signal = jax.vmap(lambda y, z: self.learning_signal(params, x, y, z))(ys, zs)
        if pathwise_z:
            log_q = jax.vmap(
                lambda y, z: self.mid_man.log_density(
                    self.posterior_mid_at(params, x, z), y
                )
            )(ys, zs)
        else:
            log_q = jax.vmap(lambda y, z: self.log_q(params, x, y, z))(ys, zs)
        return self.conjugation_baseline(params, x) + score_surrogate(signal, log_q)

    def marginal_elbo_at(
        self, key: Array, params: Array, x: Array, n_samples: int, pathwise_z: bool
    ) -> Array:
        """Estimate the ELBO $c(x) + \\mathbb E_{q(z \\mid x)}[r^*_Z - r^X_Z]$ for an exact lower edge, without sampling $y$.

        With $r_Y = 0$ the integrand does not depend on $y$, so $y$ integrates out exactly. ``pathwise_z`` is as in :meth:`elbo_at`.
        """
        q_top = self.approximate_posterior_top(params, x)
        zs = self.top_man.sample(key, q_top, n_samples)
        if not pathwise_z:
            zs = jax.lax.stop_gradient(zs)
        signal = jax.vmap(
            lambda z: self.upper_residual(params, z) - self.inner_residual(params, x, z)
        )(zs)
        if pathwise_z:
            log_q = jnp.zeros_like(signal)
        else:
            log_q = jax.vmap(lambda z: self.top_man.log_density(q_top, z))(zs)
        return self.conjugation_baseline(params, x) + score_surrogate(signal, log_q)

    def mean_elbo(
        self, key: Array, params: Array, xs: Array, n_samples: int, pathwise_z: bool
    ) -> Array:
        """Mean of :meth:`elbo_at` over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        return jnp.mean(
            jax.vmap(lambda k, x: self.elbo_at(k, params, x, n_samples, pathwise_z))(
                keys, xs
            )
        )

    def mean_marginal_elbo(
        self, key: Array, params: Array, xs: Array, n_samples: int, pathwise_z: bool
    ) -> Array:
        """Mean of :meth:`marginal_elbo_at` over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        return jnp.mean(
            jax.vmap(
                lambda k, x: self.marginal_elbo_at(k, params, x, n_samples, pathwise_z)
            )(keys, xs)
        )

    # Conjugation regularizers

    def prior_conjugation_losses(
        self, key: Array, params: Array, n_samples: int
    ) -> tuple[Array, Array]:
        """$(\\mathrm{Var}_p[r_Y], \\mathrm{Var}_p[r^*_Z])$ over ancestral samples, one penalty per edge.

        The sampling distributions share parameters with the residuals --- $p(z)$ with $r^*_Z$, and $p(y) = \\int p(y \\mid z) p(z) dz$ with $r_Y$ --- so each gradient carries a score-function piece for its distribution (:func:`~goal.geometry.exponential_family.variational.variance_with_score_correction`), through $\\log p(z)$ for $r^*_Z$ and through $\\log p(z) + \\log p(y \\mid z)$ for $r_Y$. An exact lower edge has $r_Y = 0$, so only $z$ is sampled. Needs ``n_samples >= 2``.
        """
        prr, upr_lkl, *_ = self.split_coords(params)
        if self.lower_exact:
            zs = jax.lax.stop_gradient(self.top_man.sample(key, prr, n_samples))
            ys = None
        else:
            samples = jax.lax.stop_gradient(self.sample(key, params, n_samples))
            obs_dim, mid_dim = self.obs_man.data_dim, self.mid_man.data_dim
            ys = samples[:, obs_dim : obs_dim + mid_dim]
            zs = samples[:, obs_dim + mid_dim :]
        log_p_z = jax.vmap(lambda z: self.top_man.log_density(prr, z))(zs)
        r_z = jax.vmap(lambda z: self.upper_residual(params, z))(zs)
        var_z = variance_with_score_correction(r_z, log_p_z)
        if ys is None:
            return jnp.zeros(()), var_z

        def log_p_y_given_z(y: Array, z: Array) -> Array:
            y_params = self.upr_hrm.lkl_fun_man(
                upr_lkl, self.upr_hrm.pst_man.sufficient_statistic(z)
            )
            return self.mid_man.log_density(y_params, y)

        log_p_yz = log_p_z + jax.vmap(log_p_y_given_z)(ys, zs)
        r_y = jax.vmap(lambda y: self.lower_residual(params, y))(ys)
        return variance_with_score_correction(r_y, log_p_yz), var_z

    def inner_conjugation_loss_at(
        self, key: Array, params: Array, x: Array, n_samples: int
    ) -> Array:
        """$\\mathrm{Var}_{q(z \\mid x)}[r^X_Z]$, the penalty that trains the amortized inner conjugation parameters.

        The sampling distribution $q(z \\mid x)$ shares parameters with $r^X_Z$, so the gradient carries a score-function piece for it (:func:`~goal.geometry.exponential_family.variational.variance_with_score_correction`), as in the bivariate recognition penalty. Needs ``n_samples >= 2``.
        """
        q_top = self.approximate_posterior_top(params, x)
        zs = jax.lax.stop_gradient(self.top_man.sample(key, q_top, n_samples))
        r_inner = jax.vmap(lambda z: self.inner_residual(params, x, z))(zs)
        log_q = jax.vmap(lambda z: self.top_man.log_density(q_top, z))(zs)
        return variance_with_score_correction(r_inner, log_q)

    def mean_inner_conjugation_loss(
        self, key: Array, params: Array, xs: Array, n_samples: int
    ) -> Array:
        """Mean of :meth:`inner_conjugation_loss_at` over a batch."""
        keys = jax.random.split(key, xs.shape[0])
        return jnp.mean(
            jax.vmap(
                lambda k, x: self.inner_conjugation_loss_at(k, params, x, n_samples)
            )(keys, xs)
        )

    # Harmonium coordinates

    def to_harmonium(self, params: Array) -> Array:
        """Natural parameters of :attr:`gen_hrm` with each edge's conjugation parameters absorbed into the bias below it.

        The $y$-bias becomes $\\theta_Y - \\rho_Y$ and the $z$-bias $\\theta^*_Z - \\rho^0_Z$. When both edges are exact the harmonium's joint equals the directed one.
        """
        prr, upr_lkl, lwr_lkl, _, _, _ = self.split_coords(params)
        theta_y, theta_yz = self.upr_hrm.lkl_fun_man.split_coords(upr_lkl)
        theta_x, theta_xy = self.lwr_hrm.lkl_fun_man.split_coords(lwr_lkl)
        theta_z = prr - self.upper_conjugation(params)
        upr = self.upr_hrm
        if isinstance(upr, Conjugated):
            theta_z = upr.pst_prr_emb.project(theta_z)
        upr_params = upr.join_coords(
            theta_y - self.lower_conjugation(params), theta_yz, theta_z
        )
        return self.gen_hrm.join_coords(theta_x, theta_xy, upr_params)

    def _upper_lkl_at(self, params: Array, mid_bias: Array) -> Array:
        """The upper likelihood with its $y$-bias replaced."""
        _, upr_lkl, _, _, _, _ = self.split_coords(params)
        _, theta_yz = self.upr_hrm.lkl_fun_man.split_coords(upr_lkl)
        return self.upr_hrm.lkl_fun_man.join_coords(mid_bias, theta_yz)

    def _upper_residual_at(self, upr_lkl: Array, rho_z: Array, z: Array) -> Array:
        """The upper edge's conjugation residual at the given likelihood and conjugation parameters."""
        mid_bias, _ = self.upr_hrm.lkl_fun_man.split_coords(upr_lkl)
        y_params = self.upr_hrm.lkl_fun_man(
            upr_lkl, self.upr_hrm.pst_man.sufficient_statistic(z)
        )
        return (
            jnp.dot(rho_z, self.top_man.sufficient_statistic(z))
            - self.mid_man.log_partition_function(y_params)
            + self.mid_man.log_partition_function(mid_bias)
        )

    def _sample_mid(self, key: Array, params: Array, x: Array, zs: Array) -> Array:
        """One sample of $y \\sim q(y \\mid z, x)$ per row of ``zs``."""

        def sample_y(subkey: Array, z: Array) -> Array:
            return self.mid_man.sample(subkey, self.posterior_mid_at(params, x, z), 1)[
                0
            ]

        return jax.vmap(sample_y)(jax.random.split(key, zs.shape[0]), zs)

    def _initial_params(
        self,
        lwr_params: Array,
        key_upr: Array,
        key_inr: Array,
        location: float,
        shape: float,
    ) -> Array:
        """Assemble initial parameters from the lower harmonium's."""
        theta_x, theta_xy, _ = self.lwr_hrm.split_coords(lwr_params)
        theta_y, theta_yz, theta_z = self.upr_hrm.split_coords(
            self.upr_hrm.initialize(key_upr, location, shape)
        )
        upr = self.upr_hrm
        prr = upr.pst_prr_emb.embed(theta_z) if isinstance(upr, Conjugated) else theta_z
        _, _, _, rho_y_man, rho_z_man, inr_man = self.cmp_mans
        inr = jnp.zeros(0)
        if isinstance(inr_man, MultilayerPerceptron):
            *_, last_in, last_out = inr_man.layer_dims
            n_last = last_out * last_in + last_out
            inr = inr_man.glorot_initialize(key_inr).at[-n_last:].set(0.0)
        return self.join_coords(
            prr,
            self.upr_hrm.lkl_fun_man.join_coords(theta_y, theta_yz),
            self.lwr_hrm.lkl_fun_man.join_coords(theta_x, theta_xy),
            jnp.zeros(rho_y_man.dim),
            jnp.zeros(rho_z_man.dim),
            inr,
        )


@dataclass(frozen=True)
class ObservablePair[First: Differentiable, Second: Differentiable](
    DifferentiablePair[First, Second]
):
    """The observables of two attachments, side by side over disjoint data slices."""

    # Fields

    _fst_man: First
    _snd_man: Second

    # Overrides

    @property
    @override
    def fst_man(self) -> First:
        return self._fst_man

    @property
    @override
    def snd_man(self) -> Second:
        return self._snd_man


def _placed_cliques(
    man: LinearCliques, clique: tuple[int, ...]
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Each clique of ``man``, with the clique it becomes when node $j$ is relabelled ``clique[j]``."""
    return tuple((own, tuple(clique[j] for j in own)) for own in man.cliques)


### Embeddings ###


@dataclass(frozen=True)
class LatentHarmoniumEmbedding[
    PriorHarmonium: Harmonium[Any, Any],
    PostHarmonium: Harmonium[Any, Any],
](LinearEmbedding[PriorHarmonium, PostHarmonium]):
    """Embeds one harmonium into another by embedding only the observable component.

    Used in graphical models where the posterior deep model uses a restricted observable representation (e.g., diagonal covariance) while the prior uses a fuller one. Interaction and latent components pass through unchanged.

    Mathematically, for harmonium points $(o, i, l)$: embedding maps $(o, i, l) \\mapsto (\\phi(o), i, l)$ and projection maps $(o, i, l) \\mapsto (\\pi(o), i, l)$, with $\\phi, \\pi$ those of :attr:`obs_emb`.
    """

    # Fields

    obs_emb: LinearEmbedding[Any, Any]
    """Embedding of the restricted observable manifold into the full observable manifold."""

    _amb_man: PriorHarmonium
    _sub_man: PostHarmonium

    def __post_init__(self) -> None:
        sub, amb = self.sub_man, self.amb_man
        if sub.cliques != amb.cliques or (sub.int_man.dim, sub.pst_man.dim) != (
            amb.int_man.dim,
            amb.pst_man.dim,
        ):
            raise ValueError("the two harmoniums may differ only in their observable")

    # Overrides

    @property
    @override
    def sub_man(self) -> PostHarmonium:
        return self._sub_man

    @property
    @override
    def amb_man(self) -> PriorHarmonium:
        return self._amb_man

    @override
    def project(self, coords: Array) -> Array:
        obs_params, int_params, lat_params = self.amb_man.split_coords(coords)
        prj_obs_params = self.obs_emb.project(obs_params)
        return self.sub_man.join_coords(prj_obs_params, int_params, lat_params)

    @override
    def embed(self, coords: Array) -> Array:
        obs_params, int_params, lat_params = self.sub_man.split_coords(coords)
        emb_obs_params = self.obs_emb.embed(obs_params)
        return self.amb_man.join_coords(emb_obs_params, int_params, lat_params)
