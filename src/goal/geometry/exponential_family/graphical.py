"""Graphical harmoniums: harmoniums attached to the cliques of a latent model.

A graphical harmonium is built from a latent model and one or more
harmoniums, :attr:`GraphicalHarmonium.obs_hrms`. The latent variables of each of these
harmoniums are the variables of one clique of the latent model, and their observables
together make up the composite's observable. The composite's interaction is their
interactions, and its posterior is the latent model. A hierarchical mixture of Gaussians
attaches one linear Gaussian model to a mixture; canonical correlation analysis attaches
two linear Gaussian models to one normal.

Given the latent model, the observables are independent. So if every harmonium in
``obs_hrms`` is conjugated, the composite is too, and its conjugation parameters are the
sum of theirs, each placed on its clique.

Mathematically, let $\\rho_i$ and $\\chi_i$ be the conjugation parameters and offset of the
$i$-th harmonium in ``obs_hrms``, and $\\iota_i$ the embedding of its prior into the
composite's prior on its clique. Then the composite has

.. math::
    \\rho = \\sum_i \\iota_i(\\rho_i), \\qquad \\chi = \\sum_i \\chi_i.

A graphical harmonium that is not conjugated can be fit variationally, as the underlying
harmonium of a :class:`~goal.geometry.exponential_family.variational.VariationalDifferentiable`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast, override

import jax.numpy as jnp
from jax import Array

from ..algebra.clique import shift_clique
from ..manifold.clique import CrossTerm, SubCliquesEmbedding
from ..manifold.util import split_by_dims
from .base import Analytic, Differentiable, Gibbs
from .combinators import DifferentiablePair
from .harmonium import (
    AnalyticConjugated,
    Conjugated,
    DifferentiableConjugated,
    Harmonium,
)

### Graphical Harmoniums ###


@dataclass(frozen=True)
class GraphicalHarmonium[Deep: Gibbs](Harmonium[Any, Deep], ABC):
    """A harmonium whose posterior is a latent model, with harmoniums attached to its cliques.

    A subclass declares the latent model (:attr:`pst_man`), the attached harmoniums
    and the clique each one is attached to (:attr:`obs_hrms_clqs`). The
    observable and the interaction are derived from the attached harmoniums.
    """

    # Contract

    @property
    @abstractmethod
    def obs_hrms_clqs(self) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        """The attached harmoniums, in order, each with the clique of the latent model its posterior is attached to.

        Node $j$ of the posterior is node ``clq[j]`` of the latent model. The observables of
        the harmoniums make up the composite's observable, in the same order.
        """

    # Overrides

    @property
    @override
    def obs_man(self) -> Any:
        """The observables of the harmoniums in :attr:`obs_hrms`, side by side.

        A single observable is returned as is; several are nested in :class:`ObservablePair`.
        """
        *init, last = (hrm.obs_man for hrm in self.obs_hrms)
        obs_man = last
        for fst in reversed(init):
            obs_man = ObservablePair(fst, obs_man)
        return obs_man

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """The crossings of every attached harmonium, renumbered into the composite, with their blocks.

        Observable nodes are offset by the nodes of the earlier observables, and latent nodes
        are mapped through that harmonium's clique. The blocks are used as they are: the
        posterior of an attached harmonium has the same blocks as its clique of the latent
        model.
        """
        trms: list[CrossTerm] = []
        n_rot = 0
        for hrm, clq in self.obs_hrms_clqs:
            for trm in hrm.crs_trms:
                cod_clq = shift_clique(trm.cod_clq, n_rot)
                dom_clq = tuple(clq[j] for j in trm.dom_clq)
                trms.append(CrossTerm(cod_clq, dom_clq, trm.clq_map))
            n_rot += hrm.obs_man.n_nodes
        return tuple(trms)

    # Methods

    @property
    def obs_hrms(self) -> tuple[Harmonium[Any, Any], ...]:
        """The attached harmoniums of :attr:`obs_hrms_clqs`."""
        return tuple(hrm for hrm, _ in self.obs_hrms_clqs)

    @property
    def obs_clqs(self) -> tuple[tuple[int, ...], ...]:
        """The cliques of :attr:`obs_hrms_clqs`."""
        return tuple(clq for _, clq in self.obs_hrms_clqs)

    def obs_likelihoods(self, lkl_params: Array) -> tuple[Array, ...]:
        """Split the likelihood natural parameters into those of each harmonium in :attr:`obs_hrms`."""
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)
        hrms = self.obs_hrms
        obs_blocks = split_by_dims(obs_params, tuple(hrm.obs_man.dim for hrm in hrms))
        int_blocks = split_by_dims(int_params, tuple(hrm.int_man.dim for hrm in hrms))
        return tuple(
            hrm.lkl_fun_man.join_coords(obs, int_)
            for hrm, obs, int_ in zip(hrms, obs_blocks, int_blocks)
        )

    def obs_pst_emb(self, index: int) -> SubCliquesEmbedding:
        """The embedding of the posterior of one harmonium in :attr:`obs_hrms` into the latent model, on its clique."""
        hrm, clq = self.obs_hrms_clqs[index]
        return SubCliquesEmbedding(clq, self.pst_man, hrm.pst_man)


class DifferentiableGraphical[Deep: Differentiable, PriorDeep: Differentiable](
    GraphicalHarmonium[Deep],
    DifferentiableConjugated[Any, Deep, PriorDeep],
    ABC,
):
    """A graphical harmonium whose attached harmoniums are all conjugated, which makes it conjugated.

    Its conjugation parameters are the sum of the attached harmoniums', each placed on its
    clique of the prior (:meth:`place_conjugation`), and its offset is the sum of theirs.
    The prior must contain each attached harmonium's prior on its clique.
    """

    # Overrides

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Sum the attached harmoniums' conjugation parameters, each placed on its clique of the prior."""
        rhos = tuple(
            hrm.conjugation_parameters(lkl)
            for hrm, lkl in zip(self.cnj_obs_hrms, self.obs_likelihoods(lkl_params))
        )
        return self.place_conjugation(rhos)

    @override
    def conjugation_offset(self, lkl_params: Array) -> Array:
        """Sum the attached harmoniums' conjugation offsets."""
        chis = [
            hrm.conjugation_offset(lkl)
            for hrm, lkl in zip(self.cnj_obs_hrms, self.obs_likelihoods(lkl_params))
        ]
        return jnp.sum(jnp.stack(chis))

    # Methods

    @property
    def cnj_obs_hrms(self) -> tuple[Conjugated[Any, Any, Any], ...]:
        """The harmoniums in :attr:`obs_hrms`, typed as conjugated."""
        return cast(tuple[Conjugated[Any, Any, Any], ...], self.obs_hrms)

    def place_conjugation(self, rhos: tuple[Array, ...]) -> Array:
        """Embed each attached harmonium's conjugation parameters into the prior on its clique, and sum.

        Each entry of ``rhos`` is in the coordinates of its harmonium's prior.
        """
        out = self.prr_man.zeros()
        for hrm, clq, rho in zip(self.cnj_obs_hrms, self.obs_clqs, rhos):
            out = out + SubCliquesEmbedding(clq, self.prr_man, hrm.prr_man).embed(rho)
        return out


class AnalyticGraphical[Deep: Analytic](
    AnalyticConjugated[Any, Deep],
    DifferentiableGraphical[Deep, Deep],
    ABC,
):
    """A graphical harmonium with one analytic attached harmonium and an analytic latent model, whose posterior and prior coincide."""

    # Overrides

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to likelihood natural parameters through the attached harmonium."""
        (hrm,) = self.obs_hrms
        hrm = cast(AnalyticConjugated[Any, Any], hrm)
        obs_means, int_means, lat_means = self.split_coords(means)
        att_means = hrm.join_coords(
            obs_means, int_means, self.obs_pst_emb(0).project(lat_means)
        )
        return hrm.to_natural_likelihood(att_means)


### Observables ###


@dataclass(frozen=True)
class ObservablePair[First: Differentiable, Second: Differentiable](
    DifferentiablePair[First, Second]
):
    """Two observables side by side, over disjoint slices of the data.

    Three or more observables are nested to the right: ``ObservablePair(a, ObservablePair(b, c))``.
    """

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
