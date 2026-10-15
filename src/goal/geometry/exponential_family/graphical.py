"""Graphical harmoniums: harmoniums placed on the cliques of a latent model.

A :class:`GraphicalHarmonium` is a latent model with observable harmoniums, each placed on a clique
of it; their observables make up its observable and are independent given the latent model. It
assumes no conjugation. A :class:`DifferentiableGraphical` is conjugated because every observable
harmonium is, and one that is not can be fit variationally by a
:class:`~goal.geometry.exponential_family.variational.VariationalConjugated`. A
:class:`GraphicalLevel` is a graphical harmonium assembled from parts rather than declared as a
subclass. A hierarchical mixture of Gaussians places one linear Gaussian model on a mixture;
canonical correlation analysis places two on one normal.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, cast, override

import jax.numpy as jnp
from jax import Array

from ..algebra.clique import shift_clique
from ..manifold.clique import CrossTerm, SubCliquesEmbedding
from ..manifold.util import split_by_dims
from .base import Analytic, Differentiable, Gibbs
from .combinators import AnalyticTuple, DifferentiableTuple, GenerativeTuple
from .harmonium import (
    AnalyticConjugated,
    Conjugated,
    DifferentiableConjugated,
    Harmonium,
)

### Graphical Harmoniums ###


@dataclass(frozen=True)
class GraphicalHarmonium[Deep: Gibbs](Harmonium[Any, Deep], ABC):
    """A harmonium whose posterior is a latent model, with harmoniums placed on its cliques.

    A subclass declares the latent model (:attr:`pst_man`), the observable harmoniums
    and the nodes each one is placed on (:attr:`obs_hrms_lat_nods`). The
    observable and the interaction are derived from the observable harmoniums, whose
    observables must be differentiable.
    """

    # Contract

    @property
    @abstractmethod
    def obs_hrms_lat_nods(
        self,
    ) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        """The observable harmoniums, in order, each with the nodes of the latent model its posterior is placed on.

        Node $j$ of the posterior is node ``lat_nods[j]`` of the latent model. The observables of
        the harmoniums make up the composite's observable, in the same order.
        """

    # Overrides

    @property
    @override
    def obs_man(self) -> GenerativeTuple[Any]:
        """The observables of the harmoniums in :attr:`obs_hrms`, as the elements of a tuple, also when there is only one."""
        return GenerativeTuple(tuple(obs_hrm.obs_man for obs_hrm in self.obs_hrms))

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """The crossings of every observable harmonium, renumbered into the composite, with their coordinate blocks.

        Observable nodes are offset as in :attr:`obs_man`, and latent nodes are mapped
        through that harmonium's clique. The coordinate blocks are used as they are: the
        posterior of an observable harmonium has the same coordinate blocks as its clique of the
        latent model.
        """
        trms: list[CrossTerm] = []
        for (obs_hrm, lat_nods), nod_offset in zip(
            self.obs_hrms_lat_nods, self.obs_man.elm_nod_offsets
        ):
            for trm in obs_hrm.crs_trms:
                cod_clq = shift_clique(trm.cod_clq, nod_offset)
                dom_clq = tuple(lat_nods[j] for j in trm.dom_clq)
                trms.append(CrossTerm(cod_clq, dom_clq, trm.clq_map))
        return tuple(trms)

    # Methods

    @property
    def obs_hrms(self) -> tuple[Harmonium[Any, Any], ...]:
        """The observable harmoniums of :attr:`obs_hrms_lat_nods`."""
        return tuple(obs_hrm for obs_hrm, _ in self.obs_hrms_lat_nods)

    @property
    def lat_nodss(self) -> tuple[tuple[int, ...], ...]:
        """The nodes of the latent model that each harmonium in :attr:`obs_hrms` is placed on."""
        return tuple(lat_nods for _, lat_nods in self.obs_hrms_lat_nods)

    def likelihood_functions(self, lkl_params: Array) -> tuple[Array, ...]:
        """Split the likelihood parameters into those of each harmonium in :attr:`obs_hrms`.

        The observable biases are split into the elements of :attr:`obs_man`, and the
        interaction into those of each harmonium. They are contiguous because a
        :class:`~goal.geometry.manifold.clique.CrossMap` stores its coordinates term by
        term and :attr:`crs_trms` lists the terms harmonium by harmonium.
        """
        obs_params, int_params = self.lkl_fun_man.split_coords(lkl_params)
        obs_paramss = self.obs_man.split_coords(obs_params)
        int_dims = tuple(obs_hrm.int_man.dim for obs_hrm in self.obs_hrms)
        int_paramss = split_by_dims(int_params, int_dims)
        return tuple(
            obs_hrm.lkl_fun_man.join_coords(hrm_obs_params, hrm_int_params)
            for obs_hrm, hrm_obs_params, hrm_int_params in zip(
                self.obs_hrms, obs_paramss, int_paramss
            )
        )


@dataclass(frozen=True)
class GraphicalLevel[Deep: Gibbs](GraphicalHarmonium[Deep]):
    """A graphical harmonium given by its parts: harmoniums placed on cliques of a latent model.

    It is the generic form of a graphical harmonium, for models assembled from their parts rather
    than declared as a subclass. The latent model may be a harmonium's own posterior, which makes a
    plain harmonium a graphical one, or a larger model, e.g. the joint model of the next level of a
    deep model, whose nodes carry the posteriors' cliques. It lets harmoniums be used where a
    graphical harmonium is expected, e.g. as the ``gph_hrm`` of a
    :class:`~goal.geometry.exponential_family.variational.VariationalConjugated`.
    """

    # Fields

    _obs_hrms_lat_nods: tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]
    _pst_man: Deep

    # Overrides

    @property
    @override
    def pst_man(self) -> Deep:
        return self._pst_man

    @property
    @override
    def obs_hrms_lat_nods(
        self,
    ) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        return self._obs_hrms_lat_nods


class DifferentiableGraphical[Deep: Differentiable, PriorDeep: Differentiable](
    GraphicalHarmonium[Deep],
    DifferentiableConjugated[Any, Deep, PriorDeep],
    ABC,
):
    """A graphical harmonium whose observable harmoniums are all conjugated, which makes it conjugated.

    Its conjugation parameters are the sum of the observable harmoniums', each placed on its
    clique of the prior, and its offset is the sum of theirs.
    The prior must contain each observable harmonium's prior on its clique.
    """

    # Overrides

    @property
    @override
    def obs_hrms(self) -> tuple[Conjugated[Any, Any, Any], ...]:
        """The observable harmoniums, typed as conjugated."""
        return cast(tuple[Conjugated[Any, Any, Any], ...], super().obs_hrms)

    @property
    @override
    def obs_man(self) -> DifferentiableTuple[Any]:
        """The observables of the observable harmoniums, as the elements of a differentiable tuple."""
        return DifferentiableTuple(tuple(obs_hrm.obs_man for obs_hrm in self.obs_hrms))

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Sum the observable harmoniums' conjugation parameters, each embedded into the prior on its clique."""
        lkl_paramss = self.likelihood_functions(lkl_params)
        rho = self.prr_man.zeros()
        for obs_hrm, lat_nods, hrm_lkl_params in zip(
            self.obs_hrms, self.lat_nodss, lkl_paramss
        ):
            prr_emb = SubCliquesEmbedding(lat_nods, self.prr_man, obs_hrm.prr_man)
            rho = rho + prr_emb.embed(obs_hrm.conjugation_parameters(hrm_lkl_params))
        return rho

    @override
    def conjugation_offset(self, lkl_params: Array) -> Array:
        """Sum the observable harmoniums' conjugation offsets."""
        lkl_paramss = self.likelihood_functions(lkl_params)
        return sum(
            (
                obs_hrm.conjugation_offset(hrm_lkl_params)
                for obs_hrm, hrm_lkl_params in zip(self.obs_hrms, lkl_paramss)
            ),
            start=jnp.asarray(0.0),
        )


class AnalyticGraphical[Deep: Analytic](
    AnalyticConjugated[Any, Deep],
    DifferentiableGraphical[Deep, Deep],
    ABC,
):
    """A graphical harmonium with analytic observable harmoniums and an analytic latent model, whose posterior and prior coincide."""

    # Overrides

    @property
    @override
    def obs_hrms(self) -> tuple[AnalyticConjugated[Any, Any], ...]:
        """The observable harmoniums, typed as analytic."""
        return cast(tuple[AnalyticConjugated[Any, Any], ...], super().obs_hrms)

    @property
    @override
    def obs_man(self) -> AnalyticTuple[Any]:
        """The observables of the observable harmoniums, as the elements of an analytic tuple."""
        return AnalyticTuple(tuple(obs_hrm.obs_man for obs_hrm in self.obs_hrms))

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to likelihood natural parameters, harmonium by harmonium.

        Given the latent model the observables are independent, so the expected
        log-likelihood is a sum over the observable harmoniums, and each harmonium's
        likelihood is computed from its own observable and interaction coordinates and the
        latent mean parameters on its clique. Projecting the latent mean parameters onto a
        clique is valid only in mean coordinates, where the coordinates of a clique are the
        expected sufficient statistics of that clique, and so the mean parameters of the
        observable harmonium's posterior.
        """
        obs_means, int_means, lat_means = self.split_coords(means)
        obs_meanss = self.obs_man.split_coords(obs_means)
        int_dims = tuple(obs_hrm.int_man.dim for obs_hrm in self.obs_hrms)
        int_meanss = split_by_dims(int_means, int_dims)
        obs_paramss: list[Array] = []
        int_paramss: list[Array] = []
        for obs_hrm, lat_nods, hrm_obs_means, hrm_int_means in zip(
            self.obs_hrms, self.lat_nodss, obs_meanss, int_meanss
        ):
            pst_emb = SubCliquesEmbedding(lat_nods, self.pst_man, obs_hrm.pst_man)
            pst_means = pst_emb.project(lat_means)
            hrm_means = obs_hrm.join_coords(hrm_obs_means, hrm_int_means, pst_means)
            hrm_lkl_params = obs_hrm.to_natural_likelihood(hrm_means)
            hrm_obs_params, hrm_int_params = obs_hrm.lkl_fun_man.split_coords(
                hrm_lkl_params
            )
            obs_paramss.append(hrm_obs_params)
            int_paramss.append(hrm_int_params)
        return self.lkl_fun_man.join_coords(
            self.obs_man.join_coords(*obs_paramss), jnp.concatenate(int_paramss)
        )
