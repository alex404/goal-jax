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

A graphical harmonium makes no assumption of conjugation: it is the joint family and its
two conditionals. One that is not conjugated can be fit variationally, as the
``gen_hrm`` of a :class:`~goal.geometry.exponential_family.variational.VariationalConjugated`.
A plain harmonium is given to the variational classes as an :class:`AttachedHarmonium`.
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
    """A harmonium whose posterior is a latent model, with harmoniums attached to its cliques.

    A subclass declares the latent model (:attr:`pst_man`), the attached harmoniums
    and the clique each one is attached to (:attr:`obs_hrms_att_clqs`). The
    observable and the interaction are derived from the attached harmoniums, whose
    observables must be differentiable.
    """

    # Contract

    @property
    @abstractmethod
    def obs_hrms_att_clqs(
        self,
    ) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        """The attached harmoniums, in order, each with the clique of the latent model its posterior is attached to.

        Node $j$ of the posterior is node ``att_clq[j]`` of the latent model. The observables of
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
        """The crossings of every attached harmonium, renumbered into the composite, with their coordinate blocks.

        Observable nodes are offset as in :attr:`obs_man`, and latent nodes are mapped
        through that harmonium's clique. The coordinate blocks are used as they are: the
        posterior of an attached harmonium has the same coordinate blocks as its clique of the
        latent model.
        """
        trms: list[CrossTerm] = []
        for (obs_hrm, att_clq), nod_offset in zip(
            self.obs_hrms_att_clqs, self.obs_man.elm_nod_offsets
        ):
            for trm in obs_hrm.crs_trms:
                cod_clq = shift_clique(trm.cod_clq, nod_offset)
                dom_clq = tuple(att_clq[j] for j in trm.dom_clq)
                trms.append(CrossTerm(cod_clq, dom_clq, trm.clq_map))
        return tuple(trms)

    # Methods

    @property
    def obs_hrms(self) -> tuple[Harmonium[Any, Any], ...]:
        """The attached harmoniums of :attr:`obs_hrms_att_clqs`."""
        return tuple(obs_hrm for obs_hrm, _ in self.obs_hrms_att_clqs)

    @property
    def att_clqs(self) -> tuple[tuple[int, ...], ...]:
        """The clique of the latent model that each harmonium in :attr:`obs_hrms` is attached to."""
        return tuple(att_clq for _, att_clq in self.obs_hrms_att_clqs)

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
class AttachedHarmonium[Deep: Gibbs](GraphicalHarmonium[Deep]):
    """One harmonium as a graphical harmonium, its posterior attached to the leading nodes of a latent model.

    The latent model is either the harmonium's own posterior, which makes this a plain harmonium
    seen as a graphical one, or a larger model whose leading nodes carry the posterior's cliques,
    e.g. the joint model of the next level of a deep model, whose leading nodes are its
    observable. The parameter layout is that of the harmonium followed by the rest of the latent
    model. It lets a harmonium be used where a graphical harmonium is expected, e.g. as the
    ``gen_hrm`` of a :class:`~goal.geometry.exponential_family.variational.VariationalConjugated`.
    """

    # Fields

    att_hrm: Harmonium[Any, Any]
    lat_man: Deep

    def __post_init__(self) -> None:
        pst_man = self.att_hrm.pst_man
        for clique in pst_man.cliques:
            if (
                clique not in self.lat_man.cliques
                or self.lat_man.clq_man(clique) != pst_man.clq_man(clique)
            ):
                raise ValueError(
                    f"Clique {clique} of the posterior is not a clique of the latent model with the same coordinate block"
                )

    # Overrides

    @property
    @override
    def pst_man(self) -> Deep:
        return self.lat_man

    @property
    @override
    def obs_hrms_att_clqs(
        self,
    ) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        return ((self.att_hrm, tuple(range(self.att_hrm.pst_man.n_nodes))),)


class DifferentiableGraphical[Deep: Differentiable, PriorDeep: Differentiable](
    GraphicalHarmonium[Deep],
    DifferentiableConjugated[Any, Deep, PriorDeep],
    ABC,
):
    """A graphical harmonium whose attached harmoniums are all conjugated, which makes it conjugated.

    Its conjugation parameters are the sum of the attached harmoniums', each placed on its
    clique of the prior, and its offset is the sum of theirs.
    The prior must contain each attached harmonium's prior on its clique.
    """

    # Overrides

    @property
    @override
    def obs_hrms(self) -> tuple[Conjugated[Any, Any, Any], ...]:
        """The attached harmoniums, typed as conjugated."""
        return cast(tuple[Conjugated[Any, Any, Any], ...], super().obs_hrms)

    @property
    @override
    def obs_man(self) -> DifferentiableTuple[Any]:
        """The observables of the attached harmoniums, as the elements of a differentiable tuple."""
        return DifferentiableTuple(tuple(obs_hrm.obs_man for obs_hrm in self.obs_hrms))

    @override
    def conjugation_parameters(self, lkl_params: Array) -> Array:
        """Sum the attached harmoniums' conjugation parameters, each embedded into the prior on its clique."""
        lkl_paramss = self.likelihood_functions(lkl_params)
        rho = self.prr_man.zeros()
        for obs_hrm, att_clq, hrm_lkl_params in zip(
            self.obs_hrms, self.att_clqs, lkl_paramss
        ):
            prr_emb = SubCliquesEmbedding(att_clq, self.prr_man, obs_hrm.prr_man)
            rho = rho + prr_emb.embed(obs_hrm.conjugation_parameters(hrm_lkl_params))
        return rho

    @override
    def conjugation_offset(self, lkl_params: Array) -> Array:
        """Sum the attached harmoniums' conjugation offsets."""
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
    """A graphical harmonium with analytic attached harmoniums and an analytic latent model, whose posterior and prior coincide."""

    # Overrides

    @property
    @override
    def obs_hrms(self) -> tuple[AnalyticConjugated[Any, Any], ...]:
        """The attached harmoniums, typed as analytic."""
        return cast(tuple[AnalyticConjugated[Any, Any], ...], super().obs_hrms)

    @property
    @override
    def obs_man(self) -> AnalyticTuple[Any]:
        """The observables of the attached harmoniums, as the elements of an analytic tuple."""
        return AnalyticTuple(tuple(obs_hrm.obs_man for obs_hrm in self.obs_hrms))

    @override
    def to_natural_likelihood(self, means: Array) -> Array:
        """Convert mean parameters to likelihood natural parameters, harmonium by harmonium.

        Given the latent model the observables are independent, so the expected
        log-likelihood is a sum over the attached harmoniums, and each harmonium's
        likelihood is computed from its own observable and interaction coordinates and the
        latent mean parameters on its clique. Projecting the latent mean parameters onto a
        clique is valid only in mean coordinates, where the coordinates of a clique are the
        expected sufficient statistics of that clique, and so the mean parameters of the
        attached harmonium's posterior.
        """
        obs_means, int_means, lat_means = self.split_coords(means)
        obs_meanss = self.obs_man.split_coords(obs_means)
        int_dims = tuple(obs_hrm.int_man.dim for obs_hrm in self.obs_hrms)
        int_meanss = split_by_dims(int_means, int_dims)
        obs_paramss: list[Array] = []
        int_paramss: list[Array] = []
        for obs_hrm, att_clq, hrm_obs_means, hrm_int_means in zip(
            self.obs_hrms, self.att_clqs, obs_meanss, int_meanss
        ):
            pst_emb = SubCliquesEmbedding(att_clq, self.pst_man, obs_hrm.pst_man)
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
