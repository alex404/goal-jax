"""Variational population codes: a population code fit with learned, constant conjugation parameters, and a filter over one.

The population codes themselves are exact harmoniums, in
:mod:`goal.models.harmonium.population_codes`. A :class:`PopulationCodeProcess` is a latent process
whose state follows a learned (deterministic) map on beliefs, with each observation the response of
the population code. Filtering alternates the prediction of the transition with the conjugate update
of the population code, which is exact up to its conjugation residual.
"""

from dataclasses import dataclass
from typing import Any, override

from jax import Array

from ...geometry import AttachedHarmonium, IdentityEmbedding
from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.dynamical import VariationalLatentProcess
from ...geometry.exponential_family.variational import (
    DifferentiableVariationalConjugated,
)
from ...geometry.manifold.map import Map
from ..harmonium.population_codes import PopulationCode


@dataclass(frozen=True)
class VariationalPopulationCode[Latent: Differentiable](
    DifferentiableVariationalConjugated[AttachedHarmonium[Latent], Latent, Latent]
):
    """A population code with constant conjugation parameters.

    Posterior = prior = conjugation = the latent family, so :meth:`conjugation_parameters`
    returns the stored $\\rho$.
    """

    # Fields

    pop_hrm: PopulationCode[Any, Latent]

    # Overrides

    @property
    @override
    def gen_hrm(self) -> AttachedHarmonium[Latent]:
        return AttachedHarmonium(self.pop_hrm, self.pop_hrm.pst_man)

    @property
    @override
    def cnj_fun_man(self) -> Latent:
        return self.pop_hrm.pst_man

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Latent]:
        return IdentityEmbedding(self.pop_hrm.pst_man)

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        return cnj_fun_params


@dataclass(frozen=True)
class PopulationCodeProcess[Latent: Differentiable, Transition: Map[Any, Any]](
    VariationalLatentProcess[Any, Latent, Latent]
):
    """State-space model with a population code emission and a transition map on beliefs.

    The transition maps natural parameters of the latent family to natural parameters of the
    latent family, independently of the observations; it must return valid parameters, which
    is the caller's choice of map (e.g. a :class:`~goal.geometry.MultilayerPerceptron`, or a
    subclass that keeps a precision positive).
    """

    # Fields

    pop_man: VariationalPopulationCode[Latent]
    _trn_map: Transition

    # Overrides

    @property
    @override
    def lat_man(self) -> Latent:
        return self.pop_man.pop_hrm.pst_man

    @property
    @override
    def ems_hrm(self) -> VariationalPopulationCode[Latent]:
        return self.pop_man

    @property
    @override
    def trn_map(self) -> Transition:
        return self._trn_map
