"""Variational population codes: a population code fit with learned, constant conjugation parameters, and a filter over one.

The population codes themselves are exact harmoniums, in
:mod:`goal.models.harmonium.population_codes`. A :class:`PopulationCodeProcess` is a latent process
whose state follows a learned (deterministic) map on beliefs, with each observation the response of
the population code. Filtering alternates the prediction of the transition with the conjugate update
of the population code, which is exact up to its conjugation residual.
"""

from dataclasses import dataclass
from typing import Any, override

from ...geometry import GraphicalLevel
from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.dynamical import VariationalLatentProcess
from ...geometry.exponential_family.variational import (
    ConstantConjugationFunction,
    DifferentiableDeepModel,
    DifferentiableVariationalConjugated,
)
from ...geometry.manifold.map import Map
from ..harmonium.population_codes import PopulationCode

type VariationalPopulationCode[Latent: Differentiable] = (
    DifferentiableVariationalConjugated[GraphicalLevel[Latent], Latent]
)
"""A population code with constant conjugation parameters (:func:`variational_population_code`)."""


def variational_population_code[Latent: Differentiable](
    pop_hrm: PopulationCode[Any, Latent],
) -> VariationalPopulationCode[Latent]:
    """A population code with constant conjugation parameters: its conjugation function is a :class:`~goal.geometry.exponential_family.variational.ConstantConjugationFunction`, whose parameters are $\\rho$."""
    lat = pop_hrm.pst_man
    return DifferentiableVariationalConjugated(
        GraphicalLevel(((pop_hrm, tuple(range(lat.n_nodes))),), lat),
        (ConstantConjugationFunction(lat, pop_hrm.lkl_fun_man),),
        DifferentiableDeepModel(lat),
    )


@dataclass(frozen=True)
class PopulationCodeProcess[Latent: Differentiable, Transition: Map[Any, Any]](
    VariationalLatentProcess[Any, Latent]
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
        return self.pop_man.pst_man

    @property
    @override
    def ems_hrm(self) -> VariationalPopulationCode[Latent]:
        return self.pop_man

    @property
    @override
    def trn_map(self) -> Transition:
        return self._trn_map
