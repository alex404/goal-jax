"""Latent process population code: a variational filter over a Poisson population code.

The latent state follows a learned (deterministic) map on beliefs, and each observation is a
vector of spike counts from a :class:`~goal.models.PoissonPopulationCode`. Filtering alternates
the prediction of the transition with the conjugate update of the population code, which is
exact up to its conjugation residual.
"""

from dataclasses import dataclass
from typing import Any, override

from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.combinators import AnalyticTuple
from ...geometry.exponential_family.dynamical import VariationalLatentProcess
from ...geometry.manifold.map import Map
from ..base.poisson import Poissons
from ..harmonium.population_codes import PoissonPopulationCode


@dataclass(frozen=True)
class PopulationCodeProcess[Latent: Differentiable, Transition: Map[Any, Any]](
    VariationalLatentProcess[AnalyticTuple[Poissons], Latent, Latent]
):
    """State-space model with a Poisson population code emission and a transition map on beliefs.

    The transition maps natural parameters of the latent family to natural parameters of the
    latent family, independently of the observations; it must return valid parameters, which
    is the caller's choice of map (e.g. a :class:`~goal.geometry.MultilayerPerceptron`, or a
    subclass that keeps a precision positive).
    """

    # Fields

    population_code: PoissonPopulationCode[Latent]
    transition: Transition

    # Overrides

    @property
    @override
    def lat_man(self) -> Latent:
        return self.population_code.hrm.pst_man

    @property
    @override
    def ems_hrm(self) -> PoissonPopulationCode[Latent]:
        return self.population_code

    @property
    @override
    def trn_map(self) -> Transition:
        return self.transition
