"""Variational population codes: a population code fit with learned, constant conjugation parameters.

The population codes themselves are exact harmoniums, in
:mod:`goal.models.harmonium.population_codes`.
"""

from dataclasses import dataclass
from typing import Any, override

from jax import Array

from ...geometry import AttachedHarmonium, IdentityEmbedding
from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.variational import (
    DifferentiableVariationalConjugated,
)
from ..harmonium.population_codes import GraphicalPopulationCode, PopulationCode


@dataclass(frozen=True)
class VariationalPopulationCode[Latent: Differentiable](
    DifferentiableVariationalConjugated[AttachedHarmonium[Latent], Latent, Latent]
):
    """A population code, or several subpopulations over one latent, with constant conjugation parameters.

    Posterior = prior = conjugation = the latent family, so :meth:`conjugation_parameters`
    returns the stored $\\rho$.
    """

    # Fields

    hrm: PopulationCode[Any, Latent] | GraphicalPopulationCode[Latent]

    # Overrides

    @property
    @override
    def gen_hrm(self) -> AttachedHarmonium[Latent]:
        return AttachedHarmonium(self.hrm, self.hrm.pst_man)

    @property
    @override
    def cnj_fun_man(self) -> Latent:
        return self.hrm.pst_man

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[Latent]:
        return IdentityEmbedding(self.hrm.pst_man)

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        return cnj_fun_params
