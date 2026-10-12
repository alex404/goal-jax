"""Variational population codes: a population code harmonium fit with learned, constant conjugation parameters.

The harmoniums are in :mod:`goal.models.harmonium.population_codes`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from jax import Array

from ...geometry import AttachedHarmonium, IdentityEmbedding
from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.variational import (
    DifferentiableVariationalConjugated,
)
from ..base.categorical import Bernoullis
from ..base.gaussian.boltzmann import (
    ChordalBoltzmann,
    ChordalCouplingMatrix,
    DiagonalBoltzmann,
)
from ..base.gaussian.normal import FullNormal
from ..harmonium.population_codes import (
    BoltzmannNormalHarmonium,
    PoissonPopulationHarmonium,
)


@dataclass(frozen=True)
class PoissonPopulationCode[Latent: Differentiable](
    DifferentiableVariationalConjugated[AttachedHarmonium[Latent], Latent, Latent]
):
    """Variational population code of a :class:`PoissonPopulationHarmonium`, with constant conjugation parameters.

    Posterior = prior = conjugation = the latent family, so :meth:`conjugation_parameters`
    returns the stored $\\rho$.
    """

    # Fields

    hrm: PoissonPopulationHarmonium[Latent]

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


@dataclass(frozen=True)
class BoltzmannPopulationCode[Shape: Differentiable](
    DifferentiableVariationalConjugated[
        AttachedHarmonium[FullNormal], FullNormal, FullNormal
    ]
):
    """Variational population code with a Boltzmann observable and a Normal latent.

    Posterior = prior = conjugation = ``FullNormal``, so :meth:`conjugation_parameters`
    returns the stored $\\rho$. Trained via the variational ELBO with the conjugation
    residual regularized.
    """

    # Fields

    hrm: BoltzmannNormalHarmonium[Shape]

    # Overrides

    @property
    @override
    def gen_hrm(self) -> AttachedHarmonium[FullNormal]:
        return AttachedHarmonium(self.hrm, self.hrm.pst_man)

    @property
    @override
    def cnj_fun_man(self) -> FullNormal:
        return self.hrm.pst_man

    @property
    @override
    def pst_prr_emb(self) -> IdentityEmbedding[FullNormal]:
        return IdentityEmbedding(self.hrm.pst_man)

    @override
    def conjugation_parameters(self, lkl_params: Array, cnj_fun_params: Array) -> Array:
        return cnj_fun_params

    # Methods

    @property
    def n_neurons(self) -> int:
        """Number of Boltzmann observable neurons."""
        return self.hrm.boltzmann.data_dim

    @property
    def n_latent(self) -> int:
        """Dimension of the Gaussian latent."""
        return self.hrm.lat_dim


def chordal_boltzmann_population_code(
    n_neurons: int,
    edges: Sequence[tuple[int, int]],
    lat_dim: int,
    max_treewidth: int | None = None,
) -> BoltzmannPopulationCode[ChordalCouplingMatrix]:
    """Population code whose observable is a chordal Boltzmann machine.

    ``edges`` seeds the chordal graph; triangulation fill-in becomes genuine
    couplings (see :meth:`ChordalBoltzmann.from_edges`).
    """
    boltzmann = ChordalBoltzmann.from_edges(n_neurons, edges, max_treewidth)
    return BoltzmannPopulationCode(BoltzmannNormalHarmonium(boltzmann, lat_dim))


def diagonal_boltzmann_population_code(
    n_neurons: int, lat_dim: int
) -> BoltzmannPopulationCode[Bernoullis]:
    """Population code whose observable is an independent-Bernoulli baseline."""
    boltzmann = DiagonalBoltzmann(n_neurons=n_neurons)
    return BoltzmannPopulationCode(BoltzmannNormalHarmonium(boltzmann, lat_dim))
