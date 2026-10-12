"""Population codes as harmoniums: Poisson or Boltzmann neurons over any latent family, and Poisson mixture models.

The harmoniums here are exact families. The variational models built on them are in
:mod:`goal.models.variational`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

from ...geometry import (
    CliqueMap,
    CrossTerm,
    DifferentiableTuple,
    GraphicalHarmonium,
    IdentityEmbedding,
    LinearEmbedding,
    Rectangular,
)
from ...geometry.exponential_family.base import Differentiable, ExponentialFamily
from ...geometry.exponential_family.harmonium import Harmonium
from ..base.gaussian.boltzmann import Boltzmann
from ..base.poisson import CoMPoissons, Poissons, PopulationLocationEmbedding
from .lgm import GeneralizedGaussianLocationEmbedding
from .mixture import AnalyticMixture, Mixture

# --- Population codes ---


@dataclass(frozen=True)
class PopulationCode[Observable: Differentiable, Latent: Differentiable](
    Harmonium[Observable, Latent], ABC
):
    """Harmonium with a population of neurons as its observable, coupled to every clique of a latent model.

    A subclass declares the population (:attr:`obs_man`), the latent model (:attr:`pst_man`) and
    the statistics of the neurons that the latent modulates (:attr:`nrn_emb`). Each clique of the
    latent model is coupled to those statistics by one crossing over its whole coordinate block, so
    the neurons depend on every sufficient statistic of the latent. On a latent whose cliques are
    its nodes, e.g. a tuple of independent families, each tuning curve is a product over the nodes;
    a clique of several nodes adds terms that couple them. Subpopulations tuned to different nodes
    are a :class:`GraphicalPopulationCode`.

    Mathematically, given $z$ the neurons have natural parameters $\\theta_N + \\iota(\\Theta_{NZ}
    \\cdot \\mathbf s_Z(z))$, with $\\iota$ the embedding :attr:`nrn_emb`, so the shape of each
    tuning curve is set by the sufficient statistics of the latent: a von Mises node gives a cosine
    bump in an angle, a Normal node a Gaussian bump.
    """

    # Fields

    latent: Latent

    # Contract

    @property
    @abstractmethod
    def nrn_emb(self) -> LinearEmbedding[Observable, Any]:
        """The statistics of the neurons that the latent modulates, as a subspace of the population."""

    # Overrides

    @property
    @override
    def pst_man(self) -> Latent:
        return self.latent

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """One crossing per clique of the latent, over its whole coordinate block."""
        return tuple(
            CrossTerm(
                (0,),
                clique,
                CliqueMap(
                    Rectangular(),
                    self.nrn_emb,
                    IdentityEmbedding(self.pst_man.clq_man(clique)),
                ),
            )
            for clique in self.pst_man.cliques
        )


@dataclass(frozen=True)
class PoissonPopulationCode[Latent: Differentiable](PopulationCode[Poissons, Latent]):
    """Population code of independent Poisson neurons, whose log-rates the latent modulates."""

    # Fields

    n_neurons: int

    # Overrides

    @property
    @override
    def obs_man(self) -> Poissons:
        return Poissons(self.n_neurons)

    @property
    @override
    def nrn_emb(self) -> IdentityEmbedding[Poissons]:
        return IdentityEmbedding(self.obs_man)


@dataclass(frozen=True)
class BoltzmannPopulationCode[Shape: ExponentialFamily, Latent: Differentiable](
    PopulationCode[Boltzmann[Shape], Latent]
):
    """Population code of binary neurons with noise correlations, given by a Boltzmann machine.

    The latent modulates the biases of the neurons only, so their couplings, and with them the
    noise correlations, do not depend on the latent. The neurons are a field because a
    :class:`~goal.models.base.gaussian.boltzmann.ChordalBoltzmann` carries its junction tree.
    """

    # Fields

    neurons: Boltzmann[Shape]

    # Overrides

    @property
    @override
    def obs_man(self) -> Boltzmann[Shape]:
        return self.neurons

    @property
    @override
    def nrn_emb(self) -> GeneralizedGaussianLocationEmbedding[Boltzmann[Shape]]:
        return GeneralizedGaussianLocationEmbedding(self.neurons)


@dataclass(frozen=True)
class GraphicalPopulationCode[Latent: Differentiable](GraphicalHarmonium[Latent]):
    """Subpopulations over one latent model, each a :class:`PopulationCode` attached to some of its nodes.

    The observable is the tuple of the subpopulations, and a subpopulation has no coupling to the
    nodes it is not attached to.
    """

    # Fields

    latent: Latent

    codes_att_clqs: tuple[tuple[PopulationCode[Any, Any], tuple[int, ...]], ...]
    """The subpopulations, each with the nodes of the latent model its own latent is attached to."""

    # Overrides

    @property
    @override
    def pst_man(self) -> Latent:
        return self.latent

    @property
    @override
    def obs_hrms_att_clqs(
        self,
    ) -> tuple[tuple[Harmonium[Any, Any], tuple[int, ...]], ...]:
        return self.codes_att_clqs

    @property
    @override
    def obs_man(self) -> DifferentiableTuple[Any]:
        """The populations of the subpopulations, as the elements of a tuple."""
        return DifferentiableTuple(tuple(obs_hrm.obs_man for obs_hrm in self.obs_hrms))


# --- COM-Poisson Population ---

type PoissonMixture = AnalyticMixture[Poissons]
type CoMPoissonMixture = Mixture[CoMPoissons]


def poisson_mixture(n_neurons: int, n_components: int) -> PoissonMixture:
    """Create a mixture of independent Poisson populations."""
    pop_man = Poissons(n_neurons)
    return AnalyticMixture(pop_man, n_components)


def com_poisson_mixture(n_neurons: int, n_components: int) -> CoMPoissonMixture:
    """Create a COM-Poisson mixture with shared dispersion parameters."""
    obs_emb = PopulationLocationEmbedding(n_neurons)
    return Mixture(n_components, obs_emb)
