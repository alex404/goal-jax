"""Population codes as harmoniums: Poisson populations over any latent family, Boltzmann neurons over a Normal latent, and Poisson mixture models.

The harmoniums here are exact families. The variational models built on them are in
:mod:`goal.models.variational`.
"""

from dataclasses import dataclass
from typing import Any, override

from ...geometry import (
    CliqueMap,
    CrossTerm,
    IdentityEmbedding,
    Rectangular,
)
from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.combinators import AnalyticTuple
from ...geometry.exponential_family.harmonium import Harmonium
from ..base.gaussian.boltzmann import Boltzmann
from ..base.gaussian.normal import (
    DiagonalNormal,
    FullNormal,
    diagonal_normal,
    full_normal,
)
from ..base.poisson import CoMPoissons, Poissons, PopulationLocationEmbedding
from .lgm import GeneralizedGaussianLocationEmbedding
from .mixture import AnalyticMixture, Mixture

# --- Poisson populations (any latent family) ---


@dataclass(frozen=True)
class PoissonPopulationHarmonium[Latent: Differentiable](
    Harmonium[AnalyticTuple[Poissons], Latent]
):
    """Harmonium with subpopulations of Poisson neurons over a latent family, each subpopulation tuned to some nodes of the latent.

    The observable is a tuple of independent Poisson populations, one node per subpopulation, and
    ``tuning`` lists the pairs (subpopulation, latent node) that are coupled. Coupling one
    subpopulation to every node of the latent gives mixed selectivity; coupling subpopulation $k$
    to node $k$ alone gives a code specific to each node.

    Mathematically, the log-rate of neuron $i$ in subpopulation $k$ is $\\theta_{X,i} + \\sum_j
    \\Theta_{X_k Z_j, i} \\cdot \\mathbf s_{Z_j}(z_j)$, summed over the nodes $j$ that $k$ is tuned
    to, so the shape of each tuning curve is set by the sufficient statistics of those nodes: a
    von Mises node gives a cosine bump in an angle, a Normal node a Gaussian bump.
    """

    # Fields

    pop_sizes: tuple[int, ...]
    """Number of neurons in each subpopulation."""

    latent: Latent
    """The latent family; every node that a subpopulation is tuned to must be a clique of its own."""

    tuning: tuple[tuple[int, int], ...]
    """The coupled pairs (subpopulation, latent node)."""

    # Overrides

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """One crossing per tuned pair, over the whole coordinate block on each side."""
        return tuple(
            CrossTerm(
                (pop,),
                (node,),
                CliqueMap(
                    Rectangular(),
                    IdentityEmbedding(self.obs_man.clq_man((pop,))),
                    IdentityEmbedding(self.pst_man.clq_man((node,))),
                ),
            )
            for pop, node in self.tuning
        )

    @property
    @override
    def obs_man(self) -> AnalyticTuple[Poissons]:
        return AnalyticTuple(tuple(Poissons(n) for n in self.pop_sizes))

    @property
    @override
    def pst_man(self) -> Latent:
        return self.latent


# --- Boltzmann neurons (Normal latent) ---


@dataclass(frozen=True)
class BoltzmannNormalHarmonium[Shape: Differentiable](
    Harmonium[Boltzmann[Shape], FullNormal]
):
    """Harmonium with a Boltzmann observable and a full-covariance Normal latent.

    The opposite orientation to :class:`~goal.models.harmonium.lgm.NormalBoltzmannMachine`
    (Normal observable, Boltzmann latent): here a population of correlated
    binary neurons encodes a continuous Gaussian. The observable is a field because a
    :class:`~goal.models.base.gaussian.boltzmann.ChordalBoltzmann` carries its
    junction tree.

    The interaction uses ``IdentityEmbedding`` on both sides, so it carries the
    latent's full sufficient statistic $\\mathbf s_Z(z) = (z, z z^\\top)$ into the
    Boltzmann natural parameters --- a genuine second-order coupling: the Gaussian
    second moments drive the Boltzmann couplings, and the posterior over $z$
    acquires an observation-dependent precision.
    """

    # Fields

    boltzmann: Boltzmann[Shape]
    """The Boltzmann observable (chordal, diagonal, ...)."""

    lat_dim: int
    """Dimension of the Gaussian latent."""

    # Overrides

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """The observable and the latent, coupled."""
        int_map = CliqueMap(
            Rectangular(),
            IdentityEmbedding(self.obs_man),
            IdentityEmbedding(self.pst_man),
        )
        return (CrossTerm((0,), (0,), int_map),)

    @property
    @override
    def obs_man(self) -> Boltzmann[Shape]:
        return self.boltzmann

    @property
    @override
    def pst_man(self) -> FullNormal:
        return full_normal(self.lat_dim)


@dataclass(frozen=True)
class PopulationCodeHarmonium(Harmonium[Boltzmann[Any], DiagonalNormal]):
    """Harmonium with Boltzmann neurons as its observable and a diagonal Normal latent, coupled through the activities of the neurons only.

    Neuron $i$ has the logit $\\theta_{N,i} + \\Theta_{NZ,i} \\cdot \\mathbf s_Z(z)$, so its tuning
    curve is the exponential of a quadratic in $z$, and the couplings of the neurons do not depend
    on $z$. Given $n$, $z$ is Gaussian with natural parameters affine in $n$. Unlike
    :class:`BoltzmannNormalHarmonium`, the pairwise statistics of the neurons are not
    coupled to $z$.
    """

    # Fields

    neurons: Boltzmann[Any]
    lat_dim: int

    # Overrides

    @property
    @override
    def crs_trms(self) -> tuple[CrossTerm, ...]:
        """The activities of the neurons and the statistics of the latent, coupled."""
        int_map = CliqueMap(
            Rectangular(),
            GeneralizedGaussianLocationEmbedding(self.obs_man),
            IdentityEmbedding(self.pst_man),
        )
        return (CrossTerm((0,), (0,), int_map),)

    @property
    @override
    def obs_man(self) -> Boltzmann[Any]:
        return self.neurons

    @property
    @override
    def pst_man(self) -> DiagonalNormal:
        return diagonal_normal(self.lat_dim)


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
