"""Population codes: Poisson populations over any latent family, Boltzmann-Normal harmoniums, and Poisson mixture models."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from jax import Array

from ...geometry import (
    AttachedHarmonium,
    CliqueMap,
    CrossTerm,
    IdentityEmbedding,
    Rectangular,
)
from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.combinators import AnalyticTuple
from ...geometry.exponential_family.harmonium import Harmonium
from ...geometry.exponential_family.variational import (
    DifferentiableVariationalConjugated,
)
from ..base.categorical import Bernoullis
from ..base.gaussian.boltzmann import (
    Boltzmann,
    ChordalBoltzmann,
    ChordalCouplingMatrix,
    DiagonalBoltzmann,
)
from ..base.gaussian.normal import FullNormal, full_normal
from ..base.poisson import CoMPoissons, Poissons, PopulationLocationEmbedding
from .mixture import AnalyticMixture, Mixture

# --- Poisson Population Code (any latent family) ---


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


# --- Boltzmann Population Code (Gaussian latent) ---


@dataclass(frozen=True)
class BoltzmannNormalHarmonium[Shape: Differentiable](
    Harmonium[Boltzmann[Shape], FullNormal]
):
    """Harmonium with a Boltzmann observable and a full-covariance Normal latent.

    The opposite orientation to :class:`~goal.models.harmonium.lgm.BoltzmannLGM`
    (Gaussian observable, Boltzmann latent): here a population of correlated
    binary neurons encodes a continuous Gaussian. Only ``int_man`` is supplied;
    the base derives everything else. The observable is a field because a
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
