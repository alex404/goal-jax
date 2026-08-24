"""Population codes, Poisson-VonMises and Boltzmann-Normal harmoniums, and Poisson mixture models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import jax.numpy as jnp
from jax import Array

from ...geometry import (
    EmbeddedMap,
    IdentityEmbedding,
    Rectangular,
)
from ...geometry.exponential_family.base import Differentiable
from ...geometry.exponential_family.harmonium import Harmonium
from ...geometry.exponential_family.variational import (
    VariationalSymmetric,
    regress_conjugation_parameters,
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
from ..base.von_mises import VonMisesProduct
from .mixture import AnalyticMixture, Mixture

# --- Poisson-VonMises Harmonium ---


@dataclass(frozen=True)
class PoissonVonMisesHarmonium(Harmonium[Poissons, VonMisesProduct]):
    """Harmonium with Poisson observables and VonMises latents."""

    # Fields

    n_neurons: int
    """Number of Poisson observable neurons."""

    n_latent: int
    """Number of VonMises latent dimensions."""

    # Overrides

    @property
    @override
    def int_man(self) -> EmbeddedMap[VonMisesProduct, Poissons]:
        obs = Poissons(self.n_neurons)
        lat = VonMisesProduct(self.n_latent)
        return EmbeddedMap(
            Rectangular(), IdentityEmbedding(lat), IdentityEmbedding(obs)
        )


# --- Von Mises Population Code (unified variational model) ---


@dataclass(frozen=True)
class VonMisesPopulationCode(
    VariationalSymmetric[Poissons, VonMisesProduct, VonMisesProduct]
):
    """Variational population code with Poisson observables and VonMises latents.

    Unifies 1D population codes and multi-dimensional toroidal models under
    the variational conjugation framework. Posterior = prior = conjugation =
    ``VonMisesProduct``, so :meth:`~goal.geometry.exponential_family.variational.VariationalConjugated.conjugation_parameters` returns the stored $\\rho$ unchanged.
    """

    _gen_hrm: PoissonVonMisesHarmonium

    # Overrides

    @property
    @override
    def gen_hrm(self) -> PoissonVonMisesHarmonium:
        return self._gen_hrm

    @property
    @override
    def lat_man(self) -> VonMisesProduct:
        return self._gen_hrm.pst_man

    @property
    @override
    def cnj_man(self) -> VonMisesProduct:
        return self.lat_man

    # Methods

    @property
    def n_neurons(self) -> int:
        """Number of Poisson observable neurons."""
        return self.gen_hrm.n_neurons

    @property
    def n_latent(self) -> int:
        """Number of VonMises latent dimensions."""
        return self.gen_hrm.n_latent

    def initialize_from_tuning_curves(
        self,
        key: Array,
        gains: Array,
        preferred: Array,
        baselines: Array,
        prior_mean: float = 0.0,
        prior_concentration: float = 0.0,
        n_regression_samples: int = 5000,
    ) -> Array:
        """Initialize parameters from tuning curve specification.

        Builds likelihood and prior parameters from gains, preferred directions, and baselines,
        then fits conjugation parameters $\\rho$ via least-squares regression.
        """
        vm = self.pst_man.rep_man  # underlying VonMises

        # Build likelihood params from tuning curves
        obs_params = baselines
        int_col_1 = gains * jnp.cos(preferred)
        int_col_2 = gains * jnp.sin(preferred)
        int_params = jnp.stack([int_col_1, int_col_2], axis=1).ravel()
        lkl_params = self.gen_hrm.lkl_fun_man.join_coords(obs_params, int_params)
        prior_nat = vm.join_mean_concentration(prior_mean, prior_concentration)

        # Fit rho via regression
        zero_rho = jnp.zeros(self.cnj_man.dim)
        init_params = self.join_coords(prior_nat, lkl_params, zero_rho)
        rho, _, _, _ = regress_conjugation_parameters(
            self, key, init_params, n_regression_samples
        )

        return self.join_coords(prior_nat, lkl_params, rho)


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
    def int_man(self) -> EmbeddedMap[FullNormal, Boltzmann[Shape]]:
        lat = full_normal(self.lat_dim)
        return EmbeddedMap(
            Rectangular(), IdentityEmbedding(lat), IdentityEmbedding(self.boltzmann)
        )


@dataclass(frozen=True)
class BoltzmannPopulationCode[Shape: Differentiable](
    VariationalSymmetric[Boltzmann[Shape], FullNormal, FullNormal]
):
    """Variational population code with a Boltzmann observable and a Normal latent.

    Posterior = prior = conjugation = ``FullNormal``, so
    :meth:`~goal.geometry.exponential_family.variational.VariationalConjugated.conjugation_parameters`
    returns the stored $\\rho$ unchanged. Trained via the variational ELBO with the
    conjugation residual regularized; inherits ``mean_elbo``,
    ``conjugation_residual``, ``prior_conjugation_loss``, and joint ``sample``.
    """

    _gen_hrm: BoltzmannNormalHarmonium[Shape]

    # Overrides

    @property
    @override
    def gen_hrm(self) -> BoltzmannNormalHarmonium[Shape]:
        return self._gen_hrm

    @property
    @override
    def lat_man(self) -> FullNormal:
        return self._gen_hrm.pst_man

    @property
    @override
    def cnj_man(self) -> FullNormal:
        return self.lat_man

    # Methods

    @property
    def n_neurons(self) -> int:
        """Number of Boltzmann observable neurons."""
        return self._gen_hrm.boltzmann.data_dim

    @property
    def n_latent(self) -> int:
        """Dimension of the Gaussian latent."""
        return self._gen_hrm.lat_dim


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
