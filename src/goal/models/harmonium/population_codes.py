"""Population codes, Poisson-VonMises harmoniums, and Poisson mixture models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

import jax.numpy as jnp
from jax import Array

from ...geometry import (
    EmbeddedMap,
    IdentityEmbedding,
    Rectangular,
)
from ...geometry.exponential_family.harmonium import Harmonium
from ...geometry.exponential_family.variational import VariationalConjugated
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
        return EmbeddedMap(Rectangular(), IdentityEmbedding(lat), IdentityEmbedding(obs))



# --- Von Mises Population Code (unified variational model) ---


@dataclass(frozen=True)
class VonMisesPopulationCode(
    VariationalConjugated[Poissons, VonMisesProduct, VonMisesProduct]
):
    """Variational population code with Poisson observables and VonMises latents.

    Unifies 1D population codes and multi-dimensional toroidal models under
    the variational conjugation framework. The approximate posterior has VonMises
    product form, with learned conjugation correction $\\rho$.
    """

    hrm: PoissonVonMisesHarmonium

    # Overrides

    @property
    @override
    def rho_emb(self) -> IdentityEmbedding[VonMisesProduct]:
        return IdentityEmbedding(self.hrm.pst_man)

    # Methods

    @property
    def n_neurons(self) -> int:
        """Number of Poisson observable neurons."""
        return self.hrm.n_neurons

    @property
    def n_latent(self) -> int:
        """Number of VonMises latent dimensions."""
        return self.hrm.n_latent

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

        Builds harmonium parameters from gains, preferred directions, and baselines,
        then fits conjugation parameters $\\rho$ via least-squares regression.
        """
        vm = self.pst_man.rep_man  # underlying VonMises

        # Build harmonium params from tuning curves
        obs_params = baselines
        int_col_1 = gains * jnp.cos(preferred)
        int_col_2 = gains * jnp.sin(preferred)
        int_params = jnp.stack([int_col_1, int_col_2], axis=1).ravel()
        prior_nat = vm.join_mean_concentration(prior_mean, prior_concentration)
        hrm_params = self.hrm.join_coords(obs_params, int_params, prior_nat)

        # Fit rho via regression
        zero_rho = jnp.zeros(self.rho_man.dim)
        init_params = self.join_coords(zero_rho, hrm_params)
        rho, _, _, _ = self.regress_conjugation_parameters(
            key, init_params, n_regression_samples
        )

        return self.join_coords(rho, hrm_params)


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
