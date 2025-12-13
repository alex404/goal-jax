"""Hierarchical Mixture of Gaussians (HMoG) models.

This module provides concrete implementations of hierarchical Gaussian models that combine
linear Gaussian dimensionality reduction with Gaussian mixture clustering, enabling joint
learning of latent factor representations and cluster assignments.

**Model structure**: HMoG models have two levels:

- **Lower harmonium**: Maps observations :math:`X \\in \\mathbb{R}^p` to first-level latent factors
  :math:`Y \\in \\mathbb{R}^d` using a linear Gaussian relationship (factor analysis/PCA)
- **Upper harmonium**: Models a mixture of Gaussians over the latent space :math:`Y`

The joint distribution factors as:

.. math::

    p(X, Y, Z) = p(Z) \\cdot p(Y | Z) \\cdot p(X | Y)

where :math:`Z \\in \\{1,\\ldots,K\\}` are discrete cluster assignments.

**Variants**: Three implementations with different analytical properties:

- **DifferentiableHMoG**: Gradient-based optimization, uses restricted posterior covariance
  for efficiency (e.g., diagonal)
- **SymmetricHMoG**: Symmetric posterior/prior structure, additional functionality like
  `join_conjugated`, but slower due to full covariance matrix operations
- **AnalyticHMoG**: Fully analytic, enables closed-form EM and bidirectional parameter conversion

Factory functions (`differentiable_hmog`, `symmetric_hmog`, `analytic_hmog`) provide
convenient construction for common configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from jax import Array

from ...geometry import (
    AnalyticConjugated,
    DifferentiableConjugated,
    EmbeddedMap,
    LatentHarmoniumEmbedding,
    LinearEmbedding,
    ObservableEmbedding,
    PositiveDefinite,
    SymmetricConjugated,
    hierarchical_conjugation_parameters,
    hierarchical_to_natural_likelihood,
)
from ..base.gaussian.normal import Normal
from ..harmonium.lgm import (
    NormalAnalyticLGM,
    NormalCovarianceEmbedding,
    NormalLGM,
)
from ..harmonium.mixture import AnalyticMixture, Mixture

### HMoG Classes ###


@dataclass(frozen=True)
class DifferentiableHMoG(
    DifferentiableConjugated[
        Normal,
        AnalyticMixture[Normal],
        Mixture[Normal],
    ]
):
    """Differentiable Hierarchical Mixture of Gaussians.

    This model combines:
    1. A linear Gaussian model (factor analysis) mapping observations to latents
    2. A Gaussian mixture model over the latent space

    Supports gradient-based optimization via log-likelihood descent.
    Uses full covariance Gaussians in the latent space.

    **Posterior vs Prior Structure**: The posterior latent mixture (`pst_upr_hrm`) uses an AnalyticMixture with a restricted covariance structure (`Normal`) for computational efficiency during frequent posterior computation. The prior latent mixture (`prr_upr_hrm`) is a Mixture that embeds the restricted latent covariance structure (`Normal`) into a full covariance structure (`Normal`). This embedding is necessary because the prior's shape is dictated by the conjugation parameters—it must accommodate the complete parameterization needed for conjugation parameter computation and gradient-based optimization.

    The model has three component harmoniums:
    - lwr_hrm: Maps observable Normal to latent Normal
    - pst_upr_hrm: Analytic mixture for posterior (restricted covariance)
    - prr_upr_hrm: Differentiable mixture embedding restricted latent structure into full structure for conjugation
    """

    # Fields
    lwr_hrm: NormalLGM
    """Lower harmonium: linear Gaussian model from observations to first latents."""

    pst_upr_hrm: AnalyticMixture[Normal]
    """Posterior upper harmonium: analytic mixture with restricted covariance (Normal) for inference efficiency."""

    prr_upr_hrm: Mixture[Normal]
    """Prior upper harmonium: differentiable mixture that embeds the restricted latent covariance (Normal) into full covariance (Normal) to accommodate conjugation parameters."""

    def __post_init__(self):
        """Validate that component harmoniums are compatible."""
        assert self.lwr_hrm.pst_man == self.pst_upr_hrm.obs_man
        assert self.lwr_hrm.prr_man == self.prr_upr_hrm.obs_man

    # Overrides from DifferentiableConjugated

    @property
    @override
    def int_man(self) -> EmbeddedMap[AnalyticMixture[Normal], Normal]:
        return self.lwr_hrm.int_man.prepend_embedding(  # pyright: ignore[reportReturnType]
            ObservableEmbedding(self.pst_upr_hrm)
        )

    @property
    @override
    def pst_prr_emb(
        self,
    ) -> LinearEmbedding[
        AnalyticMixture[Normal],
        Mixture[Normal],
    ]:
        # For DifferentiableHMoG, we need LatentHarmoniumEmbedding

        return LatentHarmoniumEmbedding(
            self.lwr_hrm.pst_prr_emb,
            self.pst_upr_hrm,
            self.prr_upr_hrm,
        )

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract y (first latent) from yz sample for likelihood evaluation."""
        return prr_sample[:, : self.lwr_hrm.prr_man.data_dim]

    @override
    def conjugation_parameters(
        self,
        lkl_params: Array,
    ) -> Array:
        """Compute conjugation parameters for the hierarchical structure.

        Parameters
        ----------
        lkl_params : Array
            Natural parameters for likelihood function.

        Returns
        -------
        Array
            Natural parameters for conjugation in Mixture[Normal] space.
        """
        return hierarchical_conjugation_parameters(
            self.lwr_hrm, self.prr_upr_hrm, lkl_params
        )


@dataclass(frozen=True)
class SymmetricHMoG(
    SymmetricConjugated[
        Normal,
        Mixture[Normal],
    ]
):
    """Symmetric Hierarchical Mixture of Gaussians.

    This model supports gradient-based optimization with additional functionality
    (e.g., join_conjugated) not available in DifferentiableHMoG.

    The symmetric structure means the posterior and conjugated latent spaces are
    the same, enabling bidirectional parameter transformations.

    Trade-off: Matrix inversions happen in the space of full covariance matrices
    over the latent space, which can be slower than DifferentiableHMoG.

    The model has two component harmoniums:
    - lwr_hrm: Maps observable Normal to latent Normal (analytic version)
    - upr_hrm: Differentiable mixture model with constrained latent space Normal
    """

    # Fields
    lwr_hrm: NormalAnalyticLGM
    """Lower harmonium: analytic linear Gaussian model."""

    upr_hrm: Mixture[Normal]
    """Upper harmonium: mixture model with constrained latent representation."""

    def __post_init__(self):
        """Validate that component harmoniums are compatible."""

        assert self.lwr_hrm.pst_man == self.upr_hrm.obs_man

    # Overrides from SymmetricConjugated

    @property
    @override
    def int_man(self) -> EmbeddedMap[Mixture[Normal], Normal]:
        return self.lwr_hrm.int_man.prepend_embedding(ObservableEmbedding(self.upr_hrm))  # pyright: ignore[reportReturnType]

    @property
    @override
    def lat_man(self) -> Mixture[Normal]:
        return self.upr_hrm

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract y (first latent) from yz sample for likelihood evaluation."""
        return prr_sample[:, : self.lwr_hrm.prr_man.data_dim]

    @override
    def conjugation_parameters(
        self,
        lkl_params: Array,
    ) -> Array:
        """Compute conjugation parameters for the hierarchical structure.

        Parameters
        ----------
        lkl_params : Array
            Natural parameters for likelihood function.

        Returns
        -------
        Array
            Natural parameters for conjugation in Mixture[Normal] space.
        """
        return hierarchical_conjugation_parameters(
            self.lwr_hrm, self.upr_hrm, lkl_params
        )


@dataclass(frozen=True)
class AnalyticHMoG(
    AnalyticConjugated[
        Normal,
        AnalyticMixture[Normal],
    ],
):
    """Analytic Hierarchical Mixture of Gaussians.

    This model enables:
    - Closed-form EM algorithm for learning (from AnalyticConjugated)
    - Bidirectional parameter conversion (mean ↔ natural)
    - Full analytical tractability
    - Shared implementation with SymmetricHMoG

    Requires full covariance Gaussians in the latent space.

    The model uses:
    - lwr_hrm: Analytic linear Gaussian model
    - upr_hrm: Analytic mixture model (all operations have closed forms)
    """

    # Fields
    lwr_hrm: NormalAnalyticLGM
    """Lower harmonium: analytic linear Gaussian model."""

    upr_hrm: AnalyticMixture[Normal]
    """Upper harmonium: mixture model with constrained latent representation."""

    def __post_init__(self):
        """Validate that component harmoniums are compatible."""

        assert self.lwr_hrm.pst_man == self.upr_hrm.obs_man

    # Overrides from SymmetricConjugated

    @property
    @override
    def lat_man(self) -> AnalyticMixture[Normal]:
        return self.upr_hrm

    @override
    def extract_likelihood_input(self, prr_sample: Array) -> Array:
        """Extract y (first latent) from yz sample for likelihood evaluation."""
        return prr_sample[:, : self.lwr_hrm.prr_man.data_dim]

    @override
    def conjugation_parameters(
        self,
        lkl_params: Array,
    ) -> Array:
        """Compute conjugation parameters for the hierarchical structure.

        Parameters
        ----------
        lkl_params : Array
            Natural parameters for likelihood function.

        Returns
        -------
        Array
            Natural parameters for conjugation in Mixture[Normal] space.
        """
        return hierarchical_conjugation_parameters(
            self.lwr_hrm, self.upr_hrm, lkl_params
        )

    @property
    @override
    def int_man(self) -> EmbeddedMap[AnalyticMixture[Normal], Normal]:
        return self.lwr_hrm.int_man.prepend_embedding(ObservableEmbedding(self.upr_hrm))  # pyright: ignore[reportReturnType]

    @override
    def to_natural_likelihood(
        self,
        means: Array,
    ) -> Array:
        """Convert mean parameters to natural likelihood parameters.

        Parameters
        ----------
        means : Array
            Mean parameters for the hierarchical model.

        Returns
        -------
        Array
            Natural parameters for likelihood function.
        """
        return hierarchical_to_natural_likelihood(
            self, self.lwr_hrm, self.upr_hrm, means
        )


## Factory Functions ##


def differentiable_hmog(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    lat_dim: int,
    pst_lat_rep: PositiveDefinite,
    n_components: int,
) -> DifferentiableHMoG:
    """Create a differentiable hierarchical mixture of Gaussians model.

    This function constructs a hierarchical model combining:
    1. A bottom layer with a linear Gaussian model reducing observables to first-level latents
    2. A top layer with a Gaussian mixture model for modelling the latent distribution

    This model supports optimization via log-likelihood gradient descent.
    Uses full covariance Gaussians in the latent space.
    """

    pst_y_man = Normal(lat_dim, pst_lat_rep)
    prr_y_man = Normal(lat_dim, PositiveDefinite())
    lwr_hrm = NormalLGM(obs_dim, obs_rep, lat_dim, pst_lat_rep)
    mix_sub = NormalCovarianceEmbedding(pst_y_man, prr_y_man)
    pst_upr_hrm = AnalyticMixture(pst_y_man, n_components)

    # Use the embedding directly for the mixture
    prr_upr_hrm = Mixture(n_components, mix_sub)

    return DifferentiableHMoG(
        lwr_hrm,
        pst_upr_hrm,
        prr_upr_hrm,
    )


def symmetric_hmog(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    lat_dim: int,
    lat_rep: PositiveDefinite,
    n_components: int,
) -> SymmetricHMoG:
    """Create a symmetric hierarchical mixture of Gaussians model.

    Supports optimization via log-likelihood gradient descent with additional functionality
    (e.g., `join_conjugated`) not available in `DifferentiableHMoG`. The symmetric structure
    means posterior and prior use the same latent parameterization.

    Trade-off: Matrix inversions happen in the space of full covariance matrices over the
    latent space, which can be slower than `DifferentiableHMoG`.
    """

    mid_lat_man = Normal(lat_dim, PositiveDefinite())
    sub_lat_man = Normal(lat_dim, lat_rep)
    mix_sub = NormalCovarianceEmbedding(sub_lat_man, mid_lat_man)
    lwr_hrm = NormalAnalyticLGM(obs_dim, obs_rep, lat_dim)

    # Use the embedding directly for the mixture
    upr_hrm = Mixture(n_components, mix_sub)

    return SymmetricHMoG(
        lwr_hrm,
        upr_hrm,
    )


def analytic_hmog(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    lat_dim: int,
    n_components: int,
) -> AnalyticHMoG:
    """Create an analytic hierarchical mixture of Gaussians model.

    Enables closed-form expectation-maximization for learning and bidirectional parameter
    conversion between natural and mean coordinates. Requires full covariance Gaussians in
    the latent space for complete analytical tractability.
    """

    lat_man = Normal(lat_dim, PositiveDefinite())
    lwr_hrm = NormalAnalyticLGM(obs_dim, obs_rep, lat_dim)
    upr_hrm = AnalyticMixture(lat_man, n_components)

    return AnalyticHMoG(
        lwr_hrm,
        upr_hrm,
    )
