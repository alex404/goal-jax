"""Boltzmann Hierarchical Mixture of Gaussians (Boltzmann HMoG) models.

This module provides implementations of hierarchical models that combine
Boltzmann Linear Gaussian Models with mixture clustering over the Boltzmann
latent space.

**Model structure**: Boltzmann HMoG models have two levels:

- **Lower harmonium**: Maps observations :math:`X \\in \\mathbb{R}^p` to discrete
  binary latent factors :math:`Y \\in \\{0,1\\}^d` using a linear Gaussian-Boltzmann
  relationship (BoltzmannLGM)
- **Upper harmonium**: Models a mixture over the Boltzmann latent space :math:`Y`

The joint distribution factors as:

.. math::

    p(X, Y, Z) = p(Z) \\cdot p(Y | Z) \\cdot p(X | Y)

where :math:`Z \\in \\{1,\\ldots,K\\}` are discrete cluster assignments.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from ...geometry import (
    DifferentiableConjugated,
    DifferentiableHierarchical,
    PositiveDefinite,
    SymmetricHierarchical,
)
from ..base.gaussian.boltzmann import Boltzmann, DiagonalBoltzmann
from ..base.gaussian.normal import Normal
from ..harmonium.lgm import BoltzmannLGM, DifferentiableBoltzmannLGM
from ..harmonium.mixture import CompleteMixture, Mixture


@dataclass(frozen=True)
class SymmetricBoltzmannHMoG(
    SymmetricHierarchical[BoltzmannLGM, Mixture[Boltzmann]],
    DifferentiableConjugated[Normal, Mixture[Boltzmann], Mixture[Boltzmann]],
):
    """Symmetric Hierarchical Mixture of Boltzmann machines.

    This model combines:

    1. A linear Gaussian model with Boltzmann latent variables (BoltzmannLGM)
       mapping continuous observations to discrete binary latent factors
    2. A mixture model over the Boltzmann latent space with categorical
       component assignments

    Model structure: Normal (observable) -> Boltzmann (latent) -> Categorical (mixture)

    The joint distribution factors as:

    .. math::

        p(X, Y, Z) = p(Z) \\cdot p(Y | Z) \\cdot p(X | Y)

    where:
        - X are continuous observations (Normal)
        - Y are discrete binary latent factors (Boltzmann)
        - Z are categorical mixture assignments

    The symmetric structure means posterior and prior use the same
    parameterization, enabling bidirectional parameter transformations
    via ``join_conjugated``.
    """

    def posterior_categorical(self, params: Array, x: Array) -> Array:
        """Compute posterior categorical distribution p(Z|x) in natural coordinates.

        Returns the natural parameters of the categorical distribution over
        mixture components given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components-1,) with categorical natural parameters
        """
        # Compute posterior over (Y, Z) given x
        posterior = self.posterior_at(params, x)

        # Extract categorical marginal by computing prior of upper harmonium
        return self.upr_hrm.prior(posterior)

    def posterior_soft_assignments(self, params: Array, x: Array) -> Array:
        """Compute posterior assignment probabilities p(Z|x).

        Returns the posterior probability distribution over mixture components
        given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components,) giving p(z_k|x) for each component k
        """
        cat_natural = self.posterior_categorical(params, x)
        cat_mean = self.upr_hrm.lat_man.to_mean(cat_natural)
        return self.upr_hrm.lat_man.to_probs(cat_mean)

    def posterior_hard_assignment(self, params: Array, x: Array) -> Array:
        """Compute hard assignment to most probable mixture component.

        Returns the index of the mixture component with the highest posterior
        probability given the observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Integer index of most probable component
        """
        soft_assignments = self.posterior_soft_assignments(params, x)
        return jnp.argmax(soft_assignments)

    def posterior_boltzmann(self, params: Array, x: Array) -> Array:
        """Compute posterior Boltzmann distribution p(Y|x) in natural coordinates.

        This returns the observable component of the posterior mixture,
        representing the marginal distribution over Boltzmann states.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Natural parameters for the Boltzmann observable component
        """
        # Get full posterior (Y, Z) parameters
        posterior = self.posterior_at(params, x)

        # Extract the observable (Y) component from the mixture
        y_params, _, _ = self.upr_hrm.split_coords(posterior)
        return y_params


def symmetric_boltzmann_hmog(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    boltz_dim: int,
    n_components: int,
) -> SymmetricBoltzmannHMoG:
    """Create a symmetric Boltzmann hierarchical mixture model.

    Constructs a three-level hierarchical model:
        Normal observations -> Boltzmann binary latents -> Categorical mixture

    Args:
        obs_dim: Dimension of observable Normal distribution
        obs_rep: Covariance representation for observables
            (e.g., PositiveDefinite(), Diagonal())
        boltz_dim: Number of binary Boltzmann units (latent dimension)
        n_components: Number of mixture components

    Returns:
        SymmetricBoltzmannHMoG model ready for parameter initialization

    Example:
        >>> model = symmetric_boltzmann_hmog(
        ...     obs_dim=10,
        ...     obs_rep=PositiveDefinite(),
        ...     boltz_dim=5,
        ...     n_components=3
        ... )
        >>> params = model.initialize(jax.random.PRNGKey(0))
    """
    # Build lower harmonium: Normal -> Boltzmann
    lwr_hrm = BoltzmannLGM(
        obs_dim=obs_dim,
        obs_rep=obs_rep,
        lat_dim=boltz_dim,
    )

    # Build upper harmonium: Mixture[Boltzmann]
    boltz_man = Boltzmann(boltz_dim)
    upr_hrm = CompleteMixture(boltz_man, n_components)

    return SymmetricBoltzmannHMoG(
        lwr_hrm=lwr_hrm,
        upr_hrm=upr_hrm,
    )


@dataclass(frozen=True)
class DifferentiableBoltzmannHMoG(
    DifferentiableHierarchical[
        DifferentiableBoltzmannLGM,
        Mixture[DiagonalBoltzmann],
        Mixture[Boltzmann],
    ],
):
    """Differentiable Hierarchical Mixture of Boltzmann machines.

    This model combines:

    1. A differentiable linear Gaussian model with asymmetric Boltzmann latent
       variables (DifferentiableBoltzmannLGM) mapping continuous observations
       to discrete binary latent factors
    2. Mixture models over the Boltzmann latent space with asymmetric structure

    Model structure: Normal (observable) -> DiagonalBoltzmann/Boltzmann (latent) -> Categorical

    **Asymmetric structure for efficient inference**:

    - Posterior uses DiagonalBoltzmann (mean-field, O(n) log partition)
    - Prior uses full Boltzmann (with coupling from conjugation)

    The joint distribution factors as:

    .. math::

        p(X, Y, Z) = p(Z) \\cdot p(Y | Z) \\cdot p(X | Y)

    where:
        - X are continuous observations (Normal)
        - Y are discrete binary latent factors (DiagonalBoltzmann posterior / Boltzmann prior)
        - Z are categorical mixture assignments
    """

    def posterior_categorical(self, params: Array, x: Array) -> Array:
        """Compute posterior categorical distribution p(Z|x) in natural coordinates.

        Returns the natural parameters of the categorical distribution over
        mixture components given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components-1,) with categorical natural parameters
        """
        # Compute posterior over (Y, Z) given x
        posterior = self.posterior_at(params, x)

        # Extract categorical marginal by computing prior of posterior upper harmonium
        return self.pst_upr_hrm.prior(posterior)

    def posterior_soft_assignments(self, params: Array, x: Array) -> Array:
        """Compute posterior assignment probabilities p(Z|x).

        Returns the posterior probability distribution over mixture components
        given an observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Array of shape (n_components,) giving p(z_k|x) for each component k
        """
        cat_natural = self.posterior_categorical(params, x)
        cat_mean = self.pst_upr_hrm.lat_man.to_mean(cat_natural)
        return self.pst_upr_hrm.lat_man.to_probs(cat_mean)

    def posterior_hard_assignment(self, params: Array, x: Array) -> Array:
        """Compute hard assignment to most probable mixture component.

        Returns the index of the mixture component with the highest posterior
        probability given the observation.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Integer index of most probable component
        """
        soft_assignments = self.posterior_soft_assignments(params, x)
        return jnp.argmax(soft_assignments)

    def posterior_diagonal_boltzmann(self, params: Array, x: Array) -> Array:
        """Compute posterior DiagonalBoltzmann distribution p(Y|x) in natural coordinates.

        This returns the observable component of the posterior mixture,
        representing the marginal distribution over DiagonalBoltzmann states.

        Args:
            params: Model parameters (natural coordinates)
            x: Observable data point

        Returns:
            Natural parameters for the DiagonalBoltzmann observable component
        """
        # Get full posterior (Y, Z) parameters
        posterior = self.posterior_at(params, x)

        # Extract the observable (Y) component from the mixture
        y_params, _, _ = self.pst_upr_hrm.split_coords(posterior)
        return y_params


def differentiable_boltzmann_hmog(
    obs_dim: int,
    obs_rep: PositiveDefinite,
    boltz_dim: int,
    n_components: int,
) -> DifferentiableBoltzmannHMoG:
    """Create a differentiable Boltzmann hierarchical mixture model.

    Constructs a three-level hierarchical model with asymmetric structure:
        Normal observations -> DiagonalBoltzmann/Boltzmann binary latents -> Categorical mixture

    The asymmetric structure enables efficient inference:
        - Posterior uses DiagonalBoltzmann (mean-field, O(n) computation)
        - Prior uses full Boltzmann (with coupling from conjugation)

    Args:
        obs_dim: Dimension of observable Normal distribution
        obs_rep: Covariance representation for observables
            (e.g., PositiveDefinite(), Diagonal())
        boltz_dim: Number of binary Boltzmann units (latent dimension)
        n_components: Number of mixture components

    Returns:
        DifferentiableBoltzmannHMoG model ready for parameter initialization

    Example:
        >>> model = differentiable_boltzmann_hmog(
        ...     obs_dim=10,
        ...     obs_rep=PositiveDefinite(),
        ...     boltz_dim=5,
        ...     n_components=3
        ... )
        >>> params = model.initialize(jax.random.PRNGKey(0))
    """
    # Build lower harmonium: Normal -> DiagonalBoltzmann/Boltzmann
    lwr_hrm = DifferentiableBoltzmannLGM(
        obs_dim=obs_dim,
        obs_rep=obs_rep,
        lat_dim=boltz_dim,
    )

    # Build posterior upper harmonium: Mixture[DiagonalBoltzmann]
    diag_boltz_man = DiagonalBoltzmann(boltz_dim)
    pst_upr_hrm = CompleteMixture(diag_boltz_man, n_components)

    # Build prior upper harmonium: Mixture[Boltzmann]
    boltz_man = Boltzmann(boltz_dim)
    prr_upr_hrm = CompleteMixture(boltz_man, n_components)

    return DifferentiableBoltzmannHMoG(
        lwr_hrm=lwr_hrm,
        pst_upr_hrm=pst_upr_hrm,
        prr_upr_hrm=prr_upr_hrm,
    )
