"""Statistical models built on Goal's geometric foundations.

This package provides common statistical models implemented using Goal's geometric framework:

Base Distributions:
   - Univariate:
       - Poisson: Count data with rate parameter
       - Categorical: Discrete distributions over k categories
   - Multivariate:
       - Normal: Gaussian distributions with various covariance structures
           - FullNormal: Unrestricted covariance
           - DiagonalNormal: Independent dimensions
           - IsotropicNormal: Constant variance in all dimensions

Composite Models:
   - Mixture: Probabilistic mixtures of base distributions
   - LinearGaussianModel: Linear transformations with Gaussian noise

The models follow the information geometry formulation of probability theory, where:
   - Each model is a manifold of probability distributions
   - Points on these manifolds represent specific distributions
   - Natural/mean/source coordinates represent different parameterizations
   - Geometric structures (metrics, geodesics, etc.) capture statistical relationships

Example:
   >>> from goal.models import FullNormal
   >>> model = FullNormal(dim=2)  # 2D Gaussian with full covariance
   >>> point = model.random_point()  # Sample a random point on manifold
"""

from .base.categorical import Categorical
from .base.gaussian.normal import (
    Covariance,
    DiagonalCovariance,
    DiagonalNormal,
    Euclidean,
    FullCovariance,
    FullNormal,
    IsotropicCovariance,
    IsotropicNormal,
    Normal,
)
from .base.poisson import CoMPoisson, CoMShape, Poisson
from .base.von_mises import VonMises
from .graphical.instances import (
    AnalyticHMoG,
    CoMPoissonMixture,
    CoMPoissonPopulation,
    DifferentiableHMoG,
    PoissonMixture,
    analytic_hmog,
    com_poisson_mixture,
    differentiable_hmog,
    poisson_mixture,
)
from .graphical.lgm import (
    FactorAnalysis,
    LinearGaussianModel,
    PrincipalComponentAnalysis,
)
from .graphical.mixture import AnalyticMixture, DifferentiableMixture, Mixture

__all__ = [
    "AnalyticHMoG",
    "AnalyticMixture",
    "Categorical",
    "CoMPoisson",
    "CoMPoissonMixture",
    "CoMPoissonPopulation",
    "CoMPoissonPopulation",
    "CoMShape",
    "Covariance",
    "DiagonalCovariance",
    "DiagonalNormal",
    "DifferentiableHMoG",
    "DifferentiableMixture",
    "Euclidean",
    "FactorAnalysis",
    "FullCovariance",
    "FullNormal",
    "IsotropicCovariance",
    "IsotropicNormal",
    "LinearGaussianModel",
    "Mixture",
    "Normal",
    "Poisson",
    "PoissonMixture",
    "PrincipalComponentAnalysis",
    "VonMises",
    "analytic_hmog",
    "com_poisson_mixture",
    "differentiable_hmog",
    "poisson_mixture",
]
