Harmonium Models Subpackage
===========================

Conjugate exponential family models with harmonium (coupled latent variable) structure.

This subpackage provides implementations of harmonium-based statistical models, which represent
joint distributions over observable and latent variables with conjugate structure enabling
efficient inference and learning.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   lgm
   mixture
   poisson_mixture

Core Components
---------------

**Linear Gaussian Models (lgm.py)**

  - Linear relationships between observations and latent variables
  - Gaussian observable and latent distributions
  - Flexible covariance structures (Diagonal, Isotropic, Full)
  - Both analytic and differentiable implementations
  - Specializations: Factor Analysis, Principal Component Analysis

**Mixture Models (mixture.py)**

  - Probabilistic mixtures of any exponential family
  - Categorical latent variables for component assignment
  - EM algorithm support
  - Analytic and differentiable variants

**Poisson Mixtures (poisson_mixture.py)**

  - Mixtures specialized for count data
  - Standard Poisson for homogeneous variance
  - Conway-Maxwell-Poisson for handling dispersion
  - Type aliases and factory functions for common configurations

Harmonium Structure
-------------------

Harmoniums extend basic exponential families with latent variable structure:

- **Observable manifold**: The distribution over observed data
- **Latent manifold**: The distribution over hidden variables
- **Interaction matrix**: Captures dependencies between observables and latents
- **Posterior vs Prior**: Distinction between inference-time and model-time latent distributions

Key Properties
--------------

- **Conjugacy**: Prior can be derived from likelihood parameters, enabling analytical computations
- **Symmetric variants**: When posterior and prior share the same parameterization
- **Analytic tractability**: Closed-form log-partition and entropy for certain models
- **Hierarchical composability**: Can be nested to create hierarchical models
