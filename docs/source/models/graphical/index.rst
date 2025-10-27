Graphical Models Subpackage
===========================

Hierarchical models that compose multiple harmoniums to create complex dependency structures.

This subpackage provides hierarchical statistical models that combine lower-level harmoniums
(like Linear Gaussian Models) with upper-level harmoniums to create multi-level structures
suitable for clustering, hierarchical representation learning, and structured inference.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   hmog
   mfa
   mixture

Hierarchical Model Types
------------------------

**Hierarchical Mixture of Gaussians (hmog.py)**

  - Multi-level hierarchy combining factor analysis with Gaussian clustering
  - Lower: Linear Gaussian Model maps observations to factors
  - Upper: Gaussian mixture over factor space
  - Analytic, differentiable, and symmetric variants
  - Suitable for joint dimensionality reduction and clustering
  - Factory functions for common configurations

**Mixture of Factor Analyzers (mfa.py)**

  - Combines mixture modeling with factor analysis
  - Shared loading matrix across mixture components
  - Component-specific means and optional dispersion
  - Both analytic and differentiable implementations

Supporting Infrastructure
--------------------------

**Mixture Conjugation Utilities (mixture.py)**

  - Helper functions for computing conjugation parameters in hierarchical mixtures
  - Decomposition of conjugation parameters for mixture of harmoniums structure
  - Supports efficient inference in multi-level mixture models

