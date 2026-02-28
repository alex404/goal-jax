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

**Mixture Infrastructure (mixture.py)**

  - Complete and analytic mixture models for conjugated exponential families
  - Mixture of Factor Analyzers combining mixture modeling with factor analysis
  - Helper functions for computing conjugation parameters in hierarchical mixtures

