Graphical Models Subpackage
===========================

Hierarchical models that compose multiple harmoniums: mixture of factor analyzers, hierarchical mixture of Gaussians, and supporting infrastructure.

These models extend harmoniums by nesting them: a graphical model uses one harmonium for global structure (e.g. a mixture selecting a component) and another for local structure within each component (e.g. factor analysis). The main use case is the **hierarchical mixture of Gaussians** (HMoG), which combines a Gaussian mixture at the top level with per-component factor analyzers --- effectively a mixture of factor analyzers.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   hmog
   mixture
   variational
