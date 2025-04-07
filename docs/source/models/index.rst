Models Package
==============

Statistical models built on Goal's geometric foundations.

The models package provides a comprehensive hierarchy of statistical models, from basic distributions to complex graphical models, all implemented within the exponential family framework.

.. toctree::
   :maxdepth: 1
   :caption: Subpackages:

   base/index
   graphical/index

Package Structure
-----------------

The package is organized in a progressive complexity structure:

* **Base Distributions**: Core statistical distributions as exponential families

  - Categorical: Discrete distributions over k categories
  - Poisson & Conway-Maxwell-Poisson: Count distributions with varying dispersion
  - von Mises: Circular distributions for angular data
  - Normal: Gaussian family with various covariance structures (Full, Diagonal, Isotropic)

* **Graphical Models**: Complex dependency structures

  - Mixture Models: Probabilistic combinations of base distributions
  - Linear Gaussian Models: Factor Analysis and PCA implementations
  - Hierarchical Models: Multi-level structures with latent variables

Implementation Philosophy
-------------------------

All models follow these design principles:

1. **Mathematical Rigor**: Implementations directly follow the mathematical theory of exponential families
2. **Type Safety**: Strong typing ensures parameter validity and proper model composition
3. **Efficient Algorithms**: Optimized implementations of inference and learning algorithms
4. **Composability**: Complex models are built from simpler components in a consistent manner
