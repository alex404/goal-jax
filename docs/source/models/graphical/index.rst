Graphical Models Subpackage
===========================

Complex statistical models with dependency structures.

This subpackage provides implementations of graphical models that represent
relationships and dependencies between variables, allowing for structured
probabilistic modeling of complex data.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   mixture
   lgm
   instances

Core Models
-----------

Two fundamental model types form the backbone of this subpackage:

* **Mixture Models**

  - Probabilistic combinations of simpler distributions
  - Categorical mixing of component distributions
  - EM algorithms for parameter estimation
  - Support for any exponential family observations

* **Linear Gaussian Models**

  - Linear relationships with Gaussian noise
  - Principal Component Analysis (PCA)
  - Factor Analysis (FA)
  - Flexible covariance structures

Pre-built Instances
-------------------

The `instances.py` module provides ready-to-use implementations and constructors for:

* **Hierarchical Models**

  - Multi-level models with latent variables

* **Specialized Mixtures**

  - Pre-configured mixtures of common distributions
  - Poisson mixtures for count data
  - COM-Poisson mixtures for dispersed counts

