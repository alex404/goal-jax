Exponential Family Subpackage
=============================

Statistical manifolds with elegant parametrizations and rich geometric structure.

Implementation Strategy
-----------------------

This package implements exponential families through nested classes with increasing capabilities:

- **ExponentialFamily**: The base class defining sufficient statistics and base measure
- **Generative**: Adds sampling capabilities
- **Differentiable**: Adds analytic log partition function and mapping to mean parameters
- **Analytic**: Adds analytic negative entropy and mapping to natural parameters

More complex exponential families (e.g. harmoniums) recapitulate this class
hierarchy, mixing their own structure with this basic hierarchy.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   base
   combinators
   harmonium
   hierarchical

Mathematical Framework
----------------------

An exponential family is a collection of probability distributions with densities of the form:

.. math::

   p(x; \theta) = \mu(x)\exp(\theta \cdot \mathbf{s}(x) - \psi(\theta))

where:

- $\theta$ are the natural parameters,
- $\mathbf{s}(x)$ is the sufficient statistic,
- $\mu(x)$ is the base measure, and
- $\psi(\theta)$ is the log partition function.

Exponential families have a rich geometric structure expressed through dual parameterizations:

**Natural Parameters** ($\theta$)
  - Form a convex set $\Theta$
  - Define the density directly
  - Connected to maximum likelihood estimation

**Mean Parameters** ($\eta$)
  - Given by expectations: $\eta = \mathbb{E}[\mathbf{s}(x)]$
  - Connected to moment matching and variational inference
  - Form the set $H = \{\eta : \eta = \mathbb{E}_p[\mathbf{s}(X)], p \in \mathcal{P}\}$

Mappings between parameterizations:

.. math::

   \eta = \nabla\psi(\theta) \quad \text{and} \quad \theta = \nabla\phi(\eta)

where $\phi(\eta)$ is the negative entropy, which is the convex conjugate of $\psi(\theta)$.
