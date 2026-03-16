Exponential Family Subpackage
=============================

Abstract exponential family structure: base classes with increasing capabilities, combinators for location-shape families, harmonium (latent variable) models, and hierarchical graphical models.

An exponential family is a collection of distributions with densities of the form

.. math::

   p(x; \theta) = \mu(x)\exp(\theta \cdot \mathbf{s}(x) - \psi(\theta))

where $\theta$ are the natural parameters, $\mathbf{s}(x)$ the sufficient statistic, $\mu(x)$ the base measure, and $\psi(\theta)$ the log-partition function. These families carry a dually flat Riemannian structure: the natural parameters $\theta$ and mean parameters $\eta = \nabla\psi(\theta)$ form dual coordinate systems connected by Legendre duality. This structure gives operations like maximum likelihood estimation, KL divergence, and Bayesian updating clean closed-form or gradient-based implementations.

The capability hierarchy controls what operations are available for a given model:

- **ExponentialFamily** --- sufficient statistics and base measure
- **Generative** --- additionally supports sampling
- **Differentiable** --- analytic log-partition function, enabling gradient-based optimization
- **Analytic** --- analytic negative entropy, enabling closed-form algorithms (e.g. EM)

Harmoniums are the key composed structure: latent-variable models where conjugate structure ensures the posterior stays in the same exponential family as the prior.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   base
   combinators
   protocols
   variational
   harmonium
   graphical
