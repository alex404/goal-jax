Exponential Family Subpackage
=============================

Abstract exponential family structure: base classes with increasing capabilities, combinators for location-shape families, harmonium (latent variable) models, and hierarchical graphical models.

Exponential families are parameterized families of probability distributions with special computational properties: they have natural and mean parameter coordinate systems connected by Legendre duality, and operations like maximum likelihood estimation, KL divergence, and Bayesian updating have clean closed-form or gradient-based implementations.

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
