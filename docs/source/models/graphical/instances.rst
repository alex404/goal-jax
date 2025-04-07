Pre-built Model Instances
=========================

.. automodule:: goal.models.graphical.instances
   :noindex:
   :no-members:

Hierarchical Mixtures of Gaussians
----------------------------------

Hierarchical models combining linear dimensionality reducation and clustering.

Type Aliases
~~~~~~~~~~~~

.. py:data:: goal.models.graphical.instances.DifferentiableHMoG

   A hierarchical model combining a differentiable linear Gaussian model with an analytic mixture of Gaussians as the latent distribution.

.. py:data:: goal.models.graphical.instances.SymmetricHMoG

   A symmetric hierarchical model combining a linear Gaussian model with a mixture of Gaussians, enabling additional analytical features over differentiable variants, but losing the ability to take advantage of computational speed up of diagonal covariance matrices in the latent space.

.. py:data:: goal.models.graphical.instances.AnalyticHMoG

   A fully analytic hierarchical model combining a linear Gaussian model with a mixture of Gaussians, enabling closed-form EM and parameter conversion.

Factory Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: goal.models.graphical.instances.differentiable_hmog
.. autofunction:: goal.models.graphical.instances.symmetric_hmog
.. autofunction:: goal.models.graphical.instances.analytic_hmog

Count Data Mixtures
-------------------

Mixture models based on count distributions.

Type Aliases
~~~~~~~~~~~~

.. py:data:: goal.models.graphical.instances.PoissonPopulation
   
   A collection of independent Poisson distributions.

.. py:data:: goal.models.graphical.instances.PopulationShape
   
   The shape component for a collection of COM-Poisson distributions.

.. py:data:: goal.models.graphical.instances.PoissonMixture
   
   A mixture of independent Poisson populations, useful for modeling 
   multivariate count data with multiple components.

.. py:data:: goal.models.graphical.instances.CoMPoissonMixture
   
   A mixture of Conway-Maxwell-Poisson populations, capable of modeling 
   dispersed count data with multiple components.

Factory Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: goal.models.graphical.instances.poisson_mixture
.. autofunction:: goal.models.graphical.instances.com_poisson_mixture

Embeddings
~~~~~~~~~~

\

.. inheritance-diagram:: goal.models.graphical.instances
   :parts: 2
   :top-classes: goal.geometry.exponential_family.combinators.LocationShape, goal.geometry.exponential_family.combinators.DifferentiableProduct, goal.geometry.manifold.embedding.TupleEmbedding 

\

.. autoclass:: goal.models.graphical.instances.CoMPoissonPopulation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.graphical.instances.PopulationLocationEmbedding
   :members:
   :undoc-members:
   :show-inheritance:
