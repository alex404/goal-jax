Linear Gaussian Models
======================

.. automodule:: goal.models.harmonium.lgm
   :noindex:
   :no-members:

Class Hierarchy
---------------

.. inheritance-diagram:: goal.models.harmonium.lgm
   :parts: 2

\

Generic LGM
-----------

.. autoclass:: goal.models.harmonium.lgm.LGM
   :members:
   :undoc-members:
   :show-inheritance:

Normal LGM
----------

.. autoclass:: goal.models.harmonium.lgm.NormalLGM
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.harmonium.lgm.NormalAnalyticLGM
   :members:
   :undoc-members:
   :show-inheritance:

Boltzmann LGM
-------------

.. autoclass:: goal.models.harmonium.lgm.BoltzmannLGM
   :members:
   :undoc-members:
   :show-inheritance:

Embeddings
----------

.. autoclass:: goal.models.harmonium.lgm.GeneralizedGaussianLocationEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.harmonium.lgm.NormalCovarianceEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

Specializations
---------------

.. data:: goal.models.harmonium.lgm.FactorAnalysis

   Type alias for ``NormalAnalyticLGM[Diagonal]`` — factor analysis with diagonal observable noise.

.. data:: goal.models.harmonium.lgm.PrincipalComponentAnalysis

   Type alias for ``NormalAnalyticLGM[Scale]`` — PCA with isotropic observable noise.

.. autofunction:: goal.models.harmonium.lgm.factor_analysis

.. autofunction:: goal.models.harmonium.lgm.principal_component_analysis

.. seealso::

   :doc:`/examples` --- **dimensionality_reduction**: Factor analysis recovering a latent curve from 20D observations; **mfa**: Mixture of factor analyzers.

