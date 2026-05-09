Transitions and Emissions
=========================

.. automodule:: goal.models.dynamical
   :noindex:
   :no-members:

Linear Gaussian
---------------

.. autoclass:: goal.models.dynamical.LinearGaussianTransition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.dynamical.NormalEmission
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: goal.models.dynamical.create_linear_gaussian_transition

Categorical
-----------

.. autoclass:: goal.models.dynamical.CategoricalKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.dynamical.CategoricalTransition
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.models.dynamical.CategoricalEmission
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: goal.models.dynamical.create_categorical_transition

MLP
---

For a hybrid filter, use a :class:`~goal.geometry.manifold.map.MultilayerPerceptron` with matching domain and codomain (``MultilayerPerceptron[L, L]``) directly as the transition slot of a ``LatentProcess``; no dedicated wrapper class is needed.
