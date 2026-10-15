Variational Inference
=====================

.. automodule:: goal.geometry.exponential_family.variational
   :noindex:
   :no-members:

Class Hierarchy
---------------

.. inheritance-diagram:: goal.geometry.exponential_family.variational
   :parts: 2
   :top-classes: goal.geometry.manifold.combinators.Pair, goal.geometry.manifold.map.Map

\

Variational Conjugated
----------------------

.. autoclass:: goal.geometry.exponential_family.variational.VariationalConjugated
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.variational.DifferentiableVariationalConjugated
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.variational.DeepModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.variational.DifferentiableDeepModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: goal.geometry.exponential_family.variational.variational_level

Conjugation Functions
---------------------

.. autoclass:: goal.geometry.exponential_family.variational.ExactConjugationFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.variational.ConstantConjugationFunction
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.variational.BiasConjugationFunction
   :members:
   :undoc-members:
   :show-inheritance:

Fitting and Diagnostics
-----------------------

.. autofunction:: goal.geometry.exponential_family.variational.regress_conjugation_parameters

.. autofunction:: goal.geometry.exponential_family.variational.conjugation_metrics

.. autofunction:: goal.geometry.exponential_family.variational.reconstruct

.. autofunction:: goal.geometry.exponential_family.variational.reconstruction_error
