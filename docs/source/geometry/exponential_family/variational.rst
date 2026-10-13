Variational Inference
=====================

.. automodule:: goal.geometry.exponential_family.variational
   :noindex:
   :no-members:

Class Hierarchy
---------------

.. inheritance-diagram:: goal.geometry.exponential_family.variational
   :parts: 2
   :top-classes: goal.geometry.manifold.combinators.Triple

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

Conjugation Functions
---------------------

.. autoclass:: goal.geometry.exponential_family.variational.ExactConjugationFunction
   :members:
   :undoc-members:
   :show-inheritance:

Fitting and Diagnostics
-----------------------

.. autofunction:: goal.geometry.exponential_family.variational.regress_conjugation_parameters

.. autofunction:: goal.geometry.exponential_family.variational.conjugation_metrics

.. autofunction:: goal.geometry.exponential_family.variational.reconstruct

.. autofunction:: goal.geometry.exponential_family.variational.reconstruction_error
