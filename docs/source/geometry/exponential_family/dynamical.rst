Dynamical
=========

.. automodule:: goal.geometry.exponential_family.dynamical
   :noindex:
   :no-members:

Class Hierarchy
---------------

.. inheritance-diagram:: goal.geometry.exponential_family.dynamical
   :parts: 2

\

Transitions
-----------

The transition slot of a ``LatentProcess`` is any :class:`~goal.geometry.manifold.map.Map` of the form ``Map[L, L]``: any parameterized map from a latent manifold's natural parameters back to itself. The filter scan calls it as ``predicted = trn_map(params, belief)``. ``AnalyticTransition`` is the analytic specialization backed by a conjugated harmonium kernel; for a hybrid filter, use a :class:`~goal.geometry.manifold.map.MultilayerPerceptron` with matching domain and codomain.

.. autoclass:: goal.geometry.exponential_family.dynamical.AnalyticTransition
   :members:
   :undoc-members:
   :show-inheritance:

Latent Processes
----------------

.. autoclass:: goal.geometry.exponential_family.dynamical.LatentProcess
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.dynamical.AnalyticLatentProcess
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.dynamical.VariationalLatentProcess
   :members:
   :undoc-members:
   :show-inheritance:

Helpers
-------

.. autofunction:: goal.geometry.exponential_family.dynamical.transpose_harmonium
