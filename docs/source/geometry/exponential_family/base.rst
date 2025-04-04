Exponential Families
====================

.. automodule:: goal.geometry.exponential_family.base
   :noindex:
   :no-members:


Class Hierarchy
---------------

.. inheritance-diagram:: goal.geometry.exponential_family.base
   :parts: 2

\

Coordinate Systems
----------------

.. autoclass:: goal.geometry.exponential_family.base.Natural

.. py:data:: type Mean = Dual[Natural]

   Mean parameters for exponential families.

   In practice, Mean parameters represent expected sufficient statistics and correspond
   to the moments of the distribution. They are the dual coordinates to Natural parameters
   and are used in moment matching and variational inference.

   In theory, mean parameters :math:`\eta \in \text{H}` are given by expectations of sufficient statistics:

   .. math::

      \eta = \mathbb{E}_{p(x;\theta)}[\mathbf{s}(x)]

   They form the set :math:`\text{H} = \{\eta : \eta = \mathbb{E}_p[\mathbf{s}(X)], p \in \mathcal{P}\}`.

Class Hierarchy
---------------

.. autoclass:: goal.geometry.exponential_family.base.ExponentialFamily
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.base.Generative
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.base.Differentiable
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.exponential_family.base.Analytic
   :members:
   :undoc-members:
   :show-inheritance:
