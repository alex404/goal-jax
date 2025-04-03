Base Manifold Classes
=====================

.. automodule:: goal.geometry.manifold.base
   :noindex:
   :no-members:

Class Hierarchy
---------------

.. inheritance-diagram:: goal.geometry.manifold.base
   :parts: 1

\

Manifold
--------

.. autoclass:: goal.geometry.manifold.base.Manifold
   :members:
   :undoc-members:
   :show-inheritance:

Coordinates
------------

.. autoclass:: goal.geometry.manifold.base.Coordinates
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: goal.geometry.manifold.base.Dual
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: goal.geometry.manifold.base.reduce_dual
.. autofunction:: goal.geometry.manifold.base.expand_dual

Points
------

.. data:: goal.geometry.manifold.base.Point
   
   A :class:`Point` on a :class:`Manifold` in given :class:`Coordinates`.
   
   This is a public type alias for the underlying :class:`_Point` implementation class, which should not be instantiated directly.

.. autoclass:: goal.geometry.manifold.base._Point
   :members:
   :undoc-members:
   :show-inheritance:
