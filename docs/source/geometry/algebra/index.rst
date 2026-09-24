Algebra Subpackage
==================

Stateless descriptions of how a flat parameter array is laid out and combined. Nothing here references :class:`~goal.geometry.manifold.base.Manifold` --- these are the descriptions that manifolds consume, not manifolds themselves.

:doc:`matrix` describes how parameters pack into a matrix and how to operate on it in packed form. :doc:`clique` describes which nodes of a graphical model interact, which is what fixes the block decomposition of its interaction tensor. :doc:`cut` regroups that same layout around a single node.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   clique
   cut
   matrix
