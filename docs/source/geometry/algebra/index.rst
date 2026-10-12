Algebra Subpackage
==================

Stateless descriptions of how a flat parameter array is laid out and combined. Nothing here references :class:`~goal.geometry.manifold.base.Manifold` --- these are the descriptions that manifolds consume, not manifolds themselves.

:doc:`matrix` describes how parameters pack into a matrix and how to operate on it in packed form. :doc:`clique` describes which nodes of a graphical model interact, which is what fixes the block decomposition of its interaction tensor. :doc:`junction_tree` arranges the cliques of a chordal graph in a tree and runs exact sum-product over binary variables on it.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   clique
   junction_tree
   matrix
