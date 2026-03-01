Manifold Subpackage
===================

Geometric primitives for working with structured parameter arrays. A :class:`~goal.geometry.manifold.base.Manifold` is a stateless object that pairs a dimension with operations on flat JAX arrays of that dimension. This package builds up from there:

- :mod:`~goal.geometry.manifold.base` --- the base abstraction,
- :mod:`~goal.geometry.manifold.combinators` --- products, tuples, and replication,
- :mod:`~goal.geometry.manifold.matrix` --- storage-efficient matrix representations,
- :mod:`~goal.geometry.manifold.embedding` --- subspace relationships and pullbacks,
- :mod:`~goal.geometry.manifold.linear` --- linear and affine maps between manifolds.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   base
   combinators
   embedding
   linear
   matrix
