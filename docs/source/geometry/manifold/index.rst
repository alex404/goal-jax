Manifold Subpackage
===================

Geometric primitives for working with structured parameter arrays. A manifold is a stateless object that pairs a dimension with operations on flat JAX arrays of that dimension. This subpackage builds up from the base abstraction through combinators, matrix representations, embeddings, and linear maps.

Users of existing models rarely need to interact with this layer directly --- the model classes handle manifold operations internally. This package matters when defining custom manifolds or understanding how Goal represents parameters under the hood.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   base
   combinators
   embedding
   linear
   matrix
