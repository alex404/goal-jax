Manifold Subpackage
===================

Core definitions for geometric objects, coordinate systems, and transformations.

Implementation Approach
-----------------------

This package provides:

- Abstract base classes defining manifolds,
- concrete implementations for common geometric structures,
- type-safe operations that respect manifold constraints, and
- efficient parameter handling and transformations.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   base
   combinators
   embedding
   linear
   matrix
   optimizer

Mathematical Framework
----------------------

A manifold $\mathcal{M}$ is a space that locally resembles Euclidean space. Key concepts include:

**Points and Coordinates**
  A point $p \in \mathcal{M}$ can be represented by coordinates in different systems. Coordinate charts $(U, \phi)$ map regions $U \subset \mathcal{M}$ to $\mathbb{R}^n$.

**Tangent and Cotangent Spaces**
  - The tangent space $T_p\mathcal{M}$ contains velocities of curves through $p$
  - The cotangent space $T^*_p\mathcal{M}$ contains linear functionals on $T_p\mathcal{M}$
  - Gradients of functions live in the cotangent space

**Geometric Operations**
  - Embedding: Injecting a manifold into a larger space
  - Projection: Mapping from a larger space to a submanifold 
  - Linear and affine maps: Structure-preserving transformations

Information Geometry Perspective
--------------------------------

In information geometry, manifolds often represent spaces of probability distributions.
Geometric structures like metrics and geodesics capture statistical relationships
between distributions, enabling principled parameter estimation and model comparison.
