GOAL: Geometric OptimizAtion Libraries
======================================

**(G)eometric (O)ptimiz(A)tion (L)ibraries**

A principled framework for statistical modeling and machine learning grounded in the mathematics of information geometry and exponential families.

Overview
--------

GOAL provides efficient machine learning algorithms grounded in the theory statistical manifolds and information geometry, enabling generalized approaches to inference, learning, and model evaluation. The library embeds mathematical concepts directly into its type system, ensuring that operations respect the underlying geometric constraints.

By building on the theory of information geometry, GOAL offers:
    * **Geometric Optimization** — Algorithms that respect the intrinsic properties of probabilistic models
    * **Type-Safe Statistical Models** — Ensuring validity and correctness of operations
    * **Flexible Composition** — Building complex models from simple components through mathematically sound operations

Mathematical Foundation
-----------------------

At its core, GOAL implements the dually flat structure of exponential families, where:
    * Statistical models form Riemannian manifolds
    * Natural and mean parameterizations provide dual coordinate systems
    * Divergences and metrics capture statistical relationships

The library balances mathematical rigor with computational efficiency, providing specialized implementations for common structures while maintaining a unified conceptual framework.

Library Structure
-----------------

GOAL is organized into two primary packages:

* **Geometry** — Core mathematical structures: manifolds, embeddings, exponential families
* **Models** — Statistical distributions and composed models built on the geometric foundation

These modules work together to provide a comprehensive framework for statistical modeling with a strong theoretical foundation.

.. toctree::
   :maxdepth: 1
   :caption: Documentation:

   geometry/index
   models/index
