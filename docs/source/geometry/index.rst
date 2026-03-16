Geometry Package
================

Abstract mathematical machinery for information geometry: manifolds and their operations, exponential family structure, and composed models.

Most users interact with the concrete distributions in :doc:`../models/index` rather than this package directly. The geometry layer provides the abstractions that make those models work: manifolds define how to operate on parameter arrays, and exponential families add statistical structure (sufficient statistics, log-partition functions, sampling) on top.

The layers build on each other: :doc:`manifold/index` (flat arrays with structure) → :doc:`exponential_family/index` (statistical families with increasing capabilities) → harmoniums and graphical models (composed latent-variable structures).

.. toctree::
   :maxdepth: 1
   :caption: Subpackages:

   manifold/index
   exponential_family/index
