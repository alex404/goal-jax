Dynamical Subpackage
====================

State-space models. The transition slot of a ``LatentProcess`` is any ``Map[L, L]``, so the same machinery accepts an analytic Gaussian or categorical kernel (via ``AnalyticTransition``) or a feedforward map (a :class:`~goal.geometry.manifold.map.MultilayerPerceptron` with matching domain and codomain, ``MultilayerPerceptron[L, L]``, plugged in directly) for hybrid filters with BPTT through alternating predict and conjugate update. No dedicated wrapper class is needed for the MLP case.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   kalman_filter
   hmm
