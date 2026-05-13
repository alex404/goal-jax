Dynamical Subpackage
====================

State-space models. The transition slot of a ``LatentProcess`` is any ``Map[L, L]``, so the same machinery accepts an analytic Gaussian or categorical kernel (via ``AnalyticTransition``) or a feedforward map (a ``MultilayerPerceptron[L, L]`` plugged in directly) for hybrid filters with BPTT through alternating predict and conjugate update.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   transitions
   kalman_filter
   hmm
