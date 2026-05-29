"""Variational MNIST clustering example.

Demonstrates variational conjugated model for MNIST digit clustering:
- Observable: ``Binomials`` (count pixels), ``Poissons`` (count, unbounded),
  or ``DiagonalNormal`` (continuous; pixels rescaled to ``[0, 1]``).
  ``Categorical -> Boltzmann -> Gaussian`` is the theoretically conjugated
  case for this hierarchy.
- Base latent: ``Bernoullis`` (independent binary hidden units) or
  ``ChordalBoltzmann`` (chain-coupled Boltzmann machine with exact junction-
  tree inference).
- Mixture: Categorical over K clusters.

The key insight is using analytical rho computation via least-squares
regression at each training step, which prevents the optimizer from
"cheating" by flattening the likelihood.

.. note::
   **Long-horizon NaN risk with the Normal observable.** ``Normal`` /
   ``DiagonalNormal`` has a constrained natural-parameter space: the
   precision component must stay strictly positive, otherwise
   ``log_partition_function`` (which calls ``inverse(precision)`` and
   ``logdet(precision)``) returns NaN. The affine likelihood ``b + L(s(z))``
   can push the precision across zero for an individual posterior-sampled
   ``z`` once the interaction ``L`` has grown large enough — propagating NaN
   into the ELBO gradient and corrupting the parameters.

   The conjugation regularizer only samples from the prior and so misses this
   failure mode: in observed cases ``conj_var`` was still finite at the very
   step the ELBO went NaN. Counting observables (``Binomials``,
   ``Poissons``) have bounded log-partition for all finite natural parameters
   and do not exhibit this failure, even with ``ChordalBoltzmann`` latents at
   ``||rho||`` orders of magnitude past the Normal failure threshold.

   Practical workarounds when staying with the Normal head:
   ``--grad-clip-norm 1.0`` (slows precision drift enough that no single Adam
   step crosses zero), or a precision floor inside
   ``Normal.log_partition_function`` (one-line clamp). See
   ``examples/variational_mnist/diag_stability.py`` for an instrumented
   trainer that reproduces, probes, and validates the mitigation.

Single-run workflow:
    python -m examples.variational_mnist.train
    python -m examples.variational_mnist.plot

Bernoullis-vs-ChordalBoltzmann performance comparison (matched seeds):
    python -m examples.variational_mnist.train --observable normal --latent bernoullis ...
    # move analysis.json/params.npz to a safe path, then:
    python -m examples.variational_mnist.train --observable normal \\
        --latent chordal_boltzmann --latent-graph chain ...
    python -m examples.variational_mnist.plot \\
        --compare PATH_A/analysis.json PATH_B/analysis.json \\
        --labels bernoullis chordal_boltzmann
"""
