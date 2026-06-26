"""Chordal Boltzmann PPC: conjugacy and data-learning.

A population code of correlated binary neurons (a Boltzmann machine with chordal
pairwise couplings) that encodes a continuous Gaussian latent, built on
:class:`~goal.models.harmonium.population_codes.BoltzmannPopulationCode`. The
example shows that the code can be made (near-)conjugate to the latent --- its
log-partition is affine in the latent sufficient statistics --- and that this
conjugacy survives training on data via the variational ELBO. See ``run.py``.
"""
