Examples
========

Each example lives in ``examples/<name>/`` with two entry points:

- ``run.py`` --- generates data, fits models, and saves results to ``analysis.json``
- ``plot.py`` --- loads results and produces ``plot.png``

Run an example with::

   python -m examples.<name>.run
   python -m examples.<name>.plot

Pass ``--gpu`` to run on GPU or ``--no-jit`` to disable JIT compilation for debugging.

Foundations
-----------

Basic exponential family fitting and covariance representations.

- **univariate_analytic** --- Closed-form MLE for :class:`~goal.models.base.gaussian.normal.Normal`, :class:`~goal.models.base.categorical.Categorical`, and :class:`~goal.models.base.poisson.Poisson`.
- **univariate_differentiable** --- Gradient-based fitting for :class:`~goal.models.base.von_mises.VonMises` and :class:`~goal.models.base.poisson.CoMPoisson`.
- **bivariate_normal** --- Density fitting with full, diagonal, and isotropic :class:`~goal.models.base.gaussian.normal.Normal` covariance.

Boltzmann Machines
------------------

Discrete latent models built on Boltzmann distributions.

- **boltzmann** --- MLE training and Gibbs sampling convergence diagnostics.
- **boltzmann_lgm** --- Hierarchical model: Boltzmann latent with Gaussian observable (concentric circles), using :class:`~goal.models.harmonium.lgm.BoltzmannLGM`.
- **boltzmann_lgm_cd** --- Contrastive divergence vs exact gradient comparison.

Mixture & Latent Variable Models
--------------------------------

Latent structure discovery with mixtures and factor models.

- **mixture_of_gaussians** --- EM for Gaussian mixtures (:class:`~goal.models.harmonium.mixture.AnalyticMixture`), compared against sklearn.
- **dimensionality_reduction** --- Factor analysis (:class:`~goal.models.harmonium.lgm.FactorAnalysis`) recovering a latent figure-8 curve from 20D observations.
- **mfa** --- Mixture of factor analyzers with per-component low-rank structure, using :class:`~goal.models.graphical.hmog.AnalyticHMoG`.
- **hmog** --- Hierarchical mixture of Gaussians (:class:`~goal.models.graphical.hmog.AnalyticHMoG`) with nested harmonium structure.

Neural Population Codes
-----------------------

Neuroscience applications modeling neural spike data.

- **poisson_mixture** --- Poisson vs CoM-Poisson spike count mixtures with Fano factor analysis, using :class:`~goal.models.harmonium.population_codes.PoissonVonMisesHarmonium`.
- **population_codes** --- Bayesian stimulus decoding from cosine-tuned neural responses (:class:`~goal.models.harmonium.population_codes.VonMisesPopulationCode`).
- **torus_poisson** --- Variational inference with non-differentiable von Mises sampler; compares free, regularized, and analytical conjugation parameter learning.
