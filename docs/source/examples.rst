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
- **multivariate** --- Side-by-side benchmark: analytic 2D :class:`~goal.models.base.gaussian.normal.Normal` fit via source and natural parameterizations, plus gradient-based fitting of a :class:`~goal.models.base.dirichlet.Dirichlet` on a 3-simplex.

Boltzmann Machines
------------------

Discrete latent models built on Boltzmann distributions.

- **boltzmann** --- MLE training and Gibbs sampling convergence diagnostics.
- **boltzmann_lgm** --- Hierarchical model: Boltzmann latent with Gaussian observable (concentric circles), using :class:`~goal.models.harmonium.lgm.NormalBoltzmannMachine`.

Mixture & Latent Variable Models
--------------------------------

Latent structure discovery with mixtures and factor models.

- **mixture_of_gaussians** --- EM for Gaussian mixtures (:class:`~goal.models.harmonium.mixture.AnalyticMixture`), compared against sklearn.
- **mfa** --- Mixture of factor analyzers with per-component low-rank structure, using :class:`~goal.models.graphical.hmog.AnalyticHMoG`.
- **hmog** --- Hierarchical mixture of Gaussians (:class:`~goal.models.graphical.hmog.AnalyticHMoG`) with nested harmonium structure.
- **cca** --- Two-view canonical correlation analysis: two 2D views share a 1D latent, fit by gradient ascent and EM (:class:`~goal.models.CanonicalCorrelationAnalysis`, :class:`~goal.models.AnalyticCanonicalCorrelationAnalysis`).

Dynamical Models
----------------

- **kalman_filter** --- Filtering, smoothing, gradient and EM learning of a damped oscillator (:class:`~goal.models.KalmanFilter`).
- **hmm** --- Filtering, smoothing, gradient and EM learning of a hidden Markov model with sticky transitions (:class:`~goal.models.HiddenMarkovModel`).

Neural Population Codes
-----------------------

Neuroscience applications modeling neural spike data.

- **poisson_mixture** --- Poisson vs CoM-Poisson spike count mixtures with Fano factor analysis (:func:`~goal.models.harmonium.population_codes.poisson_mixture`, :func:`~goal.models.harmonium.population_codes.com_poisson_mixture`).
- **population_codes** --- Bayesian stimulus decoding from cosine-tuned neural responses (:class:`~goal.models.harmonium.population_codes.PoissonPopulationCode`, :class:`~goal.models.variational.population_codes.VariationalPopulationCode`).
- **torus_poisson** --- Variational inference with non-differentiable von Mises sampler; compares free and regularized conjugation parameter learning.
- **chordal_boltzmann_ppc** --- A population code of correlated binary neurons on a chordal graph, made near-conjugate to a Gaussian latent and trained on data (:class:`~goal.models.harmonium.population_codes.BoltzmannPopulationCode`, :class:`~goal.models.variational.population_codes.VariationalPopulationCode`).
- **canonical_circuit** --- The one-layer canonical circuit (:class:`~goal.models.CanonicalCircuit`) on a bump on a retina, without z, partially exact and approximate, by the penalized ELBO, measured exactly by enumeration and quadrature.
- **pendulum** --- Filtering a stochastic pendulum from a Poisson population tuned to angle and velocity (:class:`~goal.models.variational.population_code_process.PopulationCodeProcess`).
