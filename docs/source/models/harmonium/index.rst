Harmonium Models Subpackage
===========================

Conjugate latent-variable models with harmonium structure: mixtures over any exponential family, linear Gaussian models (factor analysis, PCA), and Poisson mixtures for count data.

A harmonium is a latent-variable model where hidden and observed variables have a joint exponential family distribution. The conjugate structure ensures that the posterior over latent variables stays in the same exponential family as the prior, which enables exact E-steps and closed-form algorithms. The choice of latent variable determines the model type:

- **Categorical latent** → mixture model (:doc:`mixture`)
- **Gaussian latent** → factor analysis / PCA (:doc:`lgm`)
- **Von Mises latent** → population code for circular stimuli (:doc:`population_codes`)

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   lgm
   mixture
   population_codes
