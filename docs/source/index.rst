Goal: Geometric OptimizAtion Libraries
======================================

**(G)eometric (O)ptimiz(A)tion (L)ibraries**

A JAX framework for statistical modeling grounded in information geometry and exponential families.

Overview
--------

Goal provides machine learning algorithms that operate on statistical manifolds --- spaces where every point is a probability distribution. By embedding the mathematics of information geometry directly into its type system, Goal ensures that operations respect geometric constraints and enables efficient, composable algorithms for inference, learning, and model evaluation.

Mathematical Foundation
-----------------------

An exponential family is a collection of distributions with densities of the form

.. math::

   p(x; \theta) = \mu(x)\exp(\theta \cdot \mathbf{s}(x) - \psi(\theta))

where $\theta$ are the natural parameters, $\mathbf{s}(x)$ the sufficient statistic, $\mu(x)$ the base measure, and $\psi(\theta)$ the log-partition function. These families carry a dually flat Riemannian structure: the natural parameters $\theta$ and mean parameters $\eta = \nabla\psi(\theta)$ form dual coordinate systems connected by Legendre duality.

Library Structure
-----------------

The library is organized in two packages: **geometry** (abstract mathematical machinery) and **models** (concrete statistical distributions).

Geometry
~~~~~~~~

The :doc:`geometry package <geometry/index>` provides two layers:

- :doc:`geometry/manifold/index` --- Geometric primitives: manifolds, matrix representations, embeddings, and linear maps. A manifold is a stateless object that pairs a dimension with operations on flat JAX arrays.

- :doc:`geometry/exponential_family/index` --- Statistical manifolds with increasing capabilities:

  - **ExponentialFamily** --- sufficient statistics and base measure
  - **Generative** --- additionally supports sampling
  - **Differentiable** --- analytic log-partition function, enabling gradient-based optimization
  - **Analytic** --- analytic negative entropy, enabling closed-form algorithms (e.g. EM)

  Composed models (harmoniums, graphical models) recapitulate this hierarchy, mixing their own structure with these capability levels.

Models
~~~~~~

The :doc:`models package <models/index>` builds concrete distributions on this foundation:

- :doc:`models/base/index` --- Fundamental exponential family distributions:

  .. list-table::
     :header-rows: 1
     :widths: 25 35 40

     * - Distribution
       - Sufficient Statistic
       - Base Measure
     * - **Bernoulli**
       - $x \in \{0,1\}$
       - $0$
     * - **Categorical**
       - One-hot encoding for $k > 0$
       - $0$
     * - **Binomial**
       - Count $x$
       - $\log \binom{n}{x}$
     * - **Poisson**
       - Count $k$
       - $-\log(k!)$
     * - **CoM-Poisson**
       - $(k, \log(k!))$
       - $0$
     * - **von Mises**
       - $(\cos(\theta), \sin(\theta))$
       - $-\log(2\pi)$
     * - **Normal**
       - $(x, x \otimes x)$
       - $-\frac{d}{2}\log(2\pi)$
     * - **Boltzmann**
       - $x \otimes x$
       - $0$

- :doc:`models/harmonium/index` --- Conjugate latent-variable models (harmoniums): mixtures, linear Gaussian models, and Poisson mixtures.

- :doc:`models/graphical/index` --- Multi-level hierarchical models composing harmoniums, e.g. mixture of factor analyzers.

.. toctree::
   :maxdepth: 1
   :caption: API Reference:
   :hidden:

   geometry/index
   models/index
