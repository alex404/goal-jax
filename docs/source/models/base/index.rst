# docs/source/models/base/index.rst

Base Distributions Subpackage
=============================

This package provides basic probability distributions that serve as building blocks for more complex models. Each distribution is implemented as an exponential family with specific sufficient statistics and base measures.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   categorical
   poisson
   von_mises
   normal

Distribution Summary
--------------------

An exponential family is defined by its sufficient statistic and base measure
--- so far we provide implementations of the following basic distributions: 

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Distribution
     - Sufficient Statistic
     - Base Measure
   * - **Categorical**
     - One-hot encoding for $k > 0$
     - $0$
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
