"""Variational conjugated models.

This module provides concrete implementations of variational inference
for harmonium models using learnable conjugation parameters.
"""

from .binomial_vonmises import (
    BinomialVonMisesVI,
    binomial_vonmises_vi,
)
from .binomial_vonmises import (
    make_elbo_loss_and_grad_fn as make_binomial_vonmises_loss_fn,
)
from .binomial_vonmises import (
    make_elbo_metrics_fn as make_binomial_vonmises_metrics_fn,
)

__all__ = [
    "BinomialVonMisesVI",
    "binomial_vonmises_vi",
    "make_binomial_vonmises_loss_fn",
    "make_binomial_vonmises_metrics_fn",
]
