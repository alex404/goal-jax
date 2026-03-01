"""Cross-cutting capability protocols for exponential families.

These protocols express optional capabilities that cut across the main inheritance
hierarchy. They exist because Python lacks intersection types --- a class can be
both an ``ExponentialFamily`` and support ``StatisticalMoments``, but the type
system cannot express that constraint directly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class StatisticalMoments(Protocol):
    """Protocol for distributions that can compute statistical moments.

    This is a structural type: any class with matching ``statistical_mean`` and
    ``statistical_covariance`` methods satisfies it, without explicit inheritance.
    """

    def statistical_mean(self, params: Array) -> Array:
        """Compute the mean of the distribution from its natural parameters."""
        ...

    def statistical_covariance(self, params: Array) -> Array:
        """Compute the covariance matrix of the distribution from its natural parameters."""
        ...
