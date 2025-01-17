from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from ....geometry import Manifold, Mean, Natural, Point


class GeneralizedGaussian[L: Manifold, S: Manifold](Manifold, ABC):
    r"""Protocol for exponential families with Gaussian-like sufficient statistics.

    This protocol captures the shared structure between Normal distributions and
    Boltzmann machines, where the sufficient statistics take the form:

    $$s(x) = (x, x \otimes x)$$

    with appropriate constraints on the second moment term for minimality.
    """

    @abstractmethod
    def split_location_precision(
        self, params: Point[Natural, Any]
    ) -> tuple[Point[Natural, L], Point[Natural, S]]:
        """Split parameters into location and precision in natural coordinates.

        Args:
            p: Parameters in natural coordinates

        Returns:
            loc: Location parameters
            precision: Precision/interaction parameters
        """
        ...

    @abstractmethod
    def join_location_precision(
        self, loc: Point[Natural, L], precision: Point[Natural, S]
    ) -> Point[Natural, Self]:
        """Join location and precision in natural coordinates.

        Args:
            loc: Location parameters
            precision: Precision/interaction parameters

        Returns:
            p: Combined parameters
        """
        ...

    @abstractmethod
    def split_mean_second_moment(
        self, means: Point[Mean, Self]
    ) -> tuple[Point[Mean, L], Point[Mean, S]]:
        """Split parameters into mean and covariance in mean coordinates.

        Args:
            p: Parameters in mean coordinates

        Returns:
            mean: Mean parameters
            covariance: Covariance/correlation parameters
        """
        ...

    @abstractmethod
    def join_mean_second_moment(
        self, mean: Point[Mean, L], second_moment: Point[Mean, S]
    ) -> Point[Mean, Self]:
        """Join mean and covariance in mean coordinates.

        Args:
            mean: Mean parameters
            second_moment: Second moment parameters

        Returns:
            p: Combined parameters
        """
        ...
