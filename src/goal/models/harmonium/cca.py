"""Canonical correlation analysis: two observable Gaussians sharing one latent.

The first model in this library over a graph with **more than one root**. Where a linear
Gaussian model is a chain $x - z$, this is a fork

.. code-block:: text

       z
      / \\
     x   y

with $x$ and $y$ conditionally independent given $z$. That independence is what makes the
model tractable, and it is carried by the structure rather than asserted: the model is a
:class:`~goal.geometry.exponential_family.graphical.DifferentiableGraphical` with two
linear Gaussian models attached to the one latent node, and its observable is their
observables side by side, whose log-partition function is the sum of its components'.

Mathematically, the conjugation equation's left-hand side therefore factorizes,

.. math::
    \\psi_{XY}(\\theta_{XY} + \\Theta \\cdot \\mathbf s_Z(z))
      = \\psi_X(\\theta_X + \\Theta_{XZ} \\cdot \\mathbf s_Z(z))
      + \\psi_Y(\\theta_Y + \\Theta_{YZ} \\cdot \\mathbf s_Z(z)),

and since each branch is a linear Gaussian model with its own conjugation, the joint
conjugation parameters are their sum, $\\rho = \\rho_X + \\rho_Y$, and likewise the
offsets. The graphical harmonium computes both sums for any number of attached harmoniums.
The same independence makes the expected log-likelihood a sum over the branches, so with a
full-covariance latent the model is analytic and each branch's likelihood is fit on its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

from ...geometry import (
    AnalyticGraphical,
    Differentiable,
    DifferentiableConjugated,
    DifferentiableGraphical,
    Harmonium,
    PositiveDefinite,
)
from ..base.gaussian.normal import FullNormal, Normal, full_normal
from .lgm import NormalAnalyticLGM, NormalCovarianceEmbedding, NormalLGM


@dataclass(frozen=True)
class _CCABase[
    FstRep: PositiveDefinite,
    SndRep: PositiveDefinite,
    FstLGM: DifferentiableConjugated[Any, Any, Any],
    SndLGM: DifferentiableConjugated[Any, Any, Any],
](DifferentiableGraphical[Any, Any], ABC):
    """Two observable normals coupled through one shared Gaussian latent.

    Data points are the concatenation $[x, y, z]$, and the two observables may have
    different dimensions and different covariance structures. Subclasses fix the latent
    and the linear Gaussian model of each branch.
    """

    # Fields

    fst_dim: int
    """Data dimension of the first observable."""

    fst_rep: FstRep
    """Covariance structure of the first observable."""

    snd_dim: int
    """Data dimension of the second observable."""

    snd_rep: SndRep
    """Covariance structure of the second observable."""

    lat_dim: int
    """Dimension of the shared latent."""

    # Contract

    @property
    @abstractmethod
    def fst_lgm(self) -> FstLGM:
        """The first branch as a standalone linear Gaussian model."""

    @property
    @abstractmethod
    def snd_lgm(self) -> SndLGM:
        """The second branch as a standalone linear Gaussian model."""

    # Overrides

    @property
    @override
    def obs_hrms_att_clqs(
        self,
    ) -> tuple[tuple[Harmonium[Differentiable, Any], tuple[int, ...]], ...]:
        """The fork $x - z - y$: one linear Gaussian model per branch, both on the shared latent."""
        return ((self.fst_lgm, (0,)), (self.snd_lgm, (0,)))


@dataclass(frozen=True)
class CanonicalCorrelationAnalysis[
    FstRep: PositiveDefinite,
    SndRep: PositiveDefinite,
    PstRep: PositiveDefinite,
](
    _CCABase[FstRep, SndRep, NormalLGM[FstRep, PstRep], NormalLGM[SndRep, PstRep]],
    DifferentiableGraphical[Normal[PstRep], FullNormal],
):
    """Canonical correlation analysis with a posterior latent of any covariance structure, fit by gradient descent."""

    # Fields

    pst_rep: PstRep
    """Covariance structure of the posterior latent."""

    # Overrides

    @property
    @override
    def pst_man(self) -> Normal[PstRep]:
        return Normal(self.lat_dim, self.pst_rep)

    @property
    @override
    def pst_prr_emb(self) -> NormalCovarianceEmbedding[PositiveDefinite, PstRep]:
        return NormalCovarianceEmbedding(full_normal(self.lat_dim), self.pst_man)

    @property
    @override
    def fst_lgm(self) -> NormalLGM[FstRep, PstRep]:
        return NormalLGM(self.fst_dim, self.fst_rep, self.lat_dim, self.pst_rep)

    @property
    @override
    def snd_lgm(self) -> NormalLGM[SndRep, PstRep]:
        return NormalLGM(self.snd_dim, self.snd_rep, self.lat_dim, self.pst_rep)


@dataclass(frozen=True)
class AnalyticCanonicalCorrelationAnalysis[
    FstRep: PositiveDefinite,
    SndRep: PositiveDefinite,
](
    _CCABase[FstRep, SndRep, NormalAnalyticLGM[FstRep], NormalAnalyticLGM[SndRep]],
    AnalyticGraphical[FullNormal],
):
    """Canonical correlation analysis with a full-covariance latent, with closed-form conversion from mean to natural parameters and exact EM."""

    # Overrides

    @property
    @override
    def lat_man(self) -> FullNormal:
        return full_normal(self.lat_dim)

    @property
    @override
    def fst_lgm(self) -> NormalAnalyticLGM[FstRep]:
        return NormalAnalyticLGM(self.fst_dim, self.fst_rep, self.lat_dim)

    @property
    @override
    def snd_lgm(self) -> NormalAnalyticLGM[SndRep]:
        return NormalAnalyticLGM(self.snd_dim, self.snd_rep, self.lat_dim)
