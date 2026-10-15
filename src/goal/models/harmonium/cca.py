"""Canonical correlation analysis: two observable Gaussians sharing one latent.

The graph is a fork, $x - z - y$, with $x$ and $y$ conditionally independent given $z$. The model is
a :class:`~goal.geometry.exponential_family.graphical.DifferentiableGraphical` with two linear
Gaussian models placed on the one latent node, so its conjugation parameters are the sum of theirs.
With a full-covariance latent the model is analytic and each branch's likelihood is fit on its own
(:class:`AnalyticCanonicalCorrelationAnalysis`).
"""

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
    def obs_hrms_lat_nods(
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
