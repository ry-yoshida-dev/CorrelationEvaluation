from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True, eq=False)
class CovariateAdjusted(CorrelationMetric, ABC):
    """
    Correlation after linearly removing covariates.

    The p-value tests whether x and y are related given the covariates, through
    the partial correlation. Metrics compare by identity, because they hold an
    array.

    Attributes
    ----------
    covariates : FloatArray
        The covariates with shape (n_samples,) or (n_samples, n_covariates).
    is_rank_based : bool
        Whether to rank all variables first, for a Spearman-type correlation.
    """

    covariates: FloatArray = field(repr=False)
    is_rank_based: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "covariates",
            VariableKind.MULTIVARIATE.encode(self.covariates, "covariates"),
        )

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        if self.covariates.shape[0] != x.size:
            raise ValueError(
                f"covariates must have one row per sample, got {self.covariates.shape[0]} for {x.size} samples"
            )
        covariates: FloatArray = self.covariates
        if self.is_rank_based:
            x = stats.rankdata(x)
            y = stats.rankdata(y)
            covariates = stats.rankdata(covariates, axis=0)
        residual_x: FloatArray = self._residualize(x, covariates)
        residual_y: FloatArray = self._residualize(y, covariates)
        adjusted_x: FloatArray = self._adjust_x(x, residual_x)
        if self._is_negligible(adjusted_x, x) or self._is_negligible(residual_y, y):
            return CorrelationResult(statistic=None)
        statistic: float = float(stats.pearsonr(adjusted_x, residual_y).statistic)
        if self._is_negligible(residual_x, x):
            return CorrelationResult(statistic=statistic)
        num_covariates: int = np.size(covariates, axis=1)
        return CorrelationResult.from_t_test(
            statistic,
            num_samples=x.size - num_covariates,
            degrees_of_freedom=x.size - num_covariates - 2,
            tested_correlation=float(stats.pearsonr(residual_x, residual_y).statistic),
        )

    @abstractmethod
    def _adjust_x(self, x: FloatArray, residual_x: FloatArray) -> FloatArray:
        """
        Choose the x that this metric correlates with the adjusted y.

        Parameters
        ----------
        x : FloatArray
            The first variable.
        residual_x : FloatArray
            The first variable with the covariates removed.

        Returns
        -------
        FloatArray
            The x to correlate.
        """

    @staticmethod
    def _residualize(values: FloatArray, covariates: FloatArray) -> FloatArray:
        design: FloatArray = np.column_stack([np.ones(values.size), covariates])
        coefficients: FloatArray = np.linalg.lstsq(design, values)[0]
        return values - design @ coefficients

    @staticmethod
    def _is_negligible(residuals: FloatArray, values: FloatArray) -> bool:
        # Residuals of a fully explained variable are rounding noise, not signal.
        return bool(
            np.linalg.norm(residuals) <= 1e-10 * np.linalg.norm(values - values.mean())
        )
