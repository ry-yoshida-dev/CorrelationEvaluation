from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Self, override

import numpy as np
from scipy import stats

from ...computation import ColumnSpace
from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray, ScalingMethod
from ...variable_kind import VariableKind


@dataclass(frozen=True, eq=False)
class CovariateAdjusted(CorrelationMetric, ABC):
    """
    Correlation after linearly removing covariates.

    The p-value tests whether x and y are related given the covariates, through
    the partial correlation. Undefined when no degrees of freedom remain, where a
    partial correlation is always 1 in magnitude. Collinear covariates count once,
    and the units and offsets of covariates do not affect the value.
    Metrics compare by identity, because they hold an array.

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
        ParameterValidator.require_booleans(is_rank_based=self.is_rank_based)
        object.__setattr__(
            self,
            "covariates",
            VariableKind.MULTIVARIATE.encode(self.covariates, "covariates"),
        )

    @override
    def _validate_num_samples(self, num_samples: int) -> None:
        num_rows: int = np.size(self.covariates, axis=0)
        if num_rows != num_samples:
            raise ValueError(
                f"covariates must have one row per sample, got {num_rows} for {num_samples} samples"
            )

    @override
    def _with_scaled_data(self, scaling: ScalingMethod) -> Self:
        # A nonlinear method would otherwise change how x and y relate to the
        # covariates, so that removing them no longer removes their effect.
        return replace(
            self,
            covariates=scaling.build_scaler()
            .fit_transform(self.covariates)
            .astype(np.float64),
        )

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        covariates: FloatArray = self.covariates
        if self.is_rank_based:
            x = stats.rankdata(x)
            y = stats.rankdata(y)
            covariates = stats.rankdata(covariates, axis=0)
        basis: FloatArray = ColumnSpace.orthonormal_basis(covariates)
        num_covariates: int = np.size(basis, axis=1)
        degrees_of_freedom: int = x.size - num_covariates - 2
        if degrees_of_freedom <= 0:
            return CorrelationResult(statistic=None)
        residual_x: FloatArray = self._residualize(x, basis)
        residual_y: FloatArray = self._residualize(y, basis)
        adjusted_x: FloatArray = self._adjust_x(x, residual_x)
        if self._is_negligible(adjusted_x, x) or self._is_negligible(residual_y, y):
            return CorrelationResult(statistic=None)
        statistic: float = float(stats.pearsonr(adjusted_x, residual_y).statistic)
        if self._is_negligible(residual_x, x):
            return CorrelationResult(statistic=statistic)
        return CorrelationResult.from_t_test(
            statistic,
            num_samples=x.size - num_covariates,
            degrees_of_freedom=degrees_of_freedom,
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
    def _residualize(values: FloatArray, basis: FloatArray) -> FloatArray:
        centered: FloatArray = values - values.mean()
        return centered - basis @ (basis.T @ centered)

    @staticmethod
    def _is_negligible(residuals: FloatArray, values: FloatArray) -> bool:
        # Residuals of a fully explained variable are rounding noise, not signal.
        return bool(
            np.linalg.norm(residuals) <= 1e-10 * np.linalg.norm(values - values.mean())
        )
