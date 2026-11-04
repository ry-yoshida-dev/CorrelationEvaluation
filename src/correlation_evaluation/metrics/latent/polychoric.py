from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import optimize, stats

from ...contingency_table import ContingencyTable
from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray, IntArray
from ...variable_kind import VariableKind
from .bivariate_normal import BivariateNormal


@dataclass(frozen=True)
class Polychoric(CorrelationMetric):
    """
    Polychoric correlation between two ordinal variables.

    Estimates the correlation of latent bivariate normal variables that were cut
    at thresholds, by the two-step maximum likelihood method: the thresholds come
    from the margins, then the correlation maximizes the likelihood. Labels are
    ordered by their sort order.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL

    # Beyond these, normal tail probabilities are indistinguishable from 0 and 1.
    _THRESHOLD_LIMIT: ClassVar[float] = 8.3
    _CORRELATION_LIMIT: ClassVar[float] = 0.9999
    _MIN_PROBABILITY: ClassVar[float] = 1e-300

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        counts: IntArray = ContingencyTable.from_codes(x, y).counts
        thresholds_x: FloatArray = self._thresholds(counts.sum(axis=1))
        thresholds_y: FloatArray = self._thresholds(counts.sum(axis=0))
        grid_x: FloatArray
        grid_y: FloatArray
        grid_x, grid_y = np.meshgrid(thresholds_x, thresholds_y, indexing="ij")

        def negative_log_likelihood(correlation: float) -> float:
            cumulative: FloatArray = BivariateNormal.cdf(grid_x, grid_y, correlation)
            cells: FloatArray = np.diff(np.diff(cumulative, axis=0), axis=1)
            return -float(
                (counts * np.log(np.clip(cells, self._MIN_PROBABILITY, None))).sum()
            )

        return CorrelationResult(
            statistic=float(
                optimize.minimize_scalar(
                    negative_log_likelihood,
                    bounds=(-self._CORRELATION_LIMIT, self._CORRELATION_LIMIT),
                    method="bounded",
                    options={"xatol": 1e-6},
                ).x
            )
        )

    def _thresholds(self, marginal_counts: IntArray) -> FloatArray:
        cumulative: FloatArray = np.cumsum(marginal_counts) / marginal_counts.sum()
        inner: FloatArray = stats.norm.ppf(cumulative[:-1])
        return np.concatenate(
            [[-self._THRESHOLD_LIMIT], inner, [self._THRESHOLD_LIMIT]]
        )
