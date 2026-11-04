import math
from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class Winsorized(CorrelationMetric):
    """
    Pearson correlation after pulling each tail of each variable to its quantile.

    Attributes
    ----------
    limit : float
        Fraction of samples replaced in each tail, in [0, 0.5).
    """

    limit: float = 0.2

    def __post_init__(self) -> None:
        ParameterValidator.require_real_numbers(limit=self.limit)
        if not 0 <= self.limit < 0.5:
            raise ValueError("limit must be in [0, 0.5)")

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        num_replaced: int = math.floor(self.limit * x.size)
        winsorized_x: FloatArray = self._winsorize(x, num_replaced)
        winsorized_y: FloatArray = self._winsorize(y, num_replaced)
        if np.ptp(winsorized_x) == 0 or np.ptp(winsorized_y) == 0:
            return CorrelationResult(statistic=None)
        statistic: float = float(stats.pearsonr(winsorized_x, winsorized_y).statistic)
        return CorrelationResult.from_t_test(
            statistic,
            num_samples=x.size,
            degrees_of_freedom=x.size - 2 * num_replaced - 2,
        )

    @staticmethod
    def _winsorize(values: FloatArray, num_replaced: int) -> FloatArray:
        ordered: FloatArray = np.sort(values)
        return np.clip(
            values, np.take(ordered, num_replaced), np.take(ordered, -num_replaced - 1)
        )
