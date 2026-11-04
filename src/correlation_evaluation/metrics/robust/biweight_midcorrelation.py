from dataclasses import dataclass
from typing import override

import numpy as np

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class BiweightMidcorrelation(CorrelationMetric):
    """
    Median-based correlation that smoothly down-weights outliers.

    Undefined when more than half of a variable's values are identical, because
    its median absolute deviation is then zero.

    Attributes
    ----------
    tuning_constant : float
        Number of median absolute deviations at which the weight reaches zero.
    """

    tuning_constant: float = 9.0

    def __post_init__(self) -> None:
        ParameterValidator.require_not_nan(tuning_constant=self.tuning_constant)
        if self.tuning_constant <= 0:
            raise ValueError("tuning_constant must be greater than 0")

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        weighted_x: FloatArray | None = self._weigh(x)
        weighted_y: FloatArray | None = self._weigh(y)
        if weighted_x is None or weighted_y is None:
            return CorrelationResult(statistic=None)
        return CorrelationResult(statistic=float(weighted_x @ weighted_y))

    def _weigh(self, values: FloatArray) -> FloatArray | None:
        deviations: FloatArray = values - np.median(values)
        absolute_deviations: FloatArray = np.abs(deviations)
        median_absolute_deviation: float = float(np.median(absolute_deviations))
        if median_absolute_deviation == 0:
            return None
        scaled: FloatArray = deviations / (
            self.tuning_constant * median_absolute_deviation
        )
        weighted: FloatArray = deviations * (1 - scaled**2) ** 2 * (np.abs(scaled) < 1)
        return weighted / np.linalg.norm(weighted)
