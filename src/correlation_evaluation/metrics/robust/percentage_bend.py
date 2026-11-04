import math
from dataclasses import dataclass
from typing import override

import numpy as np

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import BoolArray, FloatArray


@dataclass(frozen=True)
class PercentageBend(CorrelationMetric):
    """
    Wilcox's percentage bend correlation, which bounds the influence of outliers.

    Attributes
    ----------
    bend : float
        Fraction of samples allowed to be bent, in (0, 0.5].
    """

    bend: float = 0.2

    def __post_init__(self) -> None:
        ParameterValidator.require_real_numbers(bend=self.bend)
        if not 0 < self.bend <= 0.5:
            raise ValueError("bend must be in (0, 0.5]")

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        bent_x: FloatArray | None = self._bend(x)
        bent_y: FloatArray | None = self._bend(y)
        if bent_x is None or bent_y is None:
            return CorrelationResult(statistic=None)
        denominator: float = math.sqrt(float(bent_x @ bent_x) * float(bent_y @ bent_y))
        if denominator == 0:
            return CorrelationResult(statistic=None)
        statistic: float = float(bent_x @ bent_y) / denominator
        return CorrelationResult.from_t_test(
            statistic, num_samples=x.size, degrees_of_freedom=x.size - 2
        )

    def _bend(self, values: FloatArray) -> FloatArray | None:
        median: float = float(np.median(values))
        absolute_deviations: FloatArray = np.sort(np.abs(values - median))
        bend_index: int = math.floor((1 - self.bend) * values.size + 0.5) - 1
        scale: float = float(np.take(absolute_deviations, bend_index))
        if scale == 0:
            return None
        standardized: FloatArray = (values - median) / scale
        is_low: BoolArray = standardized < -1
        is_high: BoolArray = standardized > 1
        num_low: int = int(is_low.sum())
        num_high: int = int(is_high.sum())
        location: float = (
            float(values[~is_low & ~is_high].sum()) + scale * (num_high - num_low)
        ) / (values.size - num_low - num_high)
        return np.clip((values - location) / scale, -1, 1)
