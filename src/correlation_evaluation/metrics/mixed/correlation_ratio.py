import math
import sys
from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray, IntArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class CorrelationRatio(CorrelationMetric):
    """
    Correlation ratio (eta) of a categorical x with a numeric y.

    The square root of the share of the variance of y between the groups of x.
    The p-value comes from one-way analysis of variance; it is 0 when the groups
    explain y perfectly, and None when each sample has its own group.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        codes: IntArray = x.astype(np.int64)
        group_sizes: IntArray = np.bincount(codes)
        group_means: FloatArray = np.bincount(codes, weights=y) / group_sizes
        total: float = float(((y - y.mean()) ** 2).sum())
        if total == 0:
            return CorrelationResult(statistic=None)
        within: float = float(((y - group_means[codes]) ** 2).sum())
        # Rounding of equal values within groups is not within-group variation.
        magnitudes: FloatArray = np.abs(y)
        rounding: float = 8 * sys.float_info.epsilon * float(np.max(magnitudes))
        if within <= y.size * rounding * rounding:
            within = 0.0
        between: float = max(total - within, 0.0)
        num_groups: int = group_sizes.size
        degrees_of_freedom: int = y.size - num_groups
        p_value: float | None = None
        if degrees_of_freedom > 0:
            p_value = (
                float(
                    stats.f.sf(
                        (between / (num_groups - 1)) / (within / degrees_of_freedom),
                        num_groups - 1,
                        degrees_of_freedom,
                    )
                )
                if within > 0
                else 0.0
            )
        return CorrelationResult(statistic=math.sqrt(between / total), p_value=p_value)
