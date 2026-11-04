import math
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
    The p-value comes from one-way analysis of variance.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        codes: IntArray = x.astype(np.int64)
        group_sizes: IntArray = np.bincount(codes)
        group_means: FloatArray = np.bincount(codes, weights=y) / group_sizes
        between: float = float((group_sizes * (group_means - y.mean()) ** 2).sum())
        total: float = float(((y - y.mean()) ** 2).sum())
        within: float = max(total - between, 0.0)
        num_groups: int = group_sizes.size
        degrees_of_freedom: int = y.size - num_groups
        p_value: float | None = None
        if degrees_of_freedom > 0 and within > 0:
            p_value = float(
                stats.f.sf(
                    (between / (num_groups - 1)) / (within / degrees_of_freedom),
                    num_groups - 1,
                    degrees_of_freedom,
                )
            )
        return CorrelationResult(statistic=math.sqrt(between / total), p_value=p_value)
