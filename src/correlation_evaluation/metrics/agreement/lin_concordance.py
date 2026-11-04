from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...scaling_restriction import ScalingRestriction
from ...types import FloatArray


@dataclass(frozen=True)
class LinConcordance(CorrelationMetric):
    """
    Lin's concordance correlation, the agreement of two measurements with the
    identity line. Unlike Pearson's correlation, it penalizes shifts and rescaling.
    """

    scaling_restriction: ClassVar[ScalingRestriction | None] = (
        ScalingRestriction.for_agreement()
    )

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        covariance: float = float(((x - x.mean()) * (y - y.mean())).mean())
        mean_difference: float = float(x.mean() - y.mean())
        # Unlike the power operator, multiplying floats overflows to infinity.
        denominator: float = (
            float(np.var(x)) + float(np.var(y)) + mean_difference * mean_difference
        )
        if not 0 < denominator < np.inf:
            return CorrelationResult(statistic=None)
        return CorrelationResult(statistic=2 * covariance / denominator)
