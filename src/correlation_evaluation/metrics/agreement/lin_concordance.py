from dataclasses import dataclass
from typing import override

import numpy as np

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class LinConcordance(CorrelationMetric):
    """
    Lin's concordance correlation, the agreement of two measurements with the
    identity line. Unlike Pearson's correlation, it penalizes shifts and rescaling.
    """

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        covariance: float = float(((x - x.mean()) * (y - y.mean())).mean())
        mean_difference: float = float(x.mean() - y.mean())
        return CorrelationResult(
            statistic=2
            * covariance
            / (float(np.var(x)) + float(np.var(y)) + mean_difference**2)
        )
