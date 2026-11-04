from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import BoolArray, FloatArray


@dataclass(frozen=True)
class HoeffdingD(CorrelationMetric):
    """
    Hoeffding's D, scaled by 30 so that it ranges from -0.5 to 1.

    Detects non-monotonic dependence. Ties use midranks, following SAS and Hmisc.
    Takes O(n^2) time and memory.
    """

    min_samples: ClassVar[int] = 5
    is_unit_bounded: ClassVar[bool] = False

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        n: int = x.size
        rank_x: FloatArray = stats.rankdata(x)
        rank_y: FloatArray = stats.rankdata(y)
        is_x_lower: BoolArray = x < x[:, np.newaxis]
        is_y_lower: BoolArray = y < y[:, np.newaxis]
        is_x_tied: BoolArray = ~(is_x_lower | is_x_lower.T)
        is_y_tied: BoolArray = ~(is_y_lower | is_y_lower.T)
        # Bivariate rank: ties count fractionally, and a sample is tied with itself.
        bivariate_rank: FloatArray = (
            1
            + (is_x_lower & is_y_lower).sum(axis=1)
            + 0.25 * ((is_x_tied & is_y_tied).sum(axis=1) - 1)
            + 0.5 * ((is_x_tied & is_y_lower) | (is_x_lower & is_y_tied)).sum(axis=1)
        )
        d1: float = float(((bivariate_rank - 1) * (bivariate_rank - 2)).sum())
        d2: float = float(
            ((rank_x - 1) * (rank_x - 2) * (rank_y - 1) * (rank_y - 2)).sum()
        )
        d3: float = float(((rank_x - 2) * (rank_y - 2) * (bivariate_rank - 1)).sum())
        numerator: float = (n - 2) * (n - 3) * d1 + d2 - 2 * (n - 2) * d3
        denominator: int = n * (n - 1) * (n - 2) * (n - 3) * (n - 4)
        return CorrelationResult(statistic=30 * numerator / denominator)
