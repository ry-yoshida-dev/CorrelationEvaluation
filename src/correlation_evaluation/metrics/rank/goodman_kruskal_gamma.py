import math
from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray, IntArray


@dataclass(frozen=True)
class GoodmanKruskalGamma(CorrelationMetric):
    """
    Goodman and Kruskal's gamma, which ignores tied pairs entirely.

    Recovers the concordance from Kendall's tau-b, so it takes O(n log n) time
    instead of comparing all pairs.
    """

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        tau_b: float = float(stats.kendalltau(x, y, method="asymptotic").statistic)
        total_pairs: float = self._count_tied_pairs(np.array([x.size]))
        tied_x: float = self._count_tied_pairs(np.unique(x, return_counts=True)[1])
        tied_y: float = self._count_tied_pairs(np.unique(y, return_counts=True)[1])
        tied_both: float = self._count_tied_pairs(
            np.unique(np.column_stack([x, y]), axis=0, return_counts=True)[1]
        )
        untied_pairs: float = total_pairs - tied_x - tied_y + tied_both
        concordance: float = tau_b * math.sqrt(
            (total_pairs - tied_x) * (total_pairs - tied_y)
        )
        return CorrelationResult(statistic=concordance / untied_pairs)

    @staticmethod
    def _count_tied_pairs(group_sizes: IntArray) -> float:
        return float((group_sizes * (group_sizes - 1)).sum() / 2)
