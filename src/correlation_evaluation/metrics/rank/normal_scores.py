from dataclasses import dataclass
from typing import override

from scipy import stats

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class NormalScores(CorrelationMetric):
    """
    Van der Waerden correlation, Pearson's correlation of normal quantiles of ranks.
    """

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        scores_x: FloatArray = stats.norm.ppf(stats.rankdata(x) / (x.size + 1))
        scores_y: FloatArray = stats.norm.ppf(stats.rankdata(y) / (y.size + 1))
        return CorrelationResult(
            statistic=float(stats.pearsonr(scores_x, scores_y).statistic)
        )
