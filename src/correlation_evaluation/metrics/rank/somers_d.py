import math
from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...options import Alternative
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray, IntArray


@dataclass(frozen=True)
class SomersD(CorrelationMetric):
    """
    Somers' D of y given x, Kendall's tau normalized by pairs untied on x.

    The p-value uses the asymptotic normal test of Agresti, as SciPy does, but from
    per-sample scores rather than a table with one cell per pair of distinct values.
    Takes O(n^2) time and O(n) memory.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis of the test.
    """

    alternative: Alternative = Alternative.TWO_SIDED

    is_symmetric: ClassVar[bool] = False

    _CHUNK_SIZE: ClassVar[int] = 256

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        n: int = x.size
        x_counts: IntArray
        codes_x: IntArray
        _, codes_x, x_counts = np.unique(x, return_inverse=True, return_counts=True)
        codes_y: IntArray = np.unique(y, return_inverse=True)[1]
        scores: IntArray = self._concordance_scores(codes_x, codes_y)
        # Python integers keep these exact, so a zero variance is exactly zero.
        total: int = int(scores.sum())
        scaled_variance: int = n * int((scores**2).sum()) - total**2
        statistic: float = total / (n**2 - int((x_counts**2).sum()))
        if total == 0 and scaled_variance == 0:
            return CorrelationResult(statistic=statistic)
        z_score: float = (
            total / math.sqrt(4 * scaled_variance / n)
            if scaled_variance > 0
            else math.copysign(math.inf, total)
        )
        return CorrelationResult(statistic=statistic, p_value=self._p_value(z_score))

    def _concordance_scores(self, codes_x: IntArray, codes_y: IntArray) -> IntArray:
        # Each sample's concordant minus discordant partners, a chunk at a time to
        # bound the memory of what would otherwise be an n by n comparison.
        scores: IntArray = np.zeros(codes_x.size, dtype=np.int64)
        for start in range(0, codes_x.size, self._CHUNK_SIZE):
            stop: int = start + self._CHUNK_SIZE
            scores[start:stop] = (
                np.sign(codes_x - codes_x[start:stop, np.newaxis])
                * np.sign(codes_y - codes_y[start:stop, np.newaxis])
            ).sum(axis=1)
        return scores

    def _p_value(self, z_score: float) -> float:
        match self.alternative:
            case Alternative.TWO_SIDED:
                return 2 * float(stats.norm.sf(abs(z_score)))
            case Alternative.GREATER:
                return float(stats.norm.sf(z_score))
            case Alternative.LESS:
                return float(stats.norm.cdf(z_score))
