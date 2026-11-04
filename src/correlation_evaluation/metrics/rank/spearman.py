from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ...computation import ColumnCorrelation
from ...matrix import CorrelationMatrix
from ...metric import CorrelationMetric
from ...options import Alternative
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class Spearman(CorrelationMetric):
    """
    Spearman's rank correlation, the strength of a monotonic relationship.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis of the test.
    """

    alternative: Alternative = Alternative.TWO_SIDED

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult.from_test_outcome(
            stats.spearmanr(x, y, alternative=self.alternative.option_name)
        )

    @override
    def _evaluate_columns(
        self, columns: NDArray[np.generic], encoded: list[FloatArray]
    ) -> CorrelationMatrix:
        num_samples: int = np.size(columns, axis=0)
        if num_samples < self.min_samples:
            return super()._evaluate_columns(columns, encoded)
        statistics: FloatArray = ColumnCorrelation.statistics(
            stats.rankdata(np.column_stack(encoded), axis=0)
        )
        return CorrelationMatrix(
            statistics=statistics,
            p_values=ColumnCorrelation.p_values(
                statistics, num_samples, self.alternative
            ),
        )
