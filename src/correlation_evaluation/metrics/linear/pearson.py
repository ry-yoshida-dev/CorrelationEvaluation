from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ...computation import ColumnCorrelation
from ...confidence_interval import ConfidenceInterval
from ...matrix import CorrelationMatrix
from ...metric import CorrelationMetric
from ...options import Alternative
from ...parameter_validator import ParameterValidator
from ...pearson_outcome import PearsonOutcome
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class Pearson(CorrelationMetric):
    """
    Pearson product-moment correlation, the strength of a linear relationship.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis of the test.
    confidence_level : float
        Level of the Fisher z confidence interval, in (0, 1).
    """

    alternative: Alternative = Alternative.TWO_SIDED
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)
        ParameterValidator.require_real_numbers(confidence_level=self.confidence_level)
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        outcome: PearsonOutcome = stats.pearsonr(
            x, y, alternative=self.alternative.option_name
        )
        low: float
        high: float
        # SciPy subtracts infinities at r = 1 with three samples, yet returns (-1, 1).
        with np.errstate(invalid="ignore"):
            low, high = outcome.confidence_interval(self.confidence_level)
        return CorrelationResult(
            statistic=float(outcome.statistic),
            p_value=float(outcome.pvalue),
            confidence_interval=ConfidenceInterval(
                low=float(low),
                high=float(high),
                level=self.confidence_level,
            ),
        )

    @override
    def _evaluate_columns(
        self, columns: NDArray[np.generic], encoded: list[FloatArray]
    ) -> CorrelationMatrix:
        num_samples: int = np.size(columns, axis=0)
        if num_samples < self.min_samples:
            return super()._evaluate_columns(columns, encoded)
        statistics: FloatArray = ColumnCorrelation.statistics(np.column_stack(encoded))
        return CorrelationMatrix(
            statistics=statistics,
            p_values=ColumnCorrelation.p_values(
                statistics, num_samples, self.alternative
            ),
        )
