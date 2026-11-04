from dataclasses import dataclass
from typing import override

from scipy import stats

from ...confidence_interval import ConfidenceInterval
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
        ParameterValidator.require_not_nan(confidence_level=self.confidence_level)
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        outcome: PearsonOutcome = stats.pearsonr(
            x, y, alternative=self.alternative.option_name
        )
        low: float
        high: float
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
