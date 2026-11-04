from dataclasses import dataclass
from typing import override

from scipy import stats

from ...alternative import Alternative
from ...metric import CorrelationMetric
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
