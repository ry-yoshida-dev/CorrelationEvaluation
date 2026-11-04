from dataclasses import dataclass
from typing import ClassVar, override

from scipy import stats

from ...alternative import Alternative
from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class SomersD(CorrelationMetric):
    """
    Somers' D of y given x, Kendall's tau normalized by pairs untied on x.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis of the test.
    """

    alternative: Alternative = Alternative.TWO_SIDED

    is_symmetric: ClassVar[bool] = False

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult.from_test_outcome(
            stats.somersd(x, y, alternative=self.alternative.option_name)
        )
