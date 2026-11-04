from dataclasses import dataclass
from typing import ClassVar, override

from scipy import stats

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class ChatterjeeXi(CorrelationMetric):
    """
    Chatterjee's xi, near 1 when y is a noiseless function of x and near 0 under
    independence.

    Attributes
    ----------
    is_y_continuous : bool
        Whether y is assumed to have no ties, which simplifies the p-value.
    """

    is_y_continuous: bool = False

    is_symmetric: ClassVar[bool] = False

    def __post_init__(self) -> None:
        ParameterValidator.require_booleans(is_y_continuous=self.is_y_continuous)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult.from_test_outcome(
            stats.chatterjeexi(x, y, y_continuous=self.is_y_continuous)
        )
