from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...options import Alternative
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class LaggedPearson(CorrelationMetric):
    """
    Pearson correlation between x at time t and y at time t + lag.

    Attributes
    ----------
    lag : int
        Time steps by which y follows x; negative when y leads.
    alternative : Alternative
        Alternative hypothesis of the test.
    """

    lag: int = 1
    alternative: Alternative = Alternative.TWO_SIDED

    is_symmetric: ClassVar[bool] = False

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)
        ParameterValidator.require_integers(lag=self.lag)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        overlap: int = x.size - abs(self.lag)
        if overlap < self.min_samples:
            return CorrelationResult(statistic=None)
        leading: FloatArray = x[:overlap] if self.lag >= 0 else x[-overlap:]
        following: FloatArray = y[-overlap:] if self.lag >= 0 else y[:overlap]
        if np.ptp(leading) == 0 or np.ptp(following) == 0:
            return CorrelationResult(statistic=None)
        return CorrelationResult.from_test_outcome(
            stats.pearsonr(leading, following, alternative=self.alternative.option_name)
        )
