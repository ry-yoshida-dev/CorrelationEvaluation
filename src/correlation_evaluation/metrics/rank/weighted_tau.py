from dataclasses import dataclass
from typing import override

from scipy import stats

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class WeightedTau(CorrelationMetric):
    """
    Kendall's tau with hyperbolic weights that emphasize top-ranked items.

    Averages over ranking by decreasing x and by decreasing y, so it is symmetric.

    Attributes
    ----------
    is_additive : bool
        Whether a pair's weight adds the weights of its elements rather than
        multiplies them.
    """

    is_additive: bool = True

    def __post_init__(self) -> None:
        ParameterValidator.require_booleans(is_additive=self.is_additive)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult(
            statistic=float(
                stats.weightedtau(x, y, additive=self.is_additive).statistic
            )
        )
