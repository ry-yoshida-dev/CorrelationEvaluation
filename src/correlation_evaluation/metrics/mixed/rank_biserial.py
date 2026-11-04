from dataclasses import dataclass
from typing import ClassVar, override

from scipy import stats

from ...alternative import Alternative
from ...hypothesis_outcome import HypothesisOutcome
from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import BoolArray, FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class RankBiserial(CorrelationMetric):
    """
    Rank-biserial correlation of a binary x with a numeric or ordinal y.

    The effect size of the Mann-Whitney U test: the probability that y is larger
    in the group with the larger label of x, minus the probability that it is
    smaller.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis of the test.
    """

    alternative: Alternative = Alternative.TWO_SIDED

    x_kind: ClassVar[VariableKind] = VariableKind.BINARY

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        is_upper: BoolArray = x > 0
        upper: FloatArray = y[is_upper]
        lower: FloatArray = y[~is_upper]
        outcome: HypothesisOutcome = stats.mannwhitneyu(
            upper, lower, alternative=self.alternative.option_name
        )
        return CorrelationResult(
            statistic=2 * float(outcome.statistic) / (upper.size * lower.size) - 1,
            p_value=float(outcome.pvalue),
        )
