from dataclasses import dataclass
from typing import override

from scipy import stats

from ...metric import CorrelationMetric
from ...options import Alternative, KendallVariant
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class KendallTau(CorrelationMetric):
    """
    Kendall's rank correlation, based on concordant and discordant pairs.

    Attributes
    ----------
    variant : KendallVariant
        Tie handling.
    alternative : Alternative
        Alternative hypothesis of the test.
    """

    variant: KendallVariant = KendallVariant.B
    alternative: Alternative = Alternative.TWO_SIDED

    def __post_init__(self) -> None:
        ParameterValidator.require_members(KendallVariant, variant=self.variant)
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult.from_test_outcome(
            stats.kendalltau(
                x,
                y,
                variant=self.variant.option_name,
                alternative=self.alternative.option_name,
            )
        )
