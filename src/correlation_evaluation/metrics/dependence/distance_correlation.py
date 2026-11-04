import math
from dataclasses import dataclass
from typing import ClassVar, override

from ...computation import PairwiseDistance
from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class DistanceCorrelation(CorrelationMetric):
    """
    Distance correlation, zero only under independence for finite-moment variables.

    Takes O(n^2) time and memory.

    Attributes
    ----------
    is_bias_corrected : bool
        Whether to return the unbiased estimator of the squared distance
        correlation, which is near 0 under independence even for few samples and
        can be negative. Otherwise returns the non-squared, biased estimator.
    """

    is_bias_corrected: bool = False

    x_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE
    y_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE
    min_samples: ClassVar[int] = 4

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        distances_x: FloatArray = PairwiseDistance.euclidean(x)
        distances_y: FloatArray = PairwiseDistance.euclidean(y)
        if self.is_bias_corrected:
            centered_x: FloatArray = PairwiseDistance.u_center(distances_x)
            centered_y: FloatArray = PairwiseDistance.u_center(distances_y)
        else:
            centered_x = PairwiseDistance.double_center(distances_x)
            centered_y = PairwiseDistance.double_center(distances_y)
        denominator: float = math.sqrt(
            float((centered_x**2).sum()) * float((centered_y**2).sum())
        )
        if denominator == 0:
            return CorrelationResult(statistic=None)
        squared: float = float((centered_x * centered_y).sum()) / denominator
        return CorrelationResult(
            statistic=squared if self.is_bias_corrected else math.sqrt(max(squared, 0))
        )
