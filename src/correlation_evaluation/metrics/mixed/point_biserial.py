from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class PointBiserial(CorrelationMetric):
    """
    Point-biserial correlation of a binary x with a numeric y.

    Positive when the larger label of x goes with larger y.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.BINARY

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult.from_test_outcome(
            stats.pointbiserialr(x.astype(np.int64), y)
        )
