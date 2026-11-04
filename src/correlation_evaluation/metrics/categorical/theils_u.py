from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray, IntArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class TheilsU(CorrelationMetric):
    """
    Theil's U (uncertainty coefficient) of y given x.

    The fraction of the entropy of y that knowing x removes.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    is_symmetric: ClassVar[bool] = False

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        codes_x: IntArray = x.astype(np.int64)
        codes_y: IntArray = y.astype(np.int64)
        return CorrelationResult(
            statistic=float(mutual_info_score(codes_x, codes_y))
            / float(stats.entropy(np.bincount(codes_y)))
        )
