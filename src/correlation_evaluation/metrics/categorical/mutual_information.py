from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from sklearn.metrics import mutual_info_score

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class MutualInformation(CorrelationMetric):
    """
    Mutual information between categorical variables, in nats.

    Zero when a variable is constant, which is meaningful here.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    min_samples: ClassVar[int] = 1
    is_unit_bounded: ClassVar[bool] = False
    is_defined_for_constant: ClassVar[bool] = True

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult(
            statistic=float(mutual_info_score(x.astype(np.int64), y.astype(np.int64)))
        )
