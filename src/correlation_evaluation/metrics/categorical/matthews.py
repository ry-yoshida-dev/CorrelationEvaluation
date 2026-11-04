from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from sklearn.metrics import matthews_corrcoef

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class Matthews(CorrelationMetric):
    """
    Matthews correlation coefficient between true and predicted classes.

    Generalizes the phi coefficient to more than two classes.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    is_label_set_shared: ClassVar[bool] = True

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult(
            statistic=float(matthews_corrcoef(x.astype(np.int64), y.astype(np.int64)))
        )
