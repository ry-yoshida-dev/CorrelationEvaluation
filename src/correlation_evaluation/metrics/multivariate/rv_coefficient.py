from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class RVCoefficient(CorrelationMetric):
    """
    RV coefficient, a multivariate generalization of the squared Pearson correlation.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE
    y_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        centered_x: FloatArray = x - x.mean(axis=0)
        centered_y: FloatArray = y - y.mean(axis=0)
        cross_norm: float = float(np.linalg.norm(centered_x.T @ centered_y))
        cross: float = cross_norm * cross_norm
        denominator: float = float(
            np.linalg.norm(centered_x.T @ centered_x)
            * np.linalg.norm(centered_y.T @ centered_y)
        )
        if not 0 < denominator < np.inf:
            return CorrelationResult(statistic=None)
        return CorrelationResult(statistic=cross / denominator)
