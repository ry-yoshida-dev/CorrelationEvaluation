import sys
from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class Canonical(CorrelationMetric):
    """
    First canonical correlation, the largest correlation between linear combinations
    of the features of x and of y.

    Undefined unless there are more samples than features in x and y combined,
    because any data then fits perfectly.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE
    y_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        if x.shape[0] <= x.shape[1] + y.shape[1]:
            return CorrelationResult(statistic=None)
        singular_values: FloatArray = np.linalg.svd(
            self._orthonormal_basis(x).T @ self._orthonormal_basis(y),
            compute_uv=False,
        )
        return CorrelationResult(statistic=float(np.max(singular_values)))

    @staticmethod
    def _orthonormal_basis(values: FloatArray) -> FloatArray:
        left: FloatArray
        singular_values: FloatArray
        left, singular_values, _ = np.linalg.svd(
            values - values.mean(axis=0), full_matrices=False
        )
        tolerance: float = (
            float(np.max(singular_values))
            * max(np.size(values, axis=0), np.size(values, axis=1))
            * sys.float_info.epsilon
        )
        return left[:, singular_values > tolerance]
