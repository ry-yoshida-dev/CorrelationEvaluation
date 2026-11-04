from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np

from ...computation import ColumnSpace
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
        basis_x: FloatArray = ColumnSpace.orthonormal_basis(x)
        basis_y: FloatArray = ColumnSpace.orthonormal_basis(y)
        if basis_x.size == 0 or basis_y.size == 0:
            return CorrelationResult(statistic=None)
        singular_values: FloatArray = np.linalg.svd(
            basis_x.T @ basis_y, compute_uv=False
        )
        return CorrelationResult(statistic=float(np.max(singular_values)))
