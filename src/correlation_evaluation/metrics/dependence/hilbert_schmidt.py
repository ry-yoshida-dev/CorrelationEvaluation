import math
from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np

from ...computation import PairwiseDistance
from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class HilbertSchmidt(CorrelationMetric):
    """
    Normalized Hilbert-Schmidt independence criterion with Gaussian kernels.

    Ranges from 0 to 1. Takes O(n^2) time and memory.

    Attributes
    ----------
    bandwidth : float | None
        Kernel bandwidth; the median positive distance of each variable when None.
    """

    bandwidth: float | None = None

    x_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE
    y_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE

    def __post_init__(self) -> None:
        if self.bandwidth is None:
            return
        ParameterValidator.require_real_numbers(bandwidth=self.bandwidth)
        if not 0 < self.bandwidth < math.inf:
            raise ValueError("bandwidth must be positive and finite when set")

    @override
    def _with_safe_magnitude(
        self, x: FloatArray, y: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        # A fixed bandwidth is in the units of the data, unlike the median distance.
        return (
            (x, y) if self.bandwidth is not None else super()._with_safe_magnitude(x, y)
        )

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        kernel_x: FloatArray = self._centered_kernel(x)
        kernel_y: FloatArray = self._centered_kernel(y)
        denominator: float = math.sqrt(
            float((kernel_x**2).sum()) * float((kernel_y**2).sum())
        )
        if denominator == 0:
            return CorrelationResult(statistic=None)
        return CorrelationResult(
            statistic=float((kernel_x * kernel_y).sum()) / denominator
        )

    def _centered_kernel(self, values: FloatArray) -> FloatArray:
        distances: FloatArray = PairwiseDistance.euclidean(values)
        bandwidth: float = (
            float(np.median(distances[distances > 0]))
            if self.bandwidth is None
            else self.bandwidth
        )
        # Distances far beyond the bandwidth have a kernel value of 0 in the limit.
        with np.errstate(over="ignore"):
            return PairwiseDistance.double_center(
                np.exp(-0.5 * (distances / bandwidth) ** 2)
            )
