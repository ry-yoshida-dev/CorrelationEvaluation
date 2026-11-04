from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from data_scaler import ScalingKind

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...scaling_restriction import ScalingRestriction
from ...types import FloatArray


@dataclass(frozen=True)
class Cosine(CorrelationMetric):
    """
    Cosine similarity, a Pearson correlation that does not center the variables.

    Constant variables have a value here; only an all-zero variable leaves it
    undefined.
    """

    min_samples: ClassVar[int] = 1
    is_defined_for_constant: ClassVar[bool] = True
    scaling_restriction: ClassVar[ScalingRestriction | None] = ScalingRestriction(
        allowed_kinds=frozenset({ScalingKind.IDENTITY, ScalingKind.PROPORTIONAL}),
        reason="anything but a constant factor would change its value; centering, for one, makes it a Pearson correlation",
    )

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        denominator: float = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denominator == 0:
            return CorrelationResult(statistic=None)
        return CorrelationResult(statistic=float(x @ y) / denominator)
