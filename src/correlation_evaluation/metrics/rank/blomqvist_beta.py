from dataclasses import dataclass
from typing import override

import numpy as np

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class BlomqvistBeta(CorrelationMetric):
    """
    Blomqvist's beta (quadrant correlation), the agreement of signs around medians.

    Samples on a median are excluded.
    """

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        signs: FloatArray = np.sign(x - np.median(x)) * np.sign(y - np.median(y))
        num_counted: int = int(np.count_nonzero(signs))
        if num_counted == 0:
            return CorrelationResult(statistic=None)
        return CorrelationResult(statistic=float(signs.sum()) / num_counted)
