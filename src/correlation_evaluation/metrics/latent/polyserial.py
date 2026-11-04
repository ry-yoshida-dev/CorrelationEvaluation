from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray, IntArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class Polyserial(CorrelationMetric):
    """
    Polyserial correlation of an ordinal x with a numeric y.

    Estimates the correlation of y with a latent normal variable that was cut at
    thresholds, by Olsson's closed form, which treats levels as equally spaced.
    Being an estimate rather than a true correlation, it can exceed 1 in magnitude
    for strongly non-normal data. Labels are ordered by their sort order.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    is_unit_bounded: ClassVar[bool] = False

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        counts: IntArray = np.bincount(x.astype(np.int64))
        thresholds: FloatArray = stats.norm.ppf(np.cumsum(counts)[:-1] / x.size)
        return CorrelationResult(
            statistic=float(stats.pearsonr(x, y).statistic)
            * float(x.std())
            / float(stats.norm.pdf(thresholds).sum())
        )
