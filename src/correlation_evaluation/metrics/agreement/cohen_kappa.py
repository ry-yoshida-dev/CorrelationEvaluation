from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from sklearn.metrics import cohen_kappa_score

from ...metric import CorrelationMetric
from ...options import KappaWeighting
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class CohenKappa(CorrelationMetric):
    """
    Cohen's kappa, the agreement of two raters corrected for chance.

    One constant rater has a value here, which is 0 when the other rater varies.

    Attributes
    ----------
    weighting : KappaWeighting
        Penalty for disagreements.
    """

    weighting: KappaWeighting = KappaWeighting.UNWEIGHTED

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    is_defined_for_constant: ClassVar[bool] = True
    is_label_set_shared: ClassVar[bool] = True

    def __post_init__(self) -> None:
        ParameterValidator.require_members(KappaWeighting, weighting=self.weighting)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        if np.ptp(np.concatenate([x, y])) == 0:
            return CorrelationResult(statistic=None)
        return CorrelationResult(
            statistic=float(
                cohen_kappa_score(
                    x.astype(np.int64),
                    y.astype(np.int64),
                    weights=self.weighting.option_name,
                )
            )
        )
