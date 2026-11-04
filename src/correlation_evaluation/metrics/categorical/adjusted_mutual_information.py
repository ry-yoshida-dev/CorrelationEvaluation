from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind
from .entropy_average import EntropyAverage


@dataclass(frozen=True)
class AdjustedMutualInformation(CorrelationMetric):
    """
    Normalized mutual information corrected for chance, near 0 under independence.

    Attributes
    ----------
    average : EntropyAverage
        The average of the two entropies.
    """

    average: EntropyAverage = EntropyAverage.ARITHMETIC

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    is_unit_bounded: ClassVar[bool] = False

    def __post_init__(self) -> None:
        ParameterValidator.require_members(EntropyAverage, average=self.average)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult(
            statistic=float(
                adjusted_mutual_info_score(
                    x.astype(np.int64),
                    y.astype(np.int64),
                    average_method=self.average.option_name,
                )
            )
        )
