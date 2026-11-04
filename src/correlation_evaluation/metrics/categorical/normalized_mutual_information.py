from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from ...metric import CorrelationMetric
from ...options import EntropyAverage
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class NormalizedMutualInformation(CorrelationMetric):
    """
    Mutual information divided by an average entropy, ranging from 0 to 1.

    Attributes
    ----------
    average : EntropyAverage
        The average of the two entropies.
    """

    average: EntropyAverage = EntropyAverage.ARITHMETIC

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL

    def __post_init__(self) -> None:
        ParameterValidator.require_members(EntropyAverage, average=self.average)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult(
            statistic=float(
                normalized_mutual_info_score(
                    x.astype(np.int64),
                    y.astype(np.int64),
                    average_method=self.average.option_name,
                )
            )
        )
