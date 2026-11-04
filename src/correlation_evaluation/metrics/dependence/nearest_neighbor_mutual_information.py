import math
from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray


@dataclass(frozen=True)
class NearestNeighborMutualInformation(CorrelationMetric):
    """
    Mutual information between continuous variables, by nearest neighbor estimation.

    Attributes
    ----------
    num_neighbors : int
        Neighbors of the estimator; more neighbors reduce variance but add bias.
    is_correlation_scaled : bool
        Whether to return Linfoot's sqrt(1 - exp(-2 I)), which ranges from 0 to 1
        and equals |Pearson| for bivariate normal data, instead of nats.
    random_state : int | None
        Seed of the small noise that the estimator adds to break ties.
    """

    num_neighbors: int = 3
    is_correlation_scaled: bool = False
    random_state: int | None = 0

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            num_neighbors=self.num_neighbors, random_state=self.random_state
        )
        if self.num_neighbors <= 0:
            raise ValueError("num_neighbors must be greater than 0")

    is_unit_bounded: ClassVar[bool] = False

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        if x.size <= self.num_neighbors:
            return CorrelationResult(statistic=None)
        per_feature: FloatArray = mutual_info_regression(
            x.reshape(-1, 1),
            y,
            n_neighbors=self.num_neighbors,
            random_state=self.random_state,
        )
        nats: float = float(np.max(per_feature))
        return CorrelationResult(
            statistic=math.sqrt(1 - math.exp(-2 * nats))
            if self.is_correlation_scaled
            else nats
        )
