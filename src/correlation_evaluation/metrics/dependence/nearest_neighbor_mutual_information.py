import math
from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray, IntArray


@dataclass(frozen=True)
class NearestNeighborMutualInformation(CorrelationMetric):
    """
    Mutual information between continuous variables, by nearest neighbor estimation.

    The estimator assumes no ties and breaks them with small random noise, so with
    ties the value depends on the seed; use `MutualInformation` for discrete data. It
    depends on neither the order of the samples nor that of the variables.

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

    is_unit_bounded: ClassVar[bool] = False

    def __post_init__(self) -> None:
        ParameterValidator.require_booleans(
            is_correlation_scaled=self.is_correlation_scaled
        )
        ParameterValidator.require_integers(num_neighbors=self.num_neighbors)
        ParameterValidator.require_seeds(random_state=self.random_state)
        if self.num_neighbors <= 0:
            raise ValueError("num_neighbors must be greater than 0")

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        if x.size <= self.num_neighbors:
            return CorrelationResult(statistic=None)
        first: FloatArray
        second: FloatArray
        first, second = self._canonical_pair(x, y)
        per_feature: FloatArray = mutual_info_regression(
            first.reshape(-1, 1),
            second,
            n_neighbors=self.num_neighbors,
            random_state=self.random_state,
        )
        nats: float = float(np.max(per_feature))
        return CorrelationResult(
            statistic=math.sqrt(1 - math.exp(-2 * nats))
            if self.is_correlation_scaled
            else nats
        )

    @staticmethod
    def _canonical_pair(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
        # The noise follows the position of a value, so a fixed arrangement of the
        # samples and variables keeps the noise from depending on the given order.
        forward: FloatArray = np.column_stack([x, y])[np.lexsort((y, x))]
        backward: FloatArray = np.column_stack([y, x])[np.lexsort((x, y))]
        differences: IntArray = np.flatnonzero(np.not_equal(forward, backward))
        chosen: FloatArray = forward
        if differences.size > 0:
            index: int = int(np.take(differences, 0))
            if float(np.take(backward, index)) < float(np.take(forward, index)):
                chosen = backward
        return chosen[:, 0], chosen[:, 1]
