from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray, IntArray


@dataclass(frozen=True)
class ChatterjeeXi(CorrelationMetric):
    """
    Chatterjee's xi, near 1 when y is a noiseless function of x and near 0 under
    independence.

    Ties in x are broken at random, as the estimator defines, so with ties the value
    depends on the seed but not on the order of the samples.

    Attributes
    ----------
    is_y_continuous : bool
        Whether y is assumed to have no ties, which simplifies the p-value.
    random_state : int | None
        Seed of the tie breaking; fresh randomness for each evaluation when None.
    """

    is_y_continuous: bool = False
    random_state: int | None = 0

    is_symmetric: ClassVar[bool] = False

    def __post_init__(self) -> None:
        ParameterValidator.require_booleans(is_y_continuous=self.is_y_continuous)
        ParameterValidator.require_seeds(random_state=self.random_state)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        # SciPy keeps tied x in the given order. Shuffling from a canonical order,
        # not the given one, makes a seeded result independent of the sample order.
        canonical: IntArray = np.lexsort((y, x))
        order: IntArray = canonical[
            np.random.default_rng(self.random_state).permutation(x.size)
        ]
        return CorrelationResult.from_test_outcome(
            stats.chatterjeexi(x[order], y[order], y_continuous=self.is_y_continuous)
        )
