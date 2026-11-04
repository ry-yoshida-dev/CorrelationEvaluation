from dataclasses import dataclass
from typing import override

from ...types import FloatArray
from .covariate_adjusted import CovariateAdjusted


@dataclass(frozen=True, eq=False)
class Partial(CovariateAdjusted):
    """
    Partial correlation, with the covariates removed from both x and y.
    """

    @override
    def _adjust_x(self, x: FloatArray, residual_x: FloatArray) -> FloatArray:
        return residual_x
