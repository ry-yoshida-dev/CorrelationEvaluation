from dataclasses import dataclass
from typing import ClassVar, override

from ...types import FloatArray
from .covariate_adjusted import CovariateAdjusted


@dataclass(frozen=True, eq=False)
class Semipartial(CovariateAdjusted):
    """
    Semipartial (part) correlation, with the covariates removed from y only.

    Its p-value equals that of `Partial`, since both test the same hypothesis; the
    t statistic of this smaller correlation would be too conservative when x
    relates to the covariates.
    """

    is_symmetric: ClassVar[bool] = False

    @override
    def _adjust_x(self, x: FloatArray, residual_x: FloatArray) -> FloatArray:
        return x
