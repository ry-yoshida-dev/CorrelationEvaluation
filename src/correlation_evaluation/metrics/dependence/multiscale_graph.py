from dataclasses import dataclass
from typing import ClassVar, override

from scipy import stats

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class MultiscaleGraph(CorrelationMetric):
    """
    Multiscale graph correlation, a distance correlation over local neighborhoods.

    The p-value comes from a permutation test, so this is far slower than other
    metrics.

    Attributes
    ----------
    num_permutations : int
        Permutations of the test; the backing library warns below 1000.
    random_state : int | None
        Seed of the permutations.
    """

    num_permutations: int = 1000
    random_state: int | None = 0

    x_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE
    y_kind: ClassVar[VariableKind] = VariableKind.MULTIVARIATE
    min_samples: ClassVar[int] = 6

    def __post_init__(self) -> None:
        ParameterValidator.require_integers(
            num_permutations=self.num_permutations, random_state=self.random_state
        )
        if self.num_permutations <= 0:
            raise ValueError("num_permutations must be greater than 0")

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult.from_test_outcome(
            stats.multiscale_graphcorr(
                x, y, reps=self.num_permutations, random_state=self.random_state
            )
        )
