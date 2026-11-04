from dataclasses import dataclass
from typing import ClassVar, override

from scipy import stats

from ...contingency_table import ContingencyTable
from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class Phi(CorrelationMetric):
    """
    Phi coefficient of two binary variables, signed by their label order.

    The p-value comes from the chi-square test of independence.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.BINARY
    y_kind: ClassVar[VariableKind] = VariableKind.BINARY

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        return CorrelationResult(
            statistic=float(stats.pearsonr(x, y).statistic),
            p_value=ContingencyTable.from_codes(x, y).chi_square_p_value(),
        )
