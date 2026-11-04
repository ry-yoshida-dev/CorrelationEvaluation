from dataclasses import dataclass
from typing import ClassVar, override

from scipy import stats

from ...contingency_table import ContingencyTable
from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class ContingencyCoefficient(CorrelationMetric):
    """
    Pearson's contingency coefficient, whose maximum depends on the table size.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        table: ContingencyTable = ContingencyTable.from_codes(x, y)
        return CorrelationResult(
            statistic=float(
                stats.contingency.association(table.counts, method="pearson")
            ),
            p_value=table.chi_square_p_value(),
        )
