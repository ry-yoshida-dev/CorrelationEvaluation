from dataclasses import dataclass
from typing import ClassVar, override

from scipy import stats

from ...contingency_table import ContingencyTable
from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class TschuprowT(CorrelationMetric):
    """
    Tschuprow's T, a chi-square association that reaches 1 only for square tables.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        table: ContingencyTable = ContingencyTable.from_codes(x, y)
        return CorrelationResult(
            statistic=float(
                stats.contingency.association(table.counts, method="tschuprow")
            ),
            p_value=table.chi_square_p_value(),
        )
