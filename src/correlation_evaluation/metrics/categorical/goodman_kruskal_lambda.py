from dataclasses import dataclass
from typing import ClassVar, override

from ...contingency_table import ContingencyTable
from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray, IntArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class GoodmanKruskalLambda(CorrelationMetric):
    """
    Goodman and Kruskal's lambda of y given x.

    The share of errors in predicting the most frequent class of y that knowing x
    removes. It is 0 whenever x never changes the most frequent class, even if the
    variables are dependent.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    is_symmetric: ClassVar[bool] = False

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        counts: IntArray = ContingencyTable.from_codes(x, y).counts
        most_frequent: int = int(counts.sum(axis=0).max())
        return CorrelationResult(
            statistic=(int(counts.max(axis=1).sum()) - most_frequent)
            / (int(counts.sum()) - most_frequent)
        )
