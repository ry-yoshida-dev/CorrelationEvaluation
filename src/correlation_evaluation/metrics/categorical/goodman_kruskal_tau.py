from dataclasses import dataclass
from typing import ClassVar, override

from ...contingency_table import ContingencyTable
from ...metric import CorrelationMetric
from ...result import CorrelationResult
from ...types import FloatArray, IntArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class GoodmanKruskalTau(CorrelationMetric):
    """
    Goodman and Kruskal's tau of y given x.

    The share of errors in predicting y at random by its class frequencies that
    knowing x removes.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    is_symmetric: ClassVar[bool] = False

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        counts: IntArray = ContingencyTable.from_codes(x, y).counts
        n: int = int(counts.sum())
        marginal_error: float = 1 - float(((counts.sum(axis=0) / n) ** 2).sum())
        conditional_error: float = (
            1 - float((counts**2 / counts.sum(axis=1, keepdims=True)).sum()) / n
        )
        return CorrelationResult(
            statistic=(marginal_error - conditional_error) / marginal_error
        )
