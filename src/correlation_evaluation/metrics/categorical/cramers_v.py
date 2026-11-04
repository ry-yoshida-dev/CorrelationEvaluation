import math
from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from scipy import stats

from ...contingency_table import ContingencyTable
from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray, IntArray
from ...variable_kind import VariableKind


@dataclass(frozen=True)
class CramersV(CorrelationMetric):
    """
    Cramér's V, a chi-square association that ranges from 0 to 1 for any table size.

    Attributes
    ----------
    is_bias_corrected : bool
        Whether to apply Bergsma's correction, since the plain estimator is
        positive under independence.
    """

    is_bias_corrected: bool = False

    x_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL
    y_kind: ClassVar[VariableKind] = VariableKind.CATEGORICAL

    def __post_init__(self) -> None:
        ParameterValidator.require_booleans(is_bias_corrected=self.is_bias_corrected)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        table: ContingencyTable = ContingencyTable.from_codes(x, y)
        statistic: float | None = (
            self._bias_corrected(table.counts)
            if self.is_bias_corrected
            else float(stats.contingency.association(table.counts, method="cramer"))
        )
        return CorrelationResult(
            statistic=statistic, p_value=table.chi_square_p_value()
        )

    @staticmethod
    def _bias_corrected(counts: IntArray) -> float | None:
        n: int = int(counts.sum())
        rows: int = np.size(counts, axis=0)
        columns: int = np.size(counts, axis=1)
        chi_square: float = float(
            stats.chi2_contingency(counts, correction=False).statistic
        )
        corrected_phi_square: float = max(
            chi_square / n - (rows - 1) * (columns - 1) / (n - 1), 0.0
        )
        corrected_rows: float = rows - (rows - 1) ** 2 / (n - 1)
        corrected_columns: float = columns - (columns - 1) ** 2 / (n - 1)
        denominator: float = min(corrected_rows - 1, corrected_columns - 1)
        if denominator <= 0:
            return None
        return math.sqrt(corrected_phi_square / denominator)
