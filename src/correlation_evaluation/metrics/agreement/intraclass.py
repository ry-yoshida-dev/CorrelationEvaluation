from dataclasses import dataclass
from typing import ClassVar, override

import numpy as np
from data_scaler import ScalingKind

from ...metric import CorrelationMetric
from ...parameter_validator import ParameterValidator
from ...result import CorrelationResult
from ...types import FloatArray
from .intraclass_form import IntraclassForm


@dataclass(frozen=True)
class Intraclass(CorrelationMetric):
    """
    Single-measure intraclass correlation of two raters, computed from two-way
    analysis of variance. x and y hold the ratings of the same subjects.

    Attributes
    ----------
    form : IntraclassForm
        The study design.
    """

    form: IntraclassForm = IntraclassForm.TWO_WAY_RANDOM

    is_unit_bounded: ClassVar[bool] = False
    allowed_scaling_kinds: ClassVar[frozenset[ScalingKind]] = frozenset(
        {ScalingKind.IDENTITY}
    )
    scaling_restriction: ClassVar[str] = (
        "scaling x and y separately would hide the differences it detects"
    )

    def __post_init__(self) -> None:
        ParameterValidator.require_members(IntraclassForm, form=self.form)

    @override
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        ratings: FloatArray = np.column_stack([x, y])
        num_subjects: int = x.size
        grand_mean: float = float(ratings.mean())
        subject_sum: float = float(2 * ((ratings.mean(axis=1) - grand_mean) ** 2).sum())
        rater_sum: float = float(
            num_subjects * ((ratings.mean(axis=0) - grand_mean) ** 2).sum()
        )
        total_sum: float = float(((ratings - grand_mean) ** 2).sum())
        subject_square: float = subject_sum / (num_subjects - 1)
        within_square: float = (total_sum - subject_sum) / num_subjects
        rater_square: float = rater_sum
        error_square: float = (total_sum - subject_sum - rater_sum) / (num_subjects - 1)
        numerator: float
        denominator: float
        match self.form:
            case IntraclassForm.ONE_WAY_RANDOM:
                numerator = subject_square - within_square
                denominator = subject_square + within_square
            case IntraclassForm.TWO_WAY_RANDOM:
                numerator = subject_square - error_square
                denominator = (
                    subject_square
                    + error_square
                    + 2 * (rater_square - error_square) / num_subjects
                )
            case IntraclassForm.TWO_WAY_MIXED:
                numerator = subject_square - error_square
                denominator = subject_square + error_square
        if denominator == 0:
            return CorrelationResult(statistic=None)
        # Only the upper bound holds by definition; few samples can go below -1.
        return CorrelationResult(statistic=min(numerator / denominator, 1.0))
