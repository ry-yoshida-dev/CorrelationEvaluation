from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from .confidence_interval import ConfidenceInterval
from .hypothesis_outcome import HypothesisOutcome


@dataclass(frozen=True)
class CorrelationResult:
    """
    Outcome of evaluating a metric on one pair of variables.

    NaN from a backing library is stored as None, so undefined values cannot
    silently propagate through later arithmetic. A p-value is at most 1. An undefined
    statistic has neither a p-value nor a confidence interval.

    Attributes
    ----------
    statistic : float | None
        The metric value; None when it is undefined for the data, such as for
        constant input.
    p_value : float | None
        The p-value; None when the metric has no established test.
    confidence_interval : ConfidenceInterval | None
        The confidence interval; None when the metric does not provide one.
    """

    statistic: float | None
    p_value: float | None = None
    confidence_interval: ConfidenceInterval | None = None

    def __post_init__(self) -> None:
        if self.statistic is not None and math.isnan(self.statistic):
            object.__setattr__(self, "statistic", None)
        if self.statistic is None:
            object.__setattr__(self, "p_value", None)
            object.__setattr__(self, "confidence_interval", None)
        elif self.p_value is not None:
            # Backing tests can round a p-value of 1 up past it.
            object.__setattr__(
                self,
                "p_value",
                None if math.isnan(self.p_value) else min(self.p_value, 1.0),
            )

    @classmethod
    def from_test_outcome(cls, outcome: HypothesisOutcome) -> CorrelationResult:
        """
        Build a result from a backing significance test.

        Parameters
        ----------
        outcome : HypothesisOutcome
            The test's statistic and p-value.

        Returns
        -------
        CorrelationResult
            The result.
        """
        return cls(statistic=float(outcome.statistic), p_value=float(outcome.pvalue))

    @classmethod
    def from_t_test(
        cls,
        statistic: float,
        *,
        num_samples: int,
        degrees_of_freedom: int,
        tested_correlation: float | None = None,
    ) -> CorrelationResult:
        """
        Attach a two-sided test with the t statistic r * sqrt((n - 2) / (1 - r^2)).

        Parameters
        ----------
        statistic : float
            The metric value.
        num_samples : int
            The number of samples n in the t statistic.
        degrees_of_freedom : int
            The degrees of freedom of the reference distribution, which robust
            and partial correlations reduce below n - 2.
        tested_correlation : float | None
            The correlation r to test, when it differs from the metric value.

        Returns
        -------
        CorrelationResult
            The result; its p-value is None when no degrees of freedom remain.
        """
        if degrees_of_freedom <= 0 or num_samples <= 2:
            return cls(statistic=statistic)
        correlation: float = (
            statistic if tested_correlation is None else tested_correlation
        )
        if abs(correlation) >= 1:
            return cls(statistic=statistic, p_value=0.0)
        t_statistic: float = abs(correlation) * math.sqrt(
            (num_samples - 2) / (1 - correlation**2)
        )
        return cls(
            statistic=statistic,
            p_value=2 * float(stats.t.sf(t_statistic, degrees_of_freedom)),
        )
