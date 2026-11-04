from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    """
    Confidence interval of a correlation statistic.

    Attributes
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound.
    level : float
        Confidence level in (0, 1).
    """

    low: float
    high: float
    level: float
