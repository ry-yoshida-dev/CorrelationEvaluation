from typing import Protocol

from .hypothesis_outcome import HypothesisOutcome


class PearsonOutcome(HypothesisOutcome, Protocol):
    """
    Result of the backing Pearson test, which also provides a confidence interval.
    """

    def confidence_interval(self, confidence_level: float, /) -> tuple[float, float]:
        """
        Compute the Fisher z confidence interval.

        Parameters
        ----------
        confidence_level : float
            The confidence level, in (0, 1).

        Returns
        -------
        tuple[float, float]
            The lower and upper bounds.
        """
        ...
