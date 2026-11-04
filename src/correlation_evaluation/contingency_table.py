from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

from .types import FloatArray, IntArray


@dataclass(frozen=True, eq=False)
class ContingencyTable:
    """
    Joint counts of two categorical variables.

    Compares by identity, because arrays have no single truth value for equality.

    Attributes
    ----------
    counts : IntArray
        The counts with shape (n_x_levels, n_y_levels).
    """

    counts: IntArray

    @classmethod
    def from_codes(cls, x: FloatArray, y: FloatArray) -> ContingencyTable:
        """
        Count the joint occurrences of two encoded variables.

        Parameters
        ----------
        x : FloatArray
            The encoded row variable.
        y : FloatArray
            The encoded column variable.

        Returns
        -------
        ContingencyTable
            The table.
        """
        return cls(counts=stats.contingency.crosstab(x, y).count)

    def chi_square_p_value(self) -> float:
        """
        Test independence with Pearson's chi-square, without continuity correction.

        Returns
        -------
        float
            The p-value.
        """
        return float(stats.chi2_contingency(self.counts, correction=False).pvalue)
