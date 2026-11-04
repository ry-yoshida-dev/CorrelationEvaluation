import numpy as np
from scipy import stats

from ..options import Alternative
from ..types import BoolArray, FloatArray


class ColumnCorrelation:
    """
    Pearson correlation of every pair of columns at once.
    """

    @staticmethod
    def statistics(columns: FloatArray) -> FloatArray:
        """
        Compute the correlation of every pair of columns.

        Parameters
        ----------
        columns : FloatArray
            The variables with shape (n_samples, n_columns).

        Returns
        -------
        FloatArray
            The correlations with shape (n_columns, n_columns); NaN for constant
            columns.
        """
        # Scaling by powers of two is exact, and keeps extreme values from overflowing.
        exponents = np.frexp(np.max(np.abs(columns), axis=0))[1]
        scaled: FloatArray = np.ldexp(columns, -exponents)
        centered: FloatArray = scaled - scaled.mean(axis=0)
        norms: FloatArray = np.linalg.norm(centered, axis=0)
        is_varying: BoolArray = np.ptp(scaled, axis=0) > 0
        normalized: FloatArray = centered / np.where(is_varying, norms, 1.0)
        correlations: FloatArray = np.clip(normalized.T @ normalized, -1.0, 1.0)
        correlations[~is_varying, :] = np.nan
        correlations[:, ~is_varying] = np.nan
        return correlations

    @staticmethod
    def p_values(
        correlations: FloatArray, num_samples: int, alternative: Alternative
    ) -> FloatArray:
        """
        Test correlations against the exact null distribution for normal data.

        Parameters
        ----------
        correlations : FloatArray
            The correlations.
        num_samples : int
            The number of samples behind them, at least 3.
        alternative : Alternative
            The alternative hypothesis.

        Returns
        -------
        FloatArray
            The p-values; NaN where the correlation is NaN.
        """
        shape: float = num_samples / 2 - 1
        match alternative:
            case Alternative.TWO_SIDED:
                return np.minimum(
                    2 * stats.beta.sf(np.abs(correlations), shape, shape, -1, 2), 1.0
                )
            case Alternative.GREATER:
                return stats.beta.sf(correlations, shape, shape, -1, 2)
            case Alternative.LESS:
                return stats.beta.cdf(correlations, shape, shape, -1, 2)
