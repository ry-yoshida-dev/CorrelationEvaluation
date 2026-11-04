import sys

import numpy as np

from ..types import BoolArray, FloatArray


class ColumnSpace:
    """
    Span of the columns of a data matrix, independent of their units.
    """

    @staticmethod
    def orthonormal_basis(values: FloatArray) -> FloatArray:
        """
        Compute an orthonormal basis of the centered columns.

        Columns are standardized first. A rank decision compares singular values,
        so a column with a small spread relative to its offset or to another column
        would otherwise count as collinear and silently vanish from the span.

        Parameters
        ----------
        values : FloatArray
            The samples with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The basis with shape (n_samples, rank); constant columns add nothing,
            so the rank is 0 when all columns are constant.
        """
        magnitudes: FloatArray = np.max(np.abs(values), axis=0)
        bounded: FloatArray = values / np.where(magnitudes > 0, magnitudes, 1.0)
        spreads: FloatArray = np.ptp(bounded, axis=0)
        # Differences of a few units in the last place are rounding, not variation.
        is_varying: BoolArray = spreads > 4 * sys.float_info.epsilon
        standardized: FloatArray = np.where(
            is_varying,
            (bounded - bounded.mean(axis=0)) / np.where(is_varying, spreads, 1.0),
            0.0,
        )
        left: FloatArray
        singular_values: FloatArray
        left, singular_values, _ = np.linalg.svd(standardized, full_matrices=False)
        tolerance: float = (
            float(np.max(singular_values))
            * max(np.size(values, axis=0), np.size(values, axis=1))
            * sys.float_info.epsilon
        )
        return left[:, singular_values > tolerance]
