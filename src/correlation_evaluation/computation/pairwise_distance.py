import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..types import FloatArray


class PairwiseDistance:
    """
    Pairwise distance operations shared by distance and kernel metrics.
    """

    @staticmethod
    def euclidean(values: FloatArray) -> FloatArray:
        """
        Compute the Euclidean distance between all pairs of samples.

        Parameters
        ----------
        values : FloatArray
            The samples with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            The distances with shape (n_samples, n_samples).
        """
        return squareform(pdist(values, metric="euclidean"))

    @staticmethod
    def double_center(matrix: FloatArray) -> FloatArray:
        """
        Subtract row and column means, and add back the grand mean.

        Parameters
        ----------
        matrix : FloatArray
            The square matrix.

        Returns
        -------
        FloatArray
            The centered matrix.
        """
        return (
            matrix
            - matrix.mean(axis=0, keepdims=True)
            - matrix.mean(axis=1, keepdims=True)
            + matrix.mean()
        )

    @staticmethod
    def u_center(matrix: FloatArray) -> FloatArray:
        """
        Center a matrix for unbiased estimation, with a zero diagonal.

        Parameters
        ----------
        matrix : FloatArray
            The square matrix with shape (n, n), where n is at least 4.

        Returns
        -------
        FloatArray
            The U-centered matrix.
        """
        n: int = np.size(matrix, axis=0)
        centered: FloatArray = (
            matrix
            - matrix.sum(axis=0, keepdims=True) / (n - 2)
            - matrix.sum(axis=1, keepdims=True) / (n - 2)
            + matrix.sum() / ((n - 1) * (n - 2))
        )
        np.fill_diagonal(centered, 0.0)
        return centered
