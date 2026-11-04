from dataclasses import dataclass

from .types import FloatArray


@dataclass(frozen=True, eq=False)
class CorrelationMatrix:
    """
    Pairwise outcomes of a metric over the columns of a matrix.

    Compares by identity, because arrays have no single truth value for equality.

    Attributes
    ----------
    statistics : FloatArray
        The metric values with shape (n_columns, n_columns); NaN when undefined.
        Entry (i, j) evaluates column i as x and column j as y, which matters
        for asymmetric metrics.
    p_values : FloatArray
        The p-values with the same shape; NaN when unavailable.
    """

    statistics: FloatArray
    p_values: FloatArray
