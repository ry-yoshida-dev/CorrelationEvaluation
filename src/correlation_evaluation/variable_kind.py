from enum import Enum

import numpy as np
from data_scaler import ScalingKind
from numpy.typing import ArrayLike, NDArray

from .types import FloatArray


class VariableKind(Enum):
    """
    Measurement scale a metric expects of a variable.

    Attributes
    ----------
    NUMERIC
        Finite numbers with shape (n_samples,).
    MULTIVARIATE
        Finite numbers with shape (n_samples,) or (n_samples, n_features).
    BINARY
        Labels with at most two levels.
    CATEGORICAL
        Labels of any number of levels; ordinal metrics rely on their sort order.
    """

    NUMERIC = "numeric"
    MULTIVARIATE = "multivariate"
    BINARY = "binary"
    CATEGORICAL = "categorical"

    @property
    def is_numeric(self) -> bool:
        """
        Get whether this kind holds numbers rather than labels.

        Returns
        -------
        bool
            True for NUMERIC and MULTIVARIATE.
        """
        match self:
            case VariableKind.NUMERIC | VariableKind.MULTIVARIATE:
                return True
            case VariableKind.BINARY | VariableKind.CATEGORICAL:
                return False

    @property
    def supported_scaling_kinds(self) -> frozenset[ScalingKind]:
        """
        Get the kinds of scaling that keep a variable of this kind meaningful.

        Returns
        -------
        frozenset[ScalingKind]
            All kinds, except that sample-wise scaling would leave only the sign of
            each NUMERIC value. Labels are not scaled, so any kind leaves them as
            they are.
        """
        match self:
            case VariableKind.NUMERIC:
                return frozenset(ScalingKind) - {ScalingKind.SAMPLE_WISE}
            case (
                VariableKind.MULTIVARIATE
                | VariableKind.BINARY
                | VariableKind.CATEGORICAL
            ):
                return frozenset(ScalingKind)

    def encode(self, values: ArrayLike, name: str) -> FloatArray:
        """
        Validate a variable and convert it to floats.

        Labels become consecutive integers in sorted order, so that every metric
        receives the same array type.

        Parameters
        ----------
        values : ArrayLike
            The variable.
        name : str
            The variable name for error messages.

        Returns
        -------
        FloatArray
            The encoded variable; two-dimensional for MULTIVARIATE.

        Raises
        ------
        TypeError
            If the values have the wrong data type for this kind.
        ValueError
            If the values are empty, have the wrong shape, are not finite, or
            have too many levels.
        """
        array: NDArray[np.generic] = np.asarray(values)
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        match self:
            case VariableKind.NUMERIC:
                return self._encode_numbers(array, name, max_ndim=1)
            case VariableKind.MULTIVARIATE:
                encoded: FloatArray = self._encode_numbers(array, name, max_ndim=2)
                return encoded.reshape(len(encoded), -1)
            case VariableKind.BINARY:
                return self._encode_labels(array, name, max_levels=2)
            case VariableKind.CATEGORICAL:
                return self._encode_labels(array, name, max_levels=None)

    @staticmethod
    def _encode_numbers(
        array: NDArray[np.generic], name: str, max_ndim: int
    ) -> FloatArray:
        if not np.issubdtype(array.dtype, np.integer) and not np.issubdtype(
            array.dtype, np.floating
        ):
            raise TypeError(f"{name} must contain numbers, got {array.dtype}")
        if not 1 <= array.ndim <= max_ndim:
            raise ValueError(f"{name} has unsupported shape {array.shape}")
        numbers: FloatArray = array.astype(np.float64)
        if not np.isfinite(numbers).all():
            raise ValueError(f"{name} must be finite")
        return numbers

    @staticmethod
    def _encode_labels(
        array: NDArray[np.generic], name: str, max_levels: int | None
    ) -> FloatArray:
        if array.dtype == np.object_:
            # Mixed types would stringify missing values into labels like "nan".
            labels: list[object] = list(array)
            if len({type(label) for label in labels}) > 1:
                raise TypeError(f"{name} must contain labels of one type")
            array = np.asarray(labels)
        if array.dtype.kind not in "biufUS":
            raise TypeError(f"{name} must contain labels, got {array.dtype}")
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
        if array.dtype.kind == "f" and not np.isfinite(array.astype(np.float64)).all():
            raise ValueError(f"{name} must be finite")
        levels: NDArray[np.generic]
        codes: NDArray[np.intp]
        levels, codes = np.unique(array, return_inverse=True)
        if max_levels is not None and levels.size > max_levels:
            raise ValueError(
                f"{name} must have at most {max_levels} levels, got {levels.size}"
            )
        return codes.astype(np.float64)
