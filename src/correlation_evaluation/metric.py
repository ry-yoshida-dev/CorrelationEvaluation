import math
from abc import ABC, abstractmethod
from dataclasses import replace
from itertools import combinations
from typing import ClassVar, Self

import numpy as np
from data_scaler import ScalingKind
from numpy.typing import ArrayLike, NDArray

from .matrix import CorrelationMatrix
from .result import CorrelationResult
from .scaling_restriction import ScalingRestriction
from .types import FloatArray, ScalingMethod
from .variable_kind import VariableKind


class CorrelationMetric(ABC):
    """
    Measure of the association between two variables.

    Concrete metrics are frozen dataclasses of their options. They describe the
    data they accept with class variables and implement only the computation on
    validated data; this class handles validation and undefined cases.

    Attributes
    ----------
    x_kind : VariableKind
        Measurement scale of x.
    y_kind : VariableKind
        Measurement scale of y.
    min_samples : int
        Fewest samples for which the metric is defined.
    is_symmetric : bool
        Whether swapping x and y leaves the value unchanged.
    is_defined_for_constant : bool
        Whether the metric has a meaningful value when a variable is constant.
    is_label_set_shared : bool
        Whether x and y draw from one label set, such as two raters or true and
        predicted classes, so equal labels must receive equal codes.
    is_unit_bounded : bool
        Whether the magnitude of the metric is at most 1 by definition, so that
        rounding beyond it is corrected.
    scaling_restriction : ScalingRestriction | None
        The limit that this metric itself sets on scaling, before the limits of its
        variables; None when it sets none. See `supported_scaling_kinds`.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.NUMERIC
    y_kind: ClassVar[VariableKind] = VariableKind.NUMERIC
    min_samples: ClassVar[int] = 3
    is_symmetric: ClassVar[bool] = True
    is_defined_for_constant: ClassVar[bool] = False
    is_label_set_shared: ClassVar[bool] = False
    is_unit_bounded: ClassVar[bool] = True
    scaling_restriction: ClassVar[ScalingRestriction | None] = None

    @classmethod
    def supported_scaling_kinds(cls) -> frozenset[ScalingKind]:
        """
        Get the kinds of scaling that `evaluate_with_scaling` accepts.

        Returns
        -------
        frozenset[ScalingKind]
            The kinds; empty when this metric takes no numeric variable. Whether
            SAMPLE_WISE applies to a multivariate variable also depends on its data:
            `evaluate_with_scaling` rejects it for a variable with one feature.
        """
        if not cls.x_kind.is_numeric and not cls.y_kind.is_numeric:
            return frozenset()
        allowed: frozenset[ScalingKind] = (
            frozenset(ScalingKind)
            if cls.scaling_restriction is None
            else cls.scaling_restriction.allowed_kinds
        )
        return (
            allowed
            & cls.x_kind.supported_scaling_kinds
            & cls.y_kind.supported_scaling_kinds
        )

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> CorrelationResult:
        """
        Evaluate the metric on one pair of variables.

        Parameters
        ----------
        x : ArrayLike
            The first variable.
        y : ArrayLike
            The second variable, aligned with x.

        Returns
        -------
        CorrelationResult
            The outcome; its statistic is None when the metric is undefined for
            this data.

        Raises
        ------
        TypeError
            If a variable has the wrong data type for this metric.
        ValueError
            If a variable is empty, misaligned, not finite, or has an
            unsupported shape or number of levels.
        """
        encoded_x: FloatArray
        encoded_y: FloatArray
        encoded_x, encoded_y = self._encode_pair(x, y)
        if encoded_x.shape[0] != encoded_y.shape[0]:
            raise ValueError(
                f"x and y must have the same length, got {encoded_x.shape[0]} and {encoded_y.shape[0]}"
            )
        self._validate_num_samples(np.size(encoded_x, axis=0))
        if encoded_x.shape[0] < self.min_samples:
            return CorrelationResult(statistic=None)
        if not self.is_defined_for_constant and (
            self._is_constant(encoded_x) or self._is_constant(encoded_y)
        ):
            return CorrelationResult(statistic=None)
        result: CorrelationResult = self._compute(encoded_x, encoded_y)
        if (
            self.is_unit_bounded
            and result.statistic is not None
            and abs(result.statistic) > 1
        ):
            return replace(result, statistic=math.copysign(1.0, result.statistic))
        return result

    def evaluate_with_scaling(
        self, x: ArrayLike, y: ArrayLike, scaling: ScalingMethod
    ) -> CorrelationResult:
        """
        Evaluate the metric after scaling each numeric variable separately.

        Labels stay as they are, and data held by the metric, such as covariates, is
        scaled along. A kind of scaling that would change what the metric measures
        is rejected before any computation; see `supported_scaling_kinds`.

        Parameters
        ----------
        x : ArrayLike
            The first variable.
        y : ArrayLike
            The second variable, aligned with x.
        scaling : ScalingMethod
            The scaling method, from which x and y each get their own scaler.

        Returns
        -------
        CorrelationResult
            The outcome on the scaled variables, as from `evaluate`.

        Raises
        ------
        TypeError
            If this metric takes no numeric variable, or if a variable has the
            wrong data type for this metric.
        ValueError
            If this metric does not support the kind of scaling, sample-wise
            scaling is applied to a variable with one feature, a variable is invalid
            as for `evaluate`, or the scaler rejects it.
        """
        if not self.x_kind.is_numeric and not self.y_kind.is_numeric:
            raise TypeError(
                f"{type(self).__name__} takes no numeric variable, so it cannot be scaled"
            )
        supported: frozenset[ScalingKind] = self.supported_scaling_kinds()
        if scaling.kind not in supported:
            allowed: str = "\n  - ".join(
                kind.name for kind in ScalingKind if kind in supported
            )
            raise ValueError(
                f"{type(self).__name__} does not support {scaling.kind.name} scaling, because {self._scaling_rejection_reason(scaling.kind)}.\nUse one of:\n  - {allowed}"
            )
        return self._with_scaled_data(scaling).evaluate(
            self._scale(x, self.x_kind, "x", scaling),
            self._scale(y, self.y_kind, "y", scaling),
        )

    def evaluate_matrix(self, X: ArrayLike) -> CorrelationMatrix:
        """
        Evaluate the metric on every pair of columns.

        Parameters
        ----------
        X : ArrayLike
            The variables with shape (n_samples, n_columns).

        Returns
        -------
        CorrelationMatrix
            The pairwise outcomes.

        Raises
        ------
        TypeError
            If this metric takes variables of different kinds, or if a column
            has the wrong data type.
        ValueError
            If X is not two-dimensional, or a column is invalid.
        """
        if self.x_kind is not self.y_kind:
            raise TypeError(
                f"{type(self).__name__} relates variables of different kinds, so it has no matrix form"
            )
        columns: NDArray[np.generic] = np.asarray(X)
        if columns.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {columns.shape}")
        num_columns: int = np.size(columns, axis=1)
        statistics: FloatArray = np.full((num_columns, num_columns), np.nan)
        p_values: FloatArray = np.full((num_columns, num_columns), np.nan)
        pairs: list[tuple[int, int]] = (
            [(i, i) for i in range(num_columns)]
            + list(combinations(range(num_columns), 2))
            if self.is_symmetric
            else [(i, j) for i in range(num_columns) for j in range(num_columns)]
        )
        for i, j in pairs:
            column_x: NDArray[np.generic] = columns[:, i]
            column_y: NDArray[np.generic] = columns[:, j]
            result: CorrelationResult = self.evaluate(column_x, column_y)
            statistic: float = np.nan if result.statistic is None else result.statistic
            p_value: float = np.nan if result.p_value is None else result.p_value
            statistics[i, j] = statistic
            p_values[i, j] = p_value
            if self.is_symmetric:
                statistics[j, i] = statistic
                p_values[j, i] = p_value
        return CorrelationMatrix(statistics=statistics, p_values=p_values)

    @abstractmethod
    def _compute(self, x: FloatArray, y: FloatArray) -> CorrelationResult:
        """
        Compute the metric on validated data.

        Parameters
        ----------
        x : FloatArray
            The encoded first variable.
        y : FloatArray
            The encoded second variable, aligned with x.

        Returns
        -------
        CorrelationResult
            The outcome.
        """

    def _encode_pair(self, x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
        if not self.is_label_set_shared:
            return self.x_kind.encode(x, "x"), self.y_kind.encode(y, "y")
        x_labels: NDArray[np.generic] = np.asarray(x)
        y_labels: NDArray[np.generic] = np.asarray(y)
        if x_labels.shape != y_labels.shape:
            raise ValueError(
                f"x and y must have the same length, got shapes {x_labels.shape} and {y_labels.shape}"
            )
        # Concatenation would stringify the numbers, so that 1.0 never equals "1".
        kinds: set[str] = {x_labels.dtype.kind, y_labels.dtype.kind}
        if kinds & set("US") and kinds & set("biuf"):
            raise TypeError(
                f"x and y must contain labels of one type, got {x_labels.dtype} and {y_labels.dtype}"
            )
        codes: FloatArray = self.x_kind.encode(
            np.concatenate([x_labels, y_labels]), "x and y"
        )
        return codes[: x_labels.shape[0]], codes[x_labels.shape[0] :]

    def _scaling_rejection_reason(self, kind: ScalingKind) -> str:
        restriction: ScalingRestriction | None = self.scaling_restriction
        if restriction is not None and kind not in restriction.allowed_kinds:
            return restriction.reason
        return (
            "it would leave only the sign of each value of a one-dimensional variable"
        )

    def _validate_num_samples(self, num_samples: int) -> None:
        # Metrics that hold data aligned with x and y override this to check it.
        _ = num_samples

    def _with_scaled_data(self, scaling: ScalingMethod) -> Self:
        # Metrics that hold data aligned with x and y override this to scale it too.
        _ = scaling
        return self

    @staticmethod
    def _scale(
        values: ArrayLike, kind: VariableKind, name: str, scaling: ScalingMethod
    ) -> ArrayLike:
        if not kind.is_numeric:
            return values
        encoded: FloatArray = kind.encode(values, name)
        samples: FloatArray = encoded.reshape(len(encoded), -1)
        if scaling.kind is ScalingKind.SAMPLE_WISE and np.size(samples, axis=1) == 1:
            raise ValueError(
                f"{name} has one feature, so {scaling.kind.name} scaling would leave only the sign of each value"
            )
        scaled: NDArray[np.floating] = scaling.build_scaler().fit_transform(samples)
        return scaled.reshape(encoded.shape)

    @staticmethod
    def _is_constant(values: FloatArray) -> bool:
        spreads: FloatArray = np.ptp(values, axis=0)
        return float(np.max(spreads)) == 0
