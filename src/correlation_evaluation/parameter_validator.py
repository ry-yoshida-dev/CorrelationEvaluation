from __future__ import annotations

import math
from enum import Enum
from numbers import Integral


class ParameterValidator:
    """
    Runtime checks that range comparisons cannot express.

    NaN compares false to everything, non-integers pass integer range checks, and
    other values reach enumeration dispatch, so all would otherwise only fail
    later with an unclear error.
    """

    @staticmethod
    def require_integers(**values: object) -> None:
        """
        Check that values are integers, or None for optional parameters.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is neither None nor an integer; booleans are rejected.
        """
        for name, value in values.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {value!r}")

    @staticmethod
    def require_members(enum_type: type[Enum], **values: object) -> None:
        """
        Check that values are members of an enumeration.

        Parameters
        ----------
        enum_type : type[Enum]
            The enumeration.
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not a member.
        """
        for name, value in values.items():
            if not isinstance(value, enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}, got {value!r}")

    @staticmethod
    def require_not_nan(**values: float) -> None:
        """
        Check that values are not NaN.

        Parameters
        ----------
        **values : float
            The values by parameter name.

        Raises
        ------
        ValueError
            If a value is NaN.
        """
        for name, value in values.items():
            if math.isnan(value):
                raise ValueError(f"{name} must not be NaN")
