from __future__ import annotations

import math
from enum import Enum
from numbers import Integral, Real

import numpy as np


class ParameterValidator:
    """
    Runtime checks that range comparisons cannot express.

    NaN compares false to everything, booleans and non-integers pass numeric range
    checks, any value has a truth value, and other values reach enumeration dispatch,
    so all would otherwise fail later with an unclear error or not at all.
    """

    @staticmethod
    def require_booleans(**values: object) -> None:
        """
        Check that values are booleans.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not a boolean.
        """
        for name, value in values.items():
            if not isinstance(value, bool | np.bool_):
                raise TypeError(f"{name} must be a boolean, got {value!r}")

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
    def require_seeds(**values: object) -> None:
        """
        Check that values are random seeds that all backing libraries accept, or None.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is neither None nor an integer.
        ValueError
            If a value is outside [0, 2**32).
        """
        ParameterValidator.require_integers(**values)
        for name, value in values.items():
            if isinstance(value, Integral) and not 0 <= int(value) < 2**32:
                raise ValueError(f"{name} must be in [0, 2**32), got {value!r}")

    @staticmethod
    def require_real_numbers(**values: object) -> None:
        """
        Check that values are real numbers other than NaN.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not a real number; booleans are rejected.
        ValueError
            If a value is NaN.
        """
        for name, value in values.items():
            if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, got {value!r}")
            if math.isnan(value):
                raise ValueError(f"{name} must not be NaN")
