from enum import Enum
from typing import Literal


class KendallVariant(Enum):
    """
    Tie handling of Kendall's tau.

    Attributes
    ----------
    B
        Tau-b, for variables with the same number of distinct values.
    C
        Tau-c (Stuart's), for rectangular tables of ordinal variables.
    """

    B = "b"
    C = "c"

    @property
    def option_name(self) -> Literal["b", "c"]:
        """
        Get the option name that the backing test accepts.

        Returns
        -------
        Literal["b", "c"]
            The option name.
        """
        match self:
            case KendallVariant.B:
                return "b"
            case KendallVariant.C:
                return "c"
