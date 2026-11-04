from enum import Enum
from typing import Literal


class Alternative(Enum):
    """
    Alternative hypothesis of a significance test.

    Attributes
    ----------
    TWO_SIDED
        The association is nonzero.
    GREATER
        The association is positive.
    LESS
        The association is negative.
    """

    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"

    @property
    def option_name(self) -> Literal["two-sided", "greater", "less"]:
        """
        Get the option name that the backing tests accept.

        Returns
        -------
        Literal["two-sided", "greater", "less"]
            The option name.
        """
        match self:
            case Alternative.TWO_SIDED:
                return "two-sided"
            case Alternative.GREATER:
                return "greater"
            case Alternative.LESS:
                return "less"
