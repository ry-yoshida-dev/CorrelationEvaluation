from enum import Enum
from typing import Literal


class KappaWeighting(Enum):
    """
    Penalty of Cohen's kappa for a disagreement between two labels.

    Attributes
    ----------
    UNWEIGHTED
        Every disagreement counts the same, for nominal labels.
    LINEAR
        Proportional to the distance between the sorted labels, for ordinal labels.
    QUADRATIC
        Proportional to the squared distance, for ordinal labels.
    """

    UNWEIGHTED = "unweighted"
    LINEAR = "linear"
    QUADRATIC = "quadratic"

    @property
    def option_name(self) -> Literal["linear", "quadratic"] | None:
        """
        Get the option name that the backing function accepts.

        Returns
        -------
        Literal["linear", "quadratic"] | None
            The option name; None for unweighted.
        """
        match self:
            case KappaWeighting.UNWEIGHTED:
                return None
            case KappaWeighting.LINEAR:
                return "linear"
            case KappaWeighting.QUADRATIC:
                return "quadratic"
