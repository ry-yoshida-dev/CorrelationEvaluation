from enum import Enum
from typing import Literal


class EntropyAverage(Enum):
    """
    Way to combine two entropies into the normalizer of mutual information.

    Attributes
    ----------
    ARITHMETIC
        Arithmetic mean.
    GEOMETRIC
        Geometric mean.
    MIN
        The smaller entropy.
    MAX
        The larger entropy.
    """

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    MIN = "min"
    MAX = "max"

    @property
    def option_name(self) -> Literal["arithmetic", "geometric", "min", "max"]:
        """
        Get the option name that the backing functions accept.

        Returns
        -------
        Literal["arithmetic", "geometric", "min", "max"]
            The option name.
        """
        match self:
            case EntropyAverage.ARITHMETIC:
                return "arithmetic"
            case EntropyAverage.GEOMETRIC:
                return "geometric"
            case EntropyAverage.MIN:
                return "min"
            case EntropyAverage.MAX:
                return "max"
