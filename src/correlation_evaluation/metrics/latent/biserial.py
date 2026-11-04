from dataclasses import dataclass
from typing import ClassVar

from ...variable_kind import VariableKind
from .polyserial import Polyserial


@dataclass(frozen=True)
class Biserial(Polyserial):
    """
    Biserial correlation, the polyserial correlation of a binary x with a numeric y.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.BINARY
