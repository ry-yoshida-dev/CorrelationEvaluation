from dataclasses import dataclass
from typing import ClassVar

from ...variable_kind import VariableKind
from .polychoric import Polychoric


@dataclass(frozen=True)
class Tetrachoric(Polychoric):
    """
    Tetrachoric correlation, the polychoric correlation of two binary variables.
    """

    x_kind: ClassVar[VariableKind] = VariableKind.BINARY
    y_kind: ClassVar[VariableKind] = VariableKind.BINARY
