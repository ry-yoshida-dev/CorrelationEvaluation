from __future__ import annotations

from dataclasses import dataclass

from data_scaler import ScalingKind


@dataclass(frozen=True)
class ScalingRestriction:
    """
    Kinds of scaling that keep what a metric measures, and why others do not.

    Attributes
    ----------
    allowed_kinds : frozenset[ScalingKind]
        The kinds that the metric accepts, before the limits of its variables.
    reason : str
        Why the other kinds are rejected, completing "because ...".
    """

    allowed_kinds: frozenset[ScalingKind]
    reason: str

    @classmethod
    def for_agreement(cls) -> ScalingRestriction:
        """
        Build the restriction of metrics that compare x and y on one scale.

        Returns
        -------
        ScalingRestriction
            The restriction, which accepts no scaling.
        """
        return cls(
            allowed_kinds=frozenset({ScalingKind.IDENTITY}),
            reason="scaling x and y separately would hide the differences it detects",
        )
