import math

import numpy as np
from scipy import special, stats

from ..types import FloatArray


class BivariateNormal:
    """
    Standard bivariate normal distribution.
    """

    # Owen's T arguments divide by the limits, and its identity holds in the limit.
    _ZERO_REPLACEMENT: float = 1e-12

    @classmethod
    def cdf(cls, h: FloatArray, k: FloatArray, correlation: float) -> FloatArray:
        """
        Evaluate the distribution function with Owen's T.

        Unlike SciPy's multivariate normal distribution function, this is
        deterministic and vectorized, which keeps likelihood maximization smooth
        and fast.

        Parameters
        ----------
        h : FloatArray
            The upper limits of the first variable.
        k : FloatArray
            The upper limits of the second variable, broadcastable with h.
        correlation : float
            The correlation, in (-1, 1).

        Returns
        -------
        FloatArray
            The probabilities.
        """
        safe_h: FloatArray = np.where((h > 0) | (h < 0), h, cls._ZERO_REPLACEMENT)
        safe_k: FloatArray = np.where((k > 0) | (k < 0), k, cls._ZERO_REPLACEMENT)
        root: float = math.sqrt(1 - correlation**2)
        probabilities: FloatArray = (
            0.5 * (stats.norm.cdf(safe_h) + stats.norm.cdf(safe_k))
            - special.owens_t(safe_h, (safe_k - correlation * safe_h) / (safe_h * root))
            - special.owens_t(safe_k, (safe_h - correlation * safe_k) / (safe_k * root))
            - 0.5 * (safe_h * safe_k < 0)
        )
        return probabilities
