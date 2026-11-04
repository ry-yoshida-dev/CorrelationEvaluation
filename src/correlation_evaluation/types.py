"""Type aliases for correlation data."""

import numpy as np
from data_scaler import FeatureScaling, ScalingParameters
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type ScalingMethod = FeatureScaling | ScalingParameters
