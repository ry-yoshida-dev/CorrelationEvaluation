import numpy as np
from numpy.typing import NDArray

def mutual_info_regression(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_neighbors: int = ...,
    random_state: int | None = ...,
) -> NDArray[np.float64]: ...
