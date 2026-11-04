import itertools

import numpy as np
import pytest
from numpy.typing import NDArray

from correlation_evaluation import (
    ChatterjeeXi,
    DistanceCorrelation,
    HilbertSchmidt,
    HoeffdingD,
    MultiscaleGraph,
    NearestNeighborMutualInformation,
    Pearson,
)


def parabola() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x: NDArray[np.float64] = np.linspace(-1.0, 1.0, 201)
    return x, x**2


def independent() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng: np.random.Generator = np.random.default_rng(0)
    return rng.standard_normal(200), rng.standard_normal(200)


def assert_detects_nonlinear_dependence(
    metric: ChatterjeeXi
    | DistanceCorrelation
    | HilbertSchmidt
    | HoeffdingD
    | NearestNeighborMutualInformation,
    threshold: float,
) -> None:
    dependent_x: NDArray[np.float64]
    dependent_y: NDArray[np.float64]
    dependent_x, dependent_y = parabola()
    assert abs(Pearson().evaluate(dependent_x, dependent_y).statistic or 0.0) < 1e-9
    dependent: float | None = metric.evaluate(dependent_x, dependent_y).statistic
    noise_x: NDArray[np.float64]
    noise_y: NDArray[np.float64]
    noise_x, noise_y = independent()
    null: float | None = metric.evaluate(noise_x, noise_y).statistic
    assert dependent is not None
    assert null is not None
    assert dependent > threshold
    assert abs(null) < threshold


def test_chatterjee_xi_detects_nonlinear_dependence() -> None:
    assert_detects_nonlinear_dependence(ChatterjeeXi(), 0.3)


def test_chatterjee_xi_is_asymmetric() -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x, y = parabola()
    forward: float | None = ChatterjeeXi().evaluate(x, y).statistic
    backward: float | None = ChatterjeeXi().evaluate(y, x).statistic
    assert forward is not None
    assert backward is not None
    assert forward > backward + 0.5


def test_chatterjee_xi_hand_computed_value() -> None:
    # 1 - 3 * sum(|r[i+1] - r[i]|) / (n^2 - 1) with ranks 1, 3, 2, 4.
    result = ChatterjeeXi(is_y_continuous=True).evaluate([1, 2, 3, 4], [1, 3, 2, 4])
    assert result.statistic == pytest.approx(1 - 3 * 5 / 15)


def test_hoeffding_d_detects_nonlinear_dependence() -> None:
    assert_detects_nonlinear_dependence(HoeffdingD(), 0.1)


def test_hoeffding_d_perfect_monotonic_relationship_is_one() -> None:
    x: NDArray[np.float64] = np.arange(10.0)
    assert HoeffdingD().evaluate(x, x**3).statistic == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]),
        ([1, 2, 3, 4, 5, 6, 7], [3, 1, 7, 2, 6, 4, 5]),
    ],
)
def test_hoeffding_d_matches_the_u_statistic_definition(
    x: list[int], y: list[int]
) -> None:
    assert HoeffdingD().evaluate(x, y).statistic == pytest.approx(
        brute_force_hoeffding(x, y)
    )


def brute_force_hoeffding(x: list[int], y: list[int]) -> float:
    # Hoeffding's symmetric kernel over all 5-subsets of untied samples.
    def psi(a: int, b: int, c: int) -> int:
        return int(a >= b) - int(a >= c)

    kernels: list[float] = []
    for subset in itertools.combinations(range(len(x)), 5):
        for i, j, k, p, q in itertools.permutations(subset):
            kernels.append(
                0.25
                * psi(x[i], x[j], x[k])
                * psi(x[i], x[p], x[q])
                * psi(y[i], y[j], y[k])
                * psi(y[i], y[p], y[q])
            )
    return 30 * float(np.mean(kernels))


def test_distance_correlation_detects_nonlinear_dependence() -> None:
    assert_detects_nonlinear_dependence(DistanceCorrelation(), 0.3)


def test_distance_correlation_linear_relationship_is_perfect() -> None:
    x: NDArray[np.float64] = np.arange(10.0)
    assert DistanceCorrelation().evaluate(x, 3 * x + 1).statistic == pytest.approx(1.0)


def test_distance_correlation_bias_correction_shrinks_the_null() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.float64] = rng.standard_normal(30)
    y: NDArray[np.float64] = rng.standard_normal(30)
    corrected = DistanceCorrelation(is_bias_corrected=True).evaluate(x, y).statistic
    plain = DistanceCorrelation().evaluate(x, y).statistic
    assert corrected is not None
    assert plain is not None
    assert abs(corrected) < plain**2


def test_distance_correlation_accepts_multivariate_input() -> None:
    x: NDArray[np.float64] = np.random.default_rng(0).standard_normal((30, 3))
    assert DistanceCorrelation().evaluate(x, x[:, :2]).statistic is not None


def test_hilbert_schmidt_detects_nonlinear_dependence() -> None:
    assert_detects_nonlinear_dependence(HilbertSchmidt(), 0.2)


def test_hilbert_schmidt_identical_variables_are_perfect() -> None:
    x: NDArray[np.float64] = np.arange(10.0)
    assert HilbertSchmidt(bandwidth=2.0).evaluate(x, x).statistic == pytest.approx(1.0)


def test_hilbert_schmidt_invalid_bandwidth_is_rejected() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        _ = HilbertSchmidt(bandwidth=0.0)


def test_multiscale_graph_detects_dependence_and_is_reproducible() -> None:
    x: NDArray[np.float64] = np.linspace(-1.0, 1.0, 20)
    metric: MultiscaleGraph = MultiscaleGraph()
    first = metric.evaluate(x, x**2)
    assert first.p_value is not None
    assert first.p_value < 0.05
    assert first == metric.evaluate(x, x**2)


def test_multiscale_graph_too_few_samples_is_undefined() -> None:
    assert (
        MultiscaleGraph().evaluate([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]).statistic is None
    )


def test_nearest_neighbor_mutual_information_detects_nonlinear_dependence() -> None:
    assert_detects_nonlinear_dependence(NearestNeighborMutualInformation(), 0.3)


def test_nearest_neighbor_mutual_information_scaled_value_approximates_pearson_for_normal_data() -> (
    None
):
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.float64] = rng.standard_normal(5000)
    y: NDArray[np.float64] = 0.8 * x + 0.6 * rng.standard_normal(5000)
    metric: NearestNeighborMutualInformation = NearestNeighborMutualInformation(
        is_correlation_scaled=True
    )
    assert metric.evaluate(x, y).statistic == pytest.approx(0.8, abs=0.03)


def test_nearest_neighbor_mutual_information_fewer_samples_than_neighbors_is_undefined() -> (
    None
):
    metric: NearestNeighborMutualInformation = NearestNeighborMutualInformation(
        num_neighbors=5
    )
    assert metric.evaluate([1, 2, 3, 4], [1, 3, 2, 4]).statistic is None
