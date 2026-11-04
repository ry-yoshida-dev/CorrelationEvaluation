import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from correlation_evaluation import (
    Alternative,
    ConfidenceInterval,
    CorrelationResult,
    Cosine,
    LaggedPearson,
    Partial,
    Pearson,
    Semipartial,
)


def test_pearson_hand_computed_value() -> None:
    result: CorrelationResult = Pearson().evaluate([1, 2, 3, 4], [1, 3, 2, 4])
    assert result.statistic == pytest.approx(0.8)
    assert result.p_value == pytest.approx(0.2)


def test_pearson_confidence_interval_contains_the_statistic() -> None:
    result: CorrelationResult = Pearson(confidence_level=0.9).evaluate(
        [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5]
    )
    interval: ConfidenceInterval | None = result.confidence_interval
    assert interval is not None
    assert result.statistic is not None
    assert interval.low < result.statistic < interval.high
    assert interval.level == 0.9


def test_pearson_one_sided_alternative_halves_the_p_value() -> None:
    x: list[int] = [1, 2, 3, 4, 5]
    y: list[int] = [1, 3, 2, 5, 4]
    two_sided: float | None = Pearson().evaluate(x, y).p_value
    greater: float | None = (
        Pearson(alternative=Alternative.GREATER).evaluate(x, y).p_value
    )
    assert two_sided is not None
    assert greater == pytest.approx(two_sided / 2)


def test_cosine_hand_computed_value() -> None:
    result: CorrelationResult = Cosine().evaluate([1, 0], [1, 1])
    assert result.statistic == pytest.approx(1 / math.sqrt(2))


def test_cosine_constant_input_is_defined() -> None:
    assert Cosine().evaluate([1, 1], [2, 2]).statistic == pytest.approx(1.0)


def test_cosine_zero_vector_is_undefined() -> None:
    assert Cosine().evaluate([0, 0], [1, 2]).statistic is None


def test_lagged_pearson_recovers_a_known_delay() -> None:
    x: NDArray[np.float64] = np.random.default_rng(0).standard_normal(50)
    y: NDArray[np.float64] = np.roll(x, 2)
    assert LaggedPearson(lag=2).evaluate(x, y).statistic == pytest.approx(1.0)
    assert LaggedPearson(lag=-2).evaluate(y, x).statistic == pytest.approx(1.0)


def test_lagged_pearson_zero_lag_equals_pearson() -> None:
    x: list[int] = [1, 2, 3, 4, 5]
    y: list[int] = [1, 3, 2, 5, 4]
    assert LaggedPearson(lag=0).evaluate(x, y).statistic == pytest.approx(
        Pearson().evaluate(x, y).statistic
    )


def test_lagged_pearson_lag_beyond_the_data_is_undefined() -> None:
    assert LaggedPearson(lag=3).evaluate([1, 2, 3, 4], [1, 2, 4, 3]).statistic is None


def test_lagged_pearson_constant_overlap_is_undefined() -> None:
    x: list[int] = [1, 1, 1, 1, 9]
    y: list[int] = [1, 2, 3, 4, 5]
    assert LaggedPearson(lag=1).evaluate(x, y).statistic is None


def covariate_sample() -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    rng: np.random.Generator = np.random.default_rng(0)
    z: NDArray[np.float64] = rng.standard_normal(200)
    return (
        z + 0.1 * rng.standard_normal(200),
        z + 0.1 * rng.standard_normal(200),
        z,
    )


def test_covariate_adjusted_partial_removes_a_confounder() -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    x, y, z = covariate_sample()
    statistic: float | None = Partial(z).evaluate(x, y).statistic
    raw: float | None = Pearson().evaluate(x, y).statistic
    assert statistic is not None
    assert raw is not None
    assert raw > 0.9
    assert abs(statistic) < 0.2


def test_covariate_adjusted_partial_matches_the_recursive_formula() -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    x, y, z = covariate_sample()
    r_xy: float = float(stats.pearsonr(x, y).statistic)
    r_xz: float = float(stats.pearsonr(x, z).statistic)
    r_yz: float = float(stats.pearsonr(y, z).statistic)
    expected: float = (r_xy - r_xz * r_yz) / math.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    assert Partial(z).evaluate(x, y).statistic == pytest.approx(expected)
    assert Semipartial(z).evaluate(x, y).statistic == pytest.approx(
        (r_xy - r_xz * r_yz) / math.sqrt(1 - r_yz**2)
    )


def test_covariate_adjusted_rank_based_partial_ignores_monotonic_transforms() -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    x, y, z = covariate_sample()
    ranked: Partial = Partial(z, is_rank_based=True)
    assert ranked.evaluate(np.exp(x), y).statistic == pytest.approx(
        ranked.evaluate(x, y).statistic
    )


def test_covariate_adjusted_fully_explained_variable_is_undefined() -> None:
    z: NDArray[np.float64] = np.arange(10.0)
    y: list[int] = [1, 3, 2, 5, 4, 7, 6, 9, 8, 10]
    assert Partial(z).evaluate(2 * z + 1, y).statistic is None


def test_covariate_adjusted_misaligned_covariates_are_rejected() -> None:
    with pytest.raises(ValueError, match="one row per sample"):
        _ = Partial(np.arange(3.0)).evaluate([1, 2, 3, 4], [1, 3, 2, 4])


def test_covariate_adjusted_invalid_covariates_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _ = Partial(np.array([1.0, np.nan]))


def test_covariate_adjusted_semipartial_p_value_tests_the_partial_correlation() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    z: NDArray[np.float64] = rng.standard_normal(50)
    x: NDArray[np.float64] = 0.9 * z + 0.4 * rng.standard_normal(50)
    y: NDArray[np.float64] = z + x + rng.standard_normal(50)
    semipartial = Semipartial(z).evaluate(x, y)
    partial = Partial(z).evaluate(x, y)
    assert semipartial.statistic != pytest.approx(partial.statistic)
    assert semipartial.p_value == pytest.approx(partial.p_value)


def test_covariate_adjusted_semipartial_p_value_is_calibrated_under_the_null() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    num_rejected: int = 0
    num_repetitions: int = 1000
    for _ in range(num_repetitions):
        z: NDArray[np.float64] = rng.standard_normal(50)
        x: NDArray[np.float64] = 0.95 * z + 0.3 * rng.standard_normal(50)
        y: NDArray[np.float64] = z + rng.standard_normal(50)
        p_value: float | None = Semipartial(z).evaluate(x, y).p_value
        assert p_value is not None
        num_rejected += p_value < 0.05
    assert 0.03 < num_rejected / num_repetitions < 0.07


def test_covariate_adjusted_semipartial_of_explained_x_has_no_p_value() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    z: NDArray[np.float64] = rng.standard_normal(20)
    result = Semipartial(z).evaluate(2 * z + 1, rng.standard_normal(20))
    assert result.statistic == pytest.approx(0.0, abs=1e-9)
    assert result.p_value is None
