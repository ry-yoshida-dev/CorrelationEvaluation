import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from correlation_evaluation import (
    BiweightMidcorrelation,
    Pearson,
    PercentageBend,
    Winsorized,
)


def sample_with_outlier() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.float64] = rng.standard_normal(100)
    y: NDArray[np.float64] = x + 0.5 * rng.standard_normal(100)
    x[0], y[0] = 100.0, -100.0
    return x, y


@pytest.mark.parametrize(
    "metric", [BiweightMidcorrelation(), PercentageBend(), Winsorized()]
)
def test_resists_an_outlier_that_flips_pearson(
    metric: BiweightMidcorrelation | PercentageBend | Winsorized,
) -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x, y = sample_with_outlier()
    pearson: float | None = Pearson().evaluate(x, y).statistic
    robust: float | None = metric.evaluate(x, y).statistic
    assert pearson is not None
    assert robust is not None
    assert pearson < 0
    assert robust > 0.6


def test_biweight_midcorrelation_equals_pearson_when_the_constant_is_large() -> None:
    x: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.5]
    y: list[float] = [2.0, 1.0, 4.0, 3.5, 6.0, 5.0]
    # The weights flatten, but the deviations stay centered on the medians.
    huge: float | None = (
        BiweightMidcorrelation(tuning_constant=1e6).evaluate(x, y).statistic
    )
    assert huge == pytest.approx(Pearson().evaluate(x, y).statistic, abs=0.01)


def test_biweight_midcorrelation_zero_median_absolute_deviation_is_undefined() -> None:
    metric: BiweightMidcorrelation = BiweightMidcorrelation()
    assert metric.evaluate([1, 1, 1, 1, 9], [1, 2, 3, 4, 5]).statistic is None


def test_biweight_midcorrelation_invalid_constant_is_rejected() -> None:
    with pytest.raises(ValueError, match="tuning_constant"):
        _ = BiweightMidcorrelation(tuning_constant=0.0)


def test_percentage_bend_monotone_linear_data_is_perfect() -> None:
    result = PercentageBend().evaluate(np.arange(20.0), 2 * np.arange(20.0))
    assert result.statistic == pytest.approx(1.0)
    assert result.p_value == 0.0


def test_percentage_bend_invalid_bend_is_rejected() -> None:
    with pytest.raises(ValueError, match="bend"):
        _ = PercentageBend(bend=0.6)


def test_winsorized_zero_limit_equals_pearson() -> None:
    x: list[int] = [1, 2, 3, 4, 5]
    y: list[int] = [1, 3, 2, 5, 4]
    result = Winsorized(limit=0.0).evaluate(x, y)
    reference = Pearson().evaluate(x, y)
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.p_value)


def test_winsorized_hand_computed_value() -> None:
    # One value per tail becomes its neighbor, so x becomes 2, 2, 3, 4, 4.
    result = Winsorized(limit=0.2).evaluate([1, 2, 3, 4, 5], [5, 2, 3, 4, 1])
    # Likewise, y becomes 4, 2, 3, 4, 2.
    assert result.statistic == pytest.approx(
        stats.pearsonr([2, 2, 3, 4, 4], [4, 2, 3, 4, 2]).statistic
    )


def test_winsorized_invalid_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit"):
        _ = Winsorized(limit=0.5)


def test_biweight_midcorrelation_without_weights_off_the_median_is_undefined() -> None:
    metric: BiweightMidcorrelation = BiweightMidcorrelation(tuning_constant=0.5)
    assert metric.evaluate([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]).statistic is None
