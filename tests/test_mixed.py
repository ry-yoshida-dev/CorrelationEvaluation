import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from correlation_evaluation import (
    CorrelationRatio,
    Pearson,
    PointBiserial,
    RankBiserial,
)


def test_point_biserial_equals_pearson_on_codes() -> None:
    x: list[int] = [0, 0, 0, 1, 1, 1]
    y: list[float] = [1.0, 2.0, 3.0, 2.5, 4.0, 5.0]
    assert PointBiserial().evaluate(x, y).statistic == pytest.approx(
        Pearson().evaluate(x, y).statistic
    )


def test_point_biserial_accepts_string_labels() -> None:
    result = PointBiserial().evaluate(["a", "a", "b", "b"], [1.0, 2.0, 3.0, 5.0])
    assert result.statistic is not None
    assert result.statistic > 0


def test_rank_biserial_complete_separation_is_perfect() -> None:
    assert RankBiserial().evaluate([0, 0, 1, 1], [1, 2, 3, 4]).statistic == 1.0
    assert RankBiserial().evaluate([1, 1, 0, 0], [1, 2, 3, 4]).statistic == -1.0


def test_rank_biserial_hand_computed_value() -> None:
    # Pairs (upper, lower): 3>1, 3>2, 1=1 tied, 1<2, so (2 - 1) / 4.
    result = RankBiserial().evaluate([0, 0, 1, 1], [1, 2, 3, 1])
    assert result.statistic == pytest.approx(0.25)


def test_correlation_ratio_hand_computed_value() -> None:
    # Group means 2 and 4 around 3: between 2 * 1 + 2 * 1 = 4, total 4 + 2 = 6.
    result = CorrelationRatio().evaluate(["a", "a", "b", "b"], [1, 3, 3, 5])
    assert result.statistic == pytest.approx(math.sqrt(4 / 8))


def test_correlation_ratio_matches_one_way_analysis_of_variance() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    y: NDArray[np.float64] = rng.standard_normal(30)
    x: NDArray[np.int64] = np.repeat(np.arange(3), 10)
    result = CorrelationRatio().evaluate(x, y)
    assert result.p_value == pytest.approx(
        stats.f_oneway(y[:10], y[10:20], y[20:]).pvalue
    )


def test_correlation_ratio_groups_that_explain_everything_have_no_test() -> None:
    result = CorrelationRatio().evaluate([0, 0, 1, 1], [1, 1, 2, 2])
    assert result.statistic == pytest.approx(1.0)
    assert result.p_value is None
