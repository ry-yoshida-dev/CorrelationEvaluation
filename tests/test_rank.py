import math
from itertools import combinations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from correlation_evaluation import (
    Alternative,
    BlomqvistBeta,
    CorrelationResult,
    GoodmanKruskalGamma,
    KendallTau,
    KendallVariant,
    NormalScores,
    SomersD,
    Spearman,
    WeightedTau,
)

X: list[int] = [1, 1, 2, 2, 3, 3, 4, 5]
Y: list[int] = [1, 2, 1, 3, 3, 2, 4, 4]


def count_pairs(x: list[int], y: list[int]) -> tuple[int, int, int, int]:
    concordant: int = 0
    discordant: int = 0
    untied_x: int = 0
    untied_y: int = 0
    for i, j in combinations(range(len(x)), 2):
        product: int = (x[i] - x[j]) * (y[i] - y[j])
        sign: int = (product > 0) - (product < 0)
        concordant += sign == 1
        discordant += sign == -1
        untied_x += x[i] != x[j]
        untied_y += y[i] != y[j]
    return concordant, discordant, untied_x, untied_y


def test_pair_counting_metrics_gamma_matches_brute_force() -> None:
    concordant, discordant, _, _ = count_pairs(X, Y)
    assert GoodmanKruskalGamma().evaluate(X, Y).statistic == pytest.approx(
        (concordant - discordant) / (concordant + discordant)
    )


def test_pair_counting_metrics_gamma_ignores_tied_pairs() -> None:
    assert GoodmanKruskalGamma().evaluate(
        [1, 1, 2], [1, 2, 2]
    ).statistic == pytest.approx(1.0)


def test_pair_counting_metrics_somers_d_divides_by_pairs_untied_on_x() -> None:
    concordant, discordant, untied_x, untied_y = count_pairs(X, Y)
    assert SomersD().evaluate(X, Y).statistic == pytest.approx(
        (concordant - discordant) / untied_x
    )
    assert SomersD().evaluate(Y, X).statistic == pytest.approx(
        (concordant - discordant) / untied_y
    )


def test_pair_counting_metrics_tau_b_matches_brute_force() -> None:
    concordant, discordant, untied_x, untied_y = count_pairs(X, Y)
    assert KendallTau().evaluate(X, Y).statistic == pytest.approx(
        (concordant - discordant) / math.sqrt(untied_x * untied_y)
    )


def test_pair_counting_metrics_tau_c_uses_the_table_dimension() -> None:
    concordant, discordant, _, _ = count_pairs(X, Y)
    num_levels: int = 4
    expected: float = (
        2 * (concordant - discordant) / (len(X) ** 2 * (num_levels - 1) / num_levels)
    )
    result = KendallTau(variant=KendallVariant.C).evaluate(X, Y)
    assert result.statistic == pytest.approx(expected)
    assert result.p_value is not None


def test_spearman_equals_pearson_of_ranks() -> None:
    x: list[float] = [1.0, 2.0, 3.0, 4.0, 50.0]
    y: list[float] = [1.0, 3.0, 2.0, 4.0, 5.0]
    assert Spearman().evaluate(x, y).statistic == pytest.approx(0.9)


def test_weighted_tau_top_ranks_outweigh_bottom_ranks() -> None:
    top_swapped = WeightedTau().evaluate([5, 4, 3, 2, 1], [4, 5, 3, 2, 1])
    bottom_swapped = WeightedTau().evaluate([5, 4, 3, 2, 1], [5, 4, 3, 1, 2])
    assert top_swapped.statistic is not None
    assert bottom_swapped.statistic is not None
    assert top_swapped.statistic < bottom_swapped.statistic


def test_weighted_tau_identical_rankings_are_perfect() -> None:
    assert WeightedTau().evaluate([1, 2, 3], [1, 2, 3]).statistic == pytest.approx(1)


def test_blomqvist_beta_hand_computed_value() -> None:
    # Signs around the medians agree for four samples and differ for two.
    result = BlomqvistBeta().evaluate([1, 2, 3, 4, 5, 6], [1, 2, 4, 3, 5, 6])
    assert result.statistic == pytest.approx(1 / 3)


def test_blomqvist_beta_all_samples_on_a_median_is_undefined() -> None:
    assert BlomqvistBeta().evaluate([1, 2, 3], [3, 2, 2]).statistic == -1.0
    assert BlomqvistBeta().evaluate([1, 2, 2], [2, 2, 3]).statistic is None


def test_normal_scores_ignores_monotonic_transforms() -> None:
    x: NDArray[np.float64] = np.random.default_rng(0).standard_normal(50)
    y: NDArray[np.float64] = x + np.random.default_rng(1).standard_normal(50)
    assert NormalScores().evaluate(np.exp(x), y).statistic == pytest.approx(
        NormalScores().evaluate(x, y).statistic
    )


@pytest.mark.parametrize("alternative", list(Alternative))
@pytest.mark.parametrize("num_levels", [2, 4, 30])
def test_somers_d_matches_scipy(alternative: Alternative, num_levels: int) -> None:
    for seed in range(5):
        rng: np.random.Generator = np.random.default_rng(seed)
        x: NDArray[np.int64] = rng.integers(0, num_levels, 40)
        y: NDArray[np.int64] = x + rng.integers(0, num_levels, 40)
        expected = stats.somersd(x, y, alternative=alternative.option_name)
        result: CorrelationResult = SomersD(alternative=alternative).evaluate(x, y)
        assert result.statistic == pytest.approx(float(expected.statistic))
        assert result.p_value == pytest.approx(float(expected.pvalue))


@pytest.mark.parametrize(
    ("alternative", "p_value"),
    [(Alternative.TWO_SIDED, 0.0), (Alternative.GREATER, 0.0), (Alternative.LESS, 1.0)],
)
def test_somers_d_perfect_relationship_has_a_limit_p_value(
    alternative: Alternative, p_value: float
) -> None:
    result: CorrelationResult = SomersD(alternative=alternative).evaluate(
        [1, 2, 3, 4], [1, 2, 3, 4]
    )
    assert result.statistic == pytest.approx(1.0)
    assert result.p_value == pytest.approx(p_value)


def test_somers_d_without_any_spread_of_scores_has_no_p_value() -> None:
    result: CorrelationResult = SomersD().evaluate([0, 1, 1, 1, 2], [1, 0, 1, 2, 1])
    assert result.statistic == pytest.approx(0.0)
    assert result.p_value is None


def test_somers_d_without_ties_equals_kendall_tau_beyond_one_chunk() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.float64] = rng.standard_normal(3000)
    y: NDArray[np.float64] = x + rng.standard_normal(3000)
    statistic: float | None = SomersD().evaluate(x, y).statistic
    assert statistic == pytest.approx(KendallTau().evaluate(x, y).statistic, abs=1e-9)
