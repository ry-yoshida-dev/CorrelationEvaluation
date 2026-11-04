import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from correlation_evaluation import (
    AdjustedMutualInformation,
    ContingencyCoefficient,
    CramersV,
    EntropyAverage,
    GoodmanKruskalLambda,
    GoodmanKruskalTau,
    Matthews,
    MutualInformation,
    NormalizedMutualInformation,
    Phi,
    TheilsU,
    TschuprowT,
)

# Rows of x by columns of y: [[10, 0], [5, 5]]; chi-square without correction is 20/3.
X: list[int] = [0] * 10 + [1] * 10
Y: list[int] = [0] * 15 + [1] * 5
CHI_SQUARE: float = 20 / 3


def test_chi_square_associations_cramers_v() -> None:
    result = CramersV().evaluate(X, Y)
    assert result.statistic == pytest.approx(math.sqrt(CHI_SQUARE / 20))
    assert result.p_value == pytest.approx(float(stats.chi2.sf(CHI_SQUARE, 1)))


def test_chi_square_associations_tschuprow_t_equals_cramers_v_for_square_tables() -> (
    None
):
    assert TschuprowT().evaluate(X, Y).statistic == pytest.approx(
        CramersV().evaluate(X, Y).statistic
    )


def test_chi_square_associations_tschuprow_t_stays_below_one_for_rectangular_tables() -> (
    None
):
    x: list[int] = [0, 0, 1, 1, 2, 2]
    y: list[int] = [0, 0, 1, 1, 1, 1]
    statistic: float | None = TschuprowT().evaluate(x, y).statistic
    assert statistic is not None
    assert statistic < 1
    assert CramersV().evaluate(x, y).statistic == pytest.approx(1.0)


def test_chi_square_associations_contingency_coefficient() -> None:
    assert ContingencyCoefficient().evaluate(X, Y).statistic == pytest.approx(
        math.sqrt(CHI_SQUARE / (CHI_SQUARE + 20))
    )


def test_chi_square_associations_bias_correction_shrinks_the_null() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.int64] = rng.integers(0, 4, 100)
    y: NDArray[np.int64] = rng.integers(0, 4, 100)
    corrected = CramersV(is_bias_corrected=True).evaluate(x, y).statistic
    plain = CramersV().evaluate(x, y).statistic
    assert corrected is not None
    assert plain is not None
    assert corrected < plain


def test_chi_square_associations_bias_correction_keeps_perfect_association() -> None:
    x: NDArray[np.int64] = np.repeat(np.arange(3), 30)
    assert CramersV(is_bias_corrected=True).evaluate(x, x).statistic == pytest.approx(
        1.0
    )


def test_phi_is_signed_and_matches_chi_square() -> None:
    assert Phi().evaluate(X, Y).statistic == pytest.approx(math.sqrt(CHI_SQUARE / 20))
    assert Phi().evaluate(X, [1 - v for v in Y]).statistic == pytest.approx(
        -math.sqrt(CHI_SQUARE / 20)
    )
    assert Phi().evaluate(X, Y).p_value == pytest.approx(
        float(stats.chi2.sf(CHI_SQUARE, 1))
    )


def test_matthews_equals_phi_for_two_classes() -> None:
    assert Matthews().evaluate(X, Y).statistic == pytest.approx(
        Phi().evaluate(X, Y).statistic
    )


def test_matthews_labels_are_matched_across_variables() -> None:
    # y never uses "a", so separate encoding would misalign the labels.
    result = Matthews().evaluate(["b", "c", "b", "c"], ["b", "c", "b", "c"])
    assert result.statistic == pytest.approx(1.0)
    partial = Matthews().evaluate(["a", "b", "c", "c"], ["b", "b", "c", "c"])
    assert partial.statistic is not None
    assert partial.statistic < 1


def test_information_metrics_theils_u_is_directional() -> None:
    # y is a function of x, so x removes all of the entropy of y, but not vice versa.
    x: list[int] = [0, 1, 2, 3, 0, 1, 2, 3]
    y: list[int] = [0, 0, 1, 1, 0, 0, 1, 1]
    assert TheilsU().evaluate(x, y).statistic == pytest.approx(1.0)
    assert TheilsU().evaluate(y, x).statistic == pytest.approx(0.5)


def test_information_metrics_mutual_information_in_nats() -> None:
    x: list[int] = [0, 0, 1, 1]
    assert MutualInformation().evaluate(x, x).statistic == pytest.approx(math.log(2))


def test_information_metrics_mutual_information_of_a_constant_is_zero() -> None:
    assert MutualInformation().evaluate([0, 1], [1, 1]).statistic == pytest.approx(0.0)


def test_information_metrics_normalized_mutual_information_uses_the_chosen_average() -> (
    None
):
    x: list[int] = [0, 1, 2, 3, 0, 1, 2, 3]
    y: list[int] = [0, 0, 1, 1, 0, 0, 1, 1]
    minimum = NormalizedMutualInformation(average=EntropyAverage.MIN)
    maximum = NormalizedMutualInformation(average=EntropyAverage.MAX)
    assert minimum.evaluate(x, y).statistic == pytest.approx(1.0)
    assert maximum.evaluate(x, y).statistic == pytest.approx(0.5)


def test_information_metrics_adjusted_mutual_information_is_near_zero_under_independence() -> (
    None
):
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.int64] = rng.integers(0, 4, 200)
    y: NDArray[np.int64] = rng.integers(0, 4, 200)
    assert AdjustedMutualInformation().evaluate(x, y).statistic == pytest.approx(
        0.0, abs=0.02
    )


def test_goodman_kruskal_lambda_hand_computed_value() -> None:
    # Without x, 5 errors; with x, 0 errors from x = 0 and 5 from x = 1.
    assert GoodmanKruskalLambda().evaluate(X, Y).statistic == pytest.approx(0.0)
    assert GoodmanKruskalLambda().evaluate(Y, X).statistic == pytest.approx(
        (15 - 10 + 5 - 5) / 10
    )


def test_goodman_kruskal_tau_hand_computed_value() -> None:
    marginal_error: float = 1 - (0.75**2 + 0.25**2)
    conditional_error: float = 0.5 * 0.0 + 0.5 * 0.5
    assert GoodmanKruskalTau().evaluate(X, Y).statistic == pytest.approx(
        (marginal_error - conditional_error) / marginal_error
    )


def test_goodman_kruskal_perfect_prediction_is_one() -> None:
    x: list[int] = [0, 0, 1, 1, 2, 2]
    y: list[int] = [0, 0, 1, 1, 1, 1]
    assert GoodmanKruskalLambda().evaluate(x, y).statistic == pytest.approx(1.0)
    assert GoodmanKruskalTau().evaluate(x, y).statistic == pytest.approx(1.0)
