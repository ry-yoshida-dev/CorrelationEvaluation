import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score

from correlation_evaluation import (
    CohenKappa,
    Intraclass,
    IntraclassForm,
    KappaWeighting,
    LinConcordance,
    Pearson,
)


def test_lin_concordance_hand_computed_value() -> None:
    # Means 2 and 3, variances 2/3 each, covariance 2/3.
    result = LinConcordance().evaluate([1, 2, 3], [2, 3, 4])
    assert result.statistic == pytest.approx(2 * (2 / 3) / (2 / 3 + 2 / 3 + 1))


def test_lin_concordance_identical_variables_are_perfect() -> None:
    assert LinConcordance().evaluate([1, 2, 4], [1, 2, 4]).statistic == pytest.approx(1)


def test_lin_concordance_penalizes_what_pearson_ignores() -> None:
    x: NDArray[np.float64] = np.arange(10.0)
    y: NDArray[np.float64] = 2 * x + 5
    concordance: float | None = LinConcordance().evaluate(x, y).statistic
    assert concordance is not None
    assert Pearson().evaluate(x, y).statistic == pytest.approx(1.0)
    assert concordance < 0.5


def test_cohen_kappa_hand_computed_value() -> None:
    # Observed agreement 3/4; chance agreement 1/2 * 3/4 + 1/2 * 1/4 = 1/2.
    result = CohenKappa().evaluate([0, 0, 1, 1], [0, 0, 1, 0])
    assert result.statistic == pytest.approx(0.5)


def test_cohen_kappa_weights_reach_the_backing_function() -> None:
    x: list[int] = [0, 1, 2, 2, 1, 0]
    y: list[int] = [0, 2, 2, 1, 1, 0]
    assert CohenKappa(weighting=KappaWeighting.QUADRATIC).evaluate(
        x, y
    ).statistic == pytest.approx(
        cohen_kappa_score(np.array(x), np.array(y), weights="quadratic")
    )


def test_cohen_kappa_labels_are_matched_across_raters() -> None:
    # Separate encoding would map both to 0, 1 and report perfect agreement.
    assert CohenKappa().evaluate([1, 2, 1, 2], [2, 3, 2, 3]).statistic != pytest.approx(
        1
    )


def test_cohen_kappa_one_constant_rater_is_zero() -> None:
    assert CohenKappa().evaluate([0, 0, 0, 0], [0, 1, 0, 1]).statistic == pytest.approx(
        0
    )


def test_cohen_kappa_two_identical_constant_raters_are_undefined() -> None:
    assert CohenKappa().evaluate([1, 1, 1], [1, 1, 1]).statistic is None


@pytest.mark.parametrize("form", list(IntraclassForm))
def test_intraclass_identical_ratings_are_perfect(form: IntraclassForm) -> None:
    ratings: list[int] = [1, 3, 2, 5, 4]
    assert Intraclass(form=form).evaluate(ratings, ratings).statistic == pytest.approx(
        1
    )


def test_intraclass_consistency_ignores_a_shift_but_agreement_does_not() -> None:
    x: NDArray[np.float64] = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    y: NDArray[np.float64] = x + 10
    assert Intraclass(form=IntraclassForm.TWO_WAY_MIXED).evaluate(
        x, y
    ).statistic == pytest.approx(1.0)
    agreement: float | None = (
        Intraclass(form=IntraclassForm.TWO_WAY_RANDOM).evaluate(x, y).statistic
    )
    assert agreement is not None
    assert agreement < 0.1
