import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from correlation_evaluation import Biserial, Polychoric, Polyserial, Tetrachoric
from correlation_evaluation.metrics.latent.bivariate_normal import BivariateNormal


def expand_table(counts: list[list[int]]) -> tuple[list[int], list[int]]:
    x: list[int] = []
    y: list[int] = []
    for i, row in enumerate(counts):
        for j, count in enumerate(row):
            x.extend([i] * count)
            y.extend([j] * count)
    return x, y


@pytest.mark.parametrize("correlation", [-0.9, -0.3, 0.0, 0.5, 0.95])
def test_bivariate_normal_matches_scipy(correlation: float) -> None:
    limits: NDArray[np.float64] = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    h: NDArray[np.float64]
    k: NDArray[np.float64]
    h, k = np.meshgrid(limits, limits, indexing="ij")
    distribution = stats.multivariate_normal(
        cov=[[1.0, correlation], [correlation, 1.0]]
    )
    expected: NDArray[np.float64] = np.asarray(distribution.cdf(np.dstack([h, k])))
    assert BivariateNormal.cdf(h, k, correlation) == pytest.approx(expected, abs=1e-4)


def test_tetrachoric_balanced_table_has_a_closed_form() -> None:
    # With median splits, P(both low) = 1/4 + arcsin(r) / (2 pi).
    x, y = expand_table([[40, 10], [10, 40]])
    assert Tetrachoric().evaluate(x, y).statistic == pytest.approx(
        math.sin(math.pi / 2 * 0.6), abs=1e-4
    )


def test_tetrachoric_independent_table_is_zero() -> None:
    x, y = expand_table([[30, 10], [60, 20]])
    assert Tetrachoric().evaluate(x, y).statistic == pytest.approx(0.0, abs=1e-4)


def test_tetrachoric_empty_cell_reaches_the_boundary() -> None:
    x, y = expand_table([[30, 0], [10, 30]])
    statistic: float | None = Tetrachoric().evaluate(x, y).statistic
    assert statistic is not None
    assert statistic > 0.99


def test_polychoric_recovers_the_latent_correlation() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    first: NDArray[np.float64] = rng.standard_normal(20000)
    second: NDArray[np.float64] = 0.6 * first + math.sqrt(
        1 - 0.6**2
    ) * rng.standard_normal(20000)
    x: NDArray[np.int64] = np.digitize(first, [-1.0, 0.0, 1.0])
    y: NDArray[np.int64] = np.digitize(second, [-0.5, 0.5])
    assert Polychoric().evaluate(x, y).statistic == pytest.approx(0.6, abs=0.02)


def test_polychoric_reversing_levels_flips_the_sign() -> None:
    x, y = expand_table([[20, 5, 1], [5, 20, 5], [1, 5, 20]])
    forward: float | None = Polychoric().evaluate(x, y).statistic
    backward: float | None = Polychoric().evaluate(x, [2 - v for v in y]).statistic
    assert forward is not None
    assert backward == pytest.approx(-forward, abs=1e-4)


def test_biserial_matches_the_textbook_formula() -> None:
    x: NDArray[np.int64] = np.array([0, 0, 0, 0, 1, 1, 1])
    y: NDArray[np.float64] = np.array([1.0, 2.0, 2.5, 3.0, 2.8, 4.0, 5.0])
    proportion: float = 3 / 7
    expected: float = (
        (y[x == 1].mean() - y[x == 0].mean())
        / y.std()
        * proportion
        * (1 - proportion)
        / float(stats.norm.pdf(stats.norm.ppf(proportion)))
    )
    assert Biserial().evaluate(x, y).statistic == pytest.approx(expected)


def test_polyserial_recovers_the_latent_correlation() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    first: NDArray[np.float64] = rng.standard_normal(20000)
    second: NDArray[np.float64] = 0.7 * first + math.sqrt(
        1 - 0.7**2
    ) * rng.standard_normal(20000)
    x: NDArray[np.int64] = np.digitize(first, [-1.0, 0.0, 1.0])
    assert Polyserial().evaluate(x, second).statistic == pytest.approx(0.7, abs=0.02)
