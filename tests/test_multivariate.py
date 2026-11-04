import numpy as np
import pytest
from numpy.typing import NDArray

from correlation_evaluation import Canonical, Pearson, RVCoefficient


def test_r_v_coefficient_equals_squared_pearson_for_vectors() -> None:
    x: list[int] = [1, 2, 3, 4, 5]
    y: list[int] = [1, 3, 2, 5, 4]
    pearson: float | None = Pearson().evaluate(x, y).statistic
    assert pearson is not None
    assert RVCoefficient().evaluate(x, y).statistic == pytest.approx(pearson**2)


def test_r_v_coefficient_ignores_rotation() -> None:
    x: NDArray[np.float64] = np.random.default_rng(0).standard_normal((30, 2))
    rotation: NDArray[np.float64] = np.array([[0.6, -0.8], [0.8, 0.6]])
    assert RVCoefficient().evaluate(x, x @ rotation).statistic == pytest.approx(1.0)


def test_canonical_equals_absolute_pearson_for_vectors() -> None:
    x: list[int] = [1, 2, 3, 4, 5]
    y: list[int] = [5, 3, 4, 1, 2]
    pearson: float | None = Pearson().evaluate(x, y).statistic
    assert pearson is not None
    assert Canonical().evaluate(x, y).statistic == pytest.approx(abs(pearson))


def test_canonical_finds_the_shared_direction() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    shared: NDArray[np.float64] = rng.standard_normal(500)
    x: NDArray[np.float64] = np.column_stack([shared, rng.standard_normal(500)])
    y: NDArray[np.float64] = np.column_stack([rng.standard_normal(500), shared])
    assert Canonical().evaluate(x, y).statistic == pytest.approx(1.0)


def test_canonical_handles_duplicate_columns() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.float64] = rng.standard_normal(50)
    y: NDArray[np.float64] = x + rng.standard_normal(50)
    expected: float | None = Canonical().evaluate(x, y).statistic
    assert Canonical().evaluate(np.column_stack([x, x]), y).statistic == pytest.approx(
        expected
    )


def test_canonical_too_many_features_is_undefined() -> None:
    x: NDArray[np.float64] = np.random.default_rng(0).standard_normal((4, 2))
    assert Canonical().evaluate(x, x[:, ::-1]).statistic is None


def test_r_v_coefficient_underflow_is_undefined() -> None:
    values: NDArray[np.float64] = np.array([1.0, 2.0, 3.0]) * 1e-200
    assert RVCoefficient().evaluate(values, values).statistic is None
