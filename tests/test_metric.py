import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import correlation_evaluation as ce
from correlation_evaluation import CorrelationMetric, CorrelationResult, VariableKind

NUMERIC_METRICS: list[CorrelationMetric] = [
    ce.Pearson(),
    ce.LaggedPearson(),
    ce.Partial(np.arange(30.0) ** 2),
    ce.Semipartial(np.arange(30.0) ** 2),
    ce.BiweightMidcorrelation(),
    ce.PercentageBend(),
    ce.Winsorized(),
    ce.Spearman(),
    ce.KendallTau(),
    ce.WeightedTau(),
    ce.SomersD(),
    ce.GoodmanKruskalGamma(),
    ce.BlomqvistBeta(),
    ce.NormalScores(),
    ce.ChatterjeeXi(),
    ce.HoeffdingD(),
    ce.DistanceCorrelation(),
    ce.HilbertSchmidt(),
    ce.NearestNeighborMutualInformation(),
    ce.RVCoefficient(),
    ce.Canonical(),
    ce.LinConcordance(),
    ce.Intraclass(),
]
BINARY_PAIR_METRICS: list[CorrelationMetric] = [ce.Phi(), ce.Tetrachoric()]
CATEGORICAL_METRICS: list[CorrelationMetric] = [
    ce.Polychoric(),
    ce.Matthews(),
    ce.CramersV(),
    ce.TschuprowT(),
    ce.ContingencyCoefficient(),
    ce.TheilsU(),
    ce.GoodmanKruskalLambda(),
    ce.GoodmanKruskalTau(),
    ce.NormalizedMutualInformation(),
    ce.AdjustedMutualInformation(),
]
MIXED_METRICS: list[CorrelationMetric] = [
    ce.PointBiserial(),
    ce.RankBiserial(),
    ce.CorrelationRatio(),
    ce.Biserial(),
    ce.Polyserial(),
]


def metric_name(metric: CorrelationMetric) -> str:
    return type(metric).__name__


def matrix_entry(matrix: NDArray[np.float64], row: int, column: int) -> float:
    return float(np.take(matrix, row * np.size(matrix, axis=1) + column))


def correlated_sample() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.float64] = rng.standard_normal(30)
    return x, x + 0.5 * rng.standard_normal(30)


@pytest.mark.parametrize("metric", NUMERIC_METRICS, ids=metric_name)
def test_every_metric_numeric_metric_gives_finite_value(
    metric: CorrelationMetric,
) -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x, y = correlated_sample()
    result: CorrelationResult = metric.evaluate(x, y)
    assert result.statistic is not None
    assert math.isfinite(result.statistic)


@pytest.mark.parametrize("metric", NUMERIC_METRICS, ids=metric_name)
def test_every_metric_constant_numeric_input_is_undefined(
    metric: CorrelationMetric,
) -> None:
    x: NDArray[np.float64]
    x, _ = correlated_sample()
    assert metric.evaluate(x, np.ones(30)).statistic is None
    assert metric.evaluate(np.ones(30), x).statistic is None


@pytest.mark.parametrize("metric", CATEGORICAL_METRICS, ids=metric_name)
def test_every_metric_categorical_metric_accepts_strings(
    metric: CorrelationMetric,
) -> None:
    x: list[str] = ["a", "a", "b", "b", "c", "c", "a", "b"]
    y: list[str] = ["a", "a", "b", "c", "c", "c", "b", "b"]
    result: CorrelationResult = metric.evaluate(x, y)
    assert result.statistic is not None
    assert math.isfinite(result.statistic)
    assert metric.evaluate(x, ["a"] * 8).statistic is None


@pytest.mark.parametrize("metric", BINARY_PAIR_METRICS, ids=metric_name)
def test_every_metric_binary_metric_rejects_three_levels(
    metric: CorrelationMetric,
) -> None:
    with pytest.raises(ValueError, match="at most 2 levels"):
        _ = metric.evaluate([0, 1, 2, 0], [0, 1, 1, 0])


@pytest.mark.parametrize("metric", MIXED_METRICS, ids=metric_name)
def test_every_metric_mixed_metric_has_no_matrix_form(
    metric: CorrelationMetric,
) -> None:
    with pytest.raises(TypeError, match="different kinds"):
        _ = metric.evaluate_matrix(np.zeros((5, 2)))


def test_validation_misaligned_variables_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        _ = ce.Pearson().evaluate([1.0, 2.0, 3.0], [1.0, 2.0])


def test_validation_misaligned_shared_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        _ = ce.CohenKappa().evaluate([1, 2, 3], [1, 2])


def test_validation_empty_variable_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _ = ce.Pearson().evaluate([], [])


def test_validation_nan_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _ = ce.Pearson().evaluate([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be finite"):
        _ = ce.CramersV().evaluate([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])


def test_validation_strings_are_rejected_for_numeric_metrics() -> None:
    with pytest.raises(TypeError, match="must contain numbers"):
        _ = ce.Pearson().evaluate(["a", "b", "c"], [1.0, 2.0, 3.0])


def test_validation_two_dimensional_input_is_rejected_for_univariate_metrics() -> None:
    with pytest.raises(ValueError, match="unsupported shape"):
        _ = ce.Pearson().evaluate(np.zeros((3, 2)), np.zeros(3))


def test_validation_too_few_samples_is_undefined() -> None:
    assert ce.Pearson().evaluate([1.0, 2.0], [1.0, 2.0]).statistic is None
    assert ce.HoeffdingD().evaluate([1, 2, 3, 4], [1, 2, 3, 4]).statistic is None


def test_validation_nan_statistic_is_stored_as_none() -> None:
    result: CorrelationResult = CorrelationResult(statistic=math.nan, p_value=math.nan)
    assert result.statistic is None
    assert result.p_value is None


@pytest.mark.parametrize("value", [math.nan, 0.0, 1.0])
def test_validation_invalid_confidence_level_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        _ = ce.Pearson(confidence_level=value)


def test_validation_non_integer_parameter_is_rejected() -> None:
    with pytest.raises(TypeError, match="num_neighbors"):
        _ = ce.NearestNeighborMutualInformation(num_neighbors=2.5)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_encoding_labels_become_sorted_codes() -> None:
    codes: NDArray[np.float64] = VariableKind.CATEGORICAL.encode(
        ["b", "a", "c", "a"], "x"
    )
    assert codes.tolist() == [1.0, 0.0, 2.0, 0.0]


def test_encoding_multivariate_vector_becomes_one_column() -> None:
    assert VariableKind.MULTIVARIATE.encode([1.0, 2.0], "x").shape == (2, 1)


def test_encoding_booleans_are_labels_not_numbers() -> None:
    assert VariableKind.BINARY.encode([True, False], "x").tolist() == [1.0, 0.0]
    with pytest.raises(TypeError, match="must contain numbers"):
        _ = VariableKind.NUMERIC.encode([True, False], "x")


def test_matrix_symmetric_matrix_matches_pairwise_evaluation() -> None:
    X: NDArray[np.float64] = np.random.default_rng(0).standard_normal((20, 3))
    matrix: ce.CorrelationMatrix = ce.Pearson().evaluate_matrix(X)
    assert matrix.statistics == pytest.approx(np.corrcoef(X, rowvar=False))
    assert matrix.statistics == pytest.approx(matrix.statistics.T)
    pair: CorrelationResult = ce.Pearson().evaluate(X[:, 0], X[:, 2])
    assert matrix.p_values[2, 0] == pytest.approx(pair.p_value)


def test_matrix_asymmetric_matrix_keeps_direction() -> None:
    x: NDArray[np.float64] = np.arange(10.0)
    X: NDArray[np.float64] = np.column_stack([x, np.floor(x / 3)])
    statistics: NDArray[np.float64] = ce.SomersD().evaluate_matrix(X).statistics
    assert statistics[0, 1] == pytest.approx(
        ce.SomersD().evaluate(X[:, 0], X[:, 1]).statistic
    )
    assert statistics[1, 0] == pytest.approx(
        ce.SomersD().evaluate(X[:, 1], X[:, 0]).statistic
    )
    assert matrix_entry(statistics, 0, 1) != pytest.approx(
        matrix_entry(statistics, 1, 0)
    )


def test_matrix_undefined_entries_are_nan() -> None:
    X: NDArray[np.float64] = np.column_stack([np.arange(5.0), np.ones(5)])
    statistics: NDArray[np.float64] = ce.Pearson().evaluate_matrix(X).statistics
    assert math.isnan(matrix_entry(statistics, 0, 1))
    assert math.isnan(matrix_entry(statistics, 1, 1))
    assert statistics[0, 0] == pytest.approx(1.0)


def test_matrix_string_columns_are_supported() -> None:
    X: NDArray[np.str_] = np.array([["a", "x"], ["a", "x"], ["b", "y"], ["b", "y"]])
    statistics: NDArray[np.float64] = ce.CramersV().evaluate_matrix(X).statistics
    assert statistics[0, 1] == pytest.approx(1.0)


def test_matrix_one_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        _ = ce.Pearson().evaluate_matrix(np.arange(4.0))


def test_encoding_object_strings_equal_string_labels() -> None:
    x: list[str] = ["a", "b", "a", "b", "a"]
    y: list[str] = ["x", "x", "y", "y", "x"]
    from_objects: float | None = (
        ce.CramersV()
        .evaluate(np.array(x, dtype=object), np.array(y, dtype=object))
        .statistic
    )
    assert from_objects == ce.CramersV().evaluate(x, y).statistic


def test_encoding_object_columns_are_labels_in_a_matrix() -> None:
    X: NDArray[np.object_] = np.array(
        [["a", 1], ["b", 2], ["a", 1], ["b", 2]], dtype=object
    )
    statistics: NDArray[np.float64] = ce.CramersV().evaluate_matrix(X).statistics
    assert matrix_entry(statistics, 0, 1) == pytest.approx(1.0)


def test_encoding_shared_labels_accept_objects_with_strings() -> None:
    labels: list[str] = ["a", "b", "a", "b"]
    from_objects: float | None = (
        ce.Matthews().evaluate(np.array(labels, dtype=object), labels).statistic
    )
    assert from_objects == pytest.approx(1.0)


@pytest.mark.parametrize("missing", [None, math.nan])
def test_encoding_missing_object_label_is_rejected(missing: float | None) -> None:
    with pytest.raises(TypeError, match="labels"):
        _ = ce.CramersV().evaluate(
            np.array(["a", "b", missing, "a"], dtype=object), [0, 1, 0, 1]
        )


def test_encoding_all_nan_object_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        _ = ce.CramersV().evaluate(np.array([math.nan] * 3, dtype=object), [0, 1, 0])


def test_encoding_all_none_object_labels_are_rejected() -> None:
    with pytest.raises(TypeError, match="labels"):
        _ = ce.CramersV().evaluate(np.array([None] * 3, dtype=object), [0, 1, 0])


@pytest.mark.parametrize(
    ("factory", "name"),
    [
        (lambda: ce.Pearson(alternative="greater"), "alternative"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.LaggedPearson(alternative="less"), "alternative"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.Spearman(alternative="greater"), "alternative"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.SomersD(alternative="greater"), "alternative"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.RankBiserial(alternative="greater"), "alternative"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.KendallTau(variant="b"), "variant"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.KendallTau(alternative="greater"), "alternative"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.Intraclass(form="one_way_random"), "form"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.CohenKappa(weighting="linear"), "weighting"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.NormalizedMutualInformation(average="min"), "average"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        (lambda: ce.AdjustedMutualInformation(average="min"), "average"),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    ],
)
def test_validation_non_member_option_is_rejected_at_construction(
    factory: Callable[[], CorrelationMetric], name: str
) -> None:
    with pytest.raises(TypeError, match=name):
        _ = factory()


def test_rounding_past_one_is_corrected_for_bounded_metrics() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    for size in range(4, 120):
        labels: NDArray[np.int64] = rng.integers(0, 3, size=size)
        result: float | None = ce.TheilsU().evaluate(labels, labels).statistic
        assert result is None or abs(result) <= 1


def test_unbounded_metric_can_exceed_one() -> None:
    assert ce.Intraclass().evaluate(
        [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]
    ).statistic == pytest.approx(-3.0)


def test_identity_comparison_of_array_holders() -> None:
    X: NDArray[np.float64] = np.arange(8.0).reshape(4, 2) ** 2
    matrix: ce.CorrelationMatrix = ce.Pearson().evaluate_matrix(X)
    assert matrix == matrix
    assert matrix != ce.Pearson().evaluate_matrix(X)
    table: ce.ContingencyTable = ce.ContingencyTable.from_codes(
        np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    assert table != ce.ContingencyTable.from_codes(
        np.array([0.0, 1.0]), np.array([1.0, 0.0])
    )
    assert len({matrix, matrix, table}) == 2


def test_identity_comparison_of_covariate_adjusted_metrics() -> None:
    first: ce.Partial = ce.Partial(covariates=np.arange(5.0))
    assert first == first
    assert first != ce.Partial(covariates=np.arange(5.0) ** 2)


def test_t_test_of_another_correlation() -> None:
    result: CorrelationResult = CorrelationResult.from_t_test(
        0.1, num_samples=20, degrees_of_freedom=18, tested_correlation=0.5
    )
    assert result.statistic == 0.1
    assert (
        result.p_value
        == CorrelationResult.from_t_test(
            0.5, num_samples=20, degrees_of_freedom=18
        ).p_value
    )


def test_t_test_without_degrees_of_freedom_has_no_p_value() -> None:
    assert (
        CorrelationResult.from_t_test(0.5, num_samples=3, degrees_of_freedom=0).p_value
        is None
    )


def test_t_test_of_perfect_correlation_has_zero_p_value() -> None:
    assert (
        CorrelationResult.from_t_test(1.0, num_samples=5, degrees_of_freedom=3).p_value
        == 0.0
    )


@pytest.mark.parametrize("metric", [ce.Matthews(), ce.CohenKappa()])
@pytest.mark.parametrize("numbers", [[1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]])
def test_validation_shared_labels_reject_numbers_with_strings(
    metric: CorrelationMetric, numbers: list[int] | list[float]
) -> None:
    with pytest.raises(TypeError, match="labels of one type"):
        _ = metric.evaluate(numbers, ["1", "0", "1", "0"])
