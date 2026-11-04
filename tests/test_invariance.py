from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

import correlation_evaluation as ce
from correlation_evaluation import CorrelationMetric, CorrelationResult, VariableKind
from correlation_evaluation.computation import ColumnSpace

type FloatMatrix = NDArray[np.float64]

NUM_SAMPLES: int = 60
COVARIATES: FloatMatrix = np.random.default_rng(7).standard_normal(NUM_SAMPLES)

# The likelihood of latent metrics is maximized only to this accuracy.
SYMMETRY_TOLERANCE: float = 1e-6
METRICS: list[CorrelationMetric] = [
    ce.Pearson(),
    ce.Cosine(),
    ce.LaggedPearson(),
    ce.Partial(COVARIATES),
    ce.Semipartial(COVARIATES),
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
    ce.MultiscaleGraph(),
    ce.NearestNeighborMutualInformation(),
    ce.RVCoefficient(),
    ce.Canonical(),
    ce.LinConcordance(),
    ce.Intraclass(),
    ce.PointBiserial(),
    ce.RankBiserial(),
    ce.CorrelationRatio(),
    ce.Biserial(),
    ce.Polyserial(),
    ce.Tetrachoric(),
    ce.Polychoric(),
    ce.Phi(),
    ce.Matthews(),
    ce.CramersV(),
    ce.TschuprowT(),
    ce.ContingencyCoefficient(),
    ce.TheilsU(),
    ce.GoodmanKruskalLambda(),
    ce.GoodmanKruskalTau(),
    ce.MutualInformation(),
    ce.NormalizedMutualInformation(),
    ce.AdjustedMutualInformation(),
    ce.CohenKappa(),
]
COVARIATE_TRANSFORMS: list[Callable[[FloatMatrix], FloatMatrix]] = [
    lambda z: 100.0 * z + 1.7e9,
    lambda z: z + 1e6,
    lambda z: 1e9 * z,
    lambda z: 1e-9 * z,
    lambda z: -2.0 * z,
]


def metric_name(metric: CorrelationMetric) -> str:
    return type(metric).__name__


def sample(kind: VariableKind, seed: int) -> FloatMatrix:
    rng: np.random.Generator = np.random.default_rng(seed)
    match kind:
        case VariableKind.NUMERIC | VariableKind.MULTIVARIATE:
            return rng.standard_normal(NUM_SAMPLES)
        case VariableKind.BINARY:
            return rng.integers(0, 2, NUM_SAMPLES).astype(np.float64)
        case VariableKind.CATEGORICAL:
            return rng.integers(0, 4, NUM_SAMPLES).astype(np.float64)


def correlated_pair(metric: CorrelationMetric) -> tuple[FloatMatrix, FloatMatrix]:
    x: FloatMatrix = sample(metric.x_kind, 1)
    y: FloatMatrix = sample(metric.y_kind, 2)
    if metric.y_kind.is_numeric:
        y = y + x
    return x, y


def statistic_of(result: CorrelationResult) -> float:
    assert result.statistic is not None
    return result.statistic


@pytest.mark.parametrize(
    "metric",
    [m for m in METRICS if m.is_symmetric and m.x_kind is m.y_kind],
    ids=metric_name,
)
def test_symmetric_metric_ignores_the_order_of_the_variables(
    metric: CorrelationMetric,
) -> None:
    x: FloatMatrix
    y: FloatMatrix
    x, y = correlated_pair(metric)
    assert statistic_of(metric.evaluate(x, y)) == pytest.approx(
        statistic_of(metric.evaluate(y, x)), abs=SYMMETRY_TOLERANCE
    )


@pytest.mark.parametrize(
    "metric",
    [
        m
        for m in METRICS
        if not isinstance(m, ce.LaggedPearson | ce.Partial | ce.Semipartial)
    ],
    ids=metric_name,
)
def test_metric_ignores_the_order_of_the_samples(metric: CorrelationMetric) -> None:
    x: FloatMatrix
    y: FloatMatrix
    x, y = correlated_pair(metric)
    order: NDArray[np.int64] = np.random.default_rng(3).permutation(NUM_SAMPLES)
    assert statistic_of(metric.evaluate(x, y)) == pytest.approx(
        statistic_of(metric.evaluate(x[order], y[order])), abs=1e-9
    )


@pytest.mark.parametrize(
    "metric",
    [
        m
        for m in METRICS
        if m.x_kind.is_numeric
        and m.y_kind.is_numeric
        and not isinstance(m, ce.Cosine | ce.LinConcordance | ce.Intraclass)
    ],
    ids=metric_name,
)
def test_metric_ignores_affine_changes_of_units(metric: CorrelationMetric) -> None:
    x: FloatMatrix
    y: FloatMatrix
    x, y = correlated_pair(metric)
    assert statistic_of(metric.evaluate(x, y)) == pytest.approx(
        statistic_of(metric.evaluate(3.0 * x + 10.0, 0.5 * y - 4.0)), abs=1e-6
    )


@pytest.mark.parametrize("transform", COVARIATE_TRANSFORMS)
@pytest.mark.parametrize("metric_type", [ce.Partial, ce.Semipartial])
def test_covariate_adjusted_ignores_the_units_of_covariates(
    metric_type: type[ce.Partial] | type[ce.Semipartial],
    transform: Callable[[FloatMatrix], FloatMatrix],
) -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    z: FloatMatrix = rng.standard_normal(200)
    x: FloatMatrix = z + 0.3 * rng.standard_normal(200)
    y: FloatMatrix = z + 0.3 * rng.standard_normal(200)
    expected: CorrelationResult = metric_type(z).evaluate(x, y)
    actual: CorrelationResult = metric_type(transform(z)).evaluate(x, y)
    assert actual.statistic == pytest.approx(expected.statistic, abs=1e-6)
    assert actual.p_value == pytest.approx(expected.p_value, abs=1e-6)


def test_covariate_adjusted_ignores_the_relative_units_of_covariates() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    z: FloatMatrix = rng.standard_normal((200, 2))
    x: FloatMatrix = z.sum(axis=1) + 0.3 * rng.standard_normal(200)
    y: FloatMatrix = z.sum(axis=1) + 0.3 * rng.standard_normal(200)
    assert ce.Partial(z * [1e-9, 1e9]).evaluate(x, y).statistic == pytest.approx(
        ce.Partial(z).evaluate(x, y).statistic, abs=1e-9
    )


def test_covariate_adjusted_removes_a_timestamp_covariate() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    timestamps: FloatMatrix = 1.7e9 + 60.0 * np.arange(200.0)
    trend: FloatMatrix = (timestamps - timestamps.mean()) / timestamps.std()
    x: FloatMatrix = trend + rng.standard_normal(200)
    y: FloatMatrix = trend + rng.standard_normal(200)
    assert ce.Partial(timestamps).evaluate(x, y).statistic == pytest.approx(
        ce.Partial(trend).evaluate(x, y).statistic, abs=1e-6
    )


@pytest.mark.parametrize("scale", [1e-16, 1e-9, 1e9])
def test_canonical_ignores_the_units_of_each_feature(scale: float) -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    x: FloatMatrix = rng.standard_normal((200, 2))
    y: FloatMatrix = x[:, [1]] + 0.5 * rng.standard_normal((200, 1))
    assert ce.Canonical().evaluate(x * [1.0, scale], y).statistic == pytest.approx(
        ce.Canonical().evaluate(x, y).statistic, abs=1e-9
    )


def test_canonical_of_rounding_noise_alone_is_undefined() -> None:
    x: FloatMatrix = np.array([1.0, 1.0 + 2.0**-52, 1.0, 1.0, 1.0, 1.0])
    y: FloatMatrix = np.arange(6.0)
    assert ce.Canonical().evaluate(x, y).statistic is None


def test_column_space_counts_collinear_and_constant_columns_once() -> None:
    z: FloatMatrix = np.random.default_rng(0).standard_normal(50)
    values: FloatMatrix = np.column_stack([z, 2.0 * z + 1.0, np.full(50, 3.0)])
    assert np.size(ColumnSpace.orthonormal_basis(values), axis=1) == 1


def test_column_space_of_constant_columns_is_empty() -> None:
    values: FloatMatrix = np.full((5, 2), 0.1)
    assert np.size(ColumnSpace.orthonormal_basis(values), axis=1) == 0


def test_result_undefined_statistic_clears_the_test() -> None:
    interval: ce.ConfidenceInterval = ce.ConfidenceInterval(0.0, 1.0, 0.95)
    for statistic in (None, float("nan")):
        result: CorrelationResult = CorrelationResult(
            statistic=statistic, p_value=0.5, confidence_interval=interval
        )
        assert result.statistic is None
        assert result.p_value is None
        assert result.confidence_interval is None


def test_bias_corrected_cramers_v_undefined_has_no_p_value() -> None:
    result: CorrelationResult = ce.CramersV(is_bias_corrected=True).evaluate(
        [0, 1, 2], [0, 1, 2]
    )
    assert result.statistic is None
    assert result.p_value is None
