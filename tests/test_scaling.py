from collections.abc import Iterator

import numpy as np
import pytest
from data_scaler import (
    FeatureScaling,
    PowerScalingParameters,
    RobustScalingParameters,
    ScalingKind,
    ScalingParameters,
)
from numpy.typing import ArrayLike, NDArray

import correlation_evaluation as ce
from correlation_evaluation import (
    CorrelationMetric,
    CorrelationResult,
    VariableKind,
)

METHOD_OF_KIND: dict[ScalingKind, FeatureScaling | ScalingParameters] = {
    ScalingKind.IDENTITY: FeatureScaling.NONE,
    ScalingKind.PROPORTIONAL: FeatureScaling.MAX_ABS,
    ScalingKind.AFFINE: FeatureScaling.STANDARD,
    ScalingKind.MONOTONIC: PowerScalingParameters(),
    ScalingKind.SAMPLE_WISE: FeatureScaling.UNIT_NORM,
}
NUM_SAMPLES: int = 40


def concrete_metrics(
    parent: type[CorrelationMetric] = CorrelationMetric,
) -> Iterator[type[CorrelationMetric]]:
    for metric_class in parent.__subclasses__():
        yield from concrete_metrics(metric_class)
        is_package_metric: bool = metric_class.__module__.startswith(
            "correlation_evaluation"
        )
        if is_package_metric and not metric_class.__abstractmethods__:
            yield metric_class


def build_metric(metric_class: type[CorrelationMetric]) -> CorrelationMetric:
    if metric_class is ce.Partial:
        return ce.Partial(np.exp(np.linspace(0.0, 2.0, NUM_SAMPLES)))
    if metric_class is ce.Semipartial:
        return ce.Semipartial(np.exp(np.linspace(0.0, 2.0, NUM_SAMPLES)))
    return metric_class()


def sample_of(kind: VariableKind, seed: int) -> ArrayLike:
    rng: np.random.Generator = np.random.default_rng(seed)
    match kind:
        case VariableKind.NUMERIC:
            return rng.standard_normal(NUM_SAMPLES)
        case VariableKind.MULTIVARIATE:
            return rng.standard_normal((NUM_SAMPLES, 2))
        case VariableKind.BINARY:
            return rng.permutation(np.arange(NUM_SAMPLES) % 2)
        case VariableKind.CATEGORICAL:
            return rng.permutation(np.arange(NUM_SAMPLES) % 3)


def expected_kinds(metric_class: type[CorrelationMetric]) -> frozenset[ScalingKind]:
    if not metric_class.x_kind.is_numeric and not metric_class.y_kind.is_numeric:
        return frozenset()
    if metric_class in (ce.LinConcordance, ce.Intraclass):
        return frozenset({ScalingKind.IDENTITY})
    if metric_class is ce.Cosine:
        return frozenset({ScalingKind.IDENTITY, ScalingKind.PROPORTIONAL})
    if VariableKind.NUMERIC in (metric_class.x_kind, metric_class.y_kind):
        return frozenset(ScalingKind) - {ScalingKind.SAMPLE_WISE}
    return frozenset(ScalingKind)


def related_columns() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng: np.random.Generator = np.random.default_rng(0)
    x: NDArray[np.float64] = rng.standard_normal((50, 2))
    y: NDArray[np.float64] = np.column_stack(
        [x[:, 0] + 0.5 * rng.standard_normal(50), rng.standard_normal(50)]
    )
    return x, y


def scale_column(
    values: NDArray[np.float64], scaling: FeatureScaling | ScalingParameters
) -> NDArray[np.float64]:
    return (
        scaling.build_scaler()
        .fit_transform(values.reshape(-1, 1))
        .ravel()
        .astype(np.float64)
    )


@pytest.mark.parametrize(
    "metric_class",
    list(concrete_metrics()),
    ids=[metric_class.__name__ for metric_class in concrete_metrics()],
)
@pytest.mark.parametrize(
    "kind", list(ScalingKind), ids=[kind.name for kind in ScalingKind]
)
def test_every_metric_accepts_exactly_its_supported_kinds(
    metric_class: type[CorrelationMetric], kind: ScalingKind
) -> None:
    metric: CorrelationMetric = build_metric(metric_class)
    x: ArrayLike = sample_of(metric_class.x_kind, seed=1)
    y: ArrayLike = sample_of(metric_class.y_kind, seed=2)
    supported: frozenset[ScalingKind] = expected_kinds(metric_class)
    assert metric_class.supported_scaling_kinds() == supported
    if not supported:
        with pytest.raises(TypeError, match="no numeric variable"):
            _ = metric.evaluate_with_scaling(x, y, METHOD_OF_KIND[kind])
    elif kind in supported:
        _ = metric.evaluate_with_scaling(x, y, METHOD_OF_KIND[kind])
    else:
        with pytest.raises(ValueError, match=f"does not support {kind.name}") as error:
            _ = metric.evaluate_with_scaling(x, y, METHOD_OF_KIND[kind])
        assert all(item.name in str(error.value) for item in supported)


def test_rejection_follows_the_configured_kind_rather_than_the_class() -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x, y = related_columns()
    with pytest.raises(ValueError, match="does not support AFFINE"):
        _ = ce.Cosine().evaluate_with_scaling(
            x[:, 0], y[:, 0], RobustScalingParameters()
        )
    assert ce.Cosine().evaluate_with_scaling(
        x[:, 0], y[:, 0], RobustScalingParameters(is_centered=False)
    ).statistic == pytest.approx(ce.Cosine().evaluate(x[:, 0], y[:, 0]).statistic)


def test_rejected_scaling_comes_before_input_validation() -> None:
    with pytest.raises(ValueError, match="does not support AFFINE"):
        _ = ce.LinConcordance().evaluate_with_scaling(
            [1.0, np.nan, 3.0], [1.0, 2.0, 3.0], FeatureScaling.STANDARD
        )


def test_scaling_multivariate_uses_each_variable_scaled_separately() -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x, y = related_columns()
    x = x * np.array([1.0, 1000.0])
    scaling: FeatureScaling = FeatureScaling.STANDARD
    metric: ce.DistanceCorrelation = ce.DistanceCorrelation()
    expected: CorrelationResult = metric.evaluate(
        scaling.build_scaler().fit_transform(x),
        scaling.build_scaler().fit_transform(y),
    )
    assert metric.evaluate_with_scaling(x, y, scaling) == expected
    assert metric.evaluate(x, y) != expected


def test_scaling_one_dimensional_variables_keep_their_shape() -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x, y = related_columns()
    scaling: PowerScalingParameters = PowerScalingParameters()
    expected: float | None = (
        ce.Pearson()
        .evaluate(scale_column(x[:, 0], scaling), scale_column(y[:, 0], scaling))
        .statistic
    )
    assert ce.Pearson().evaluate_with_scaling(
        x[:, 0], y[:, 0], scaling
    ).statistic == pytest.approx(expected)


@pytest.mark.parametrize("metric_class", [ce.Partial, ce.Semipartial])
def test_scaling_transforms_covariates_with_the_variables(
    metric_class: type[ce.Partial] | type[ce.Semipartial],
) -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    covariate: NDArray[np.float64] = rng.lognormal(size=200)
    x: NDArray[np.float64] = covariate + 0.3 * rng.lognormal(size=200)
    y: NDArray[np.float64] = covariate + 0.3 * rng.lognormal(size=200)
    scaling: PowerScalingParameters = PowerScalingParameters()
    expected: CorrelationResult = metric_class(
        scale_column(covariate, scaling)
    ).evaluate(scale_column(x, scaling), scale_column(y, scaling))
    metric: ce.Partial | ce.Semipartial = metric_class(covariate)
    original: NDArray[np.float64] = metric.covariates.copy()
    result: CorrelationResult = metric.evaluate_with_scaling(x, y, scaling)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.p_value)
    assert np.array_equal(metric.covariates, original)


def test_affine_scaling_keeps_a_partial_correlation() -> None:
    rng: np.random.Generator = np.random.default_rng(1)
    covariate: NDArray[np.float64] = rng.standard_normal(100)
    x: NDArray[np.float64] = covariate + rng.standard_normal(100)
    y: NDArray[np.float64] = covariate + rng.standard_normal(100)
    metric: ce.Partial = ce.Partial(covariate)
    assert metric.evaluate_with_scaling(
        x, y, FeatureScaling.STANDARD
    ).statistic == pytest.approx(metric.evaluate(x, y).statistic)


def test_scaling_leaves_inputs_unchanged() -> None:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x, y = related_columns()
    original_x: NDArray[np.float64] = x.copy()
    _ = ce.RVCoefficient().evaluate_with_scaling(x, y, FeatureScaling.STANDARD)
    assert np.array_equal(x, original_x)


def test_scaling_mixed_metric_scales_only_the_numeric_variable() -> None:
    labels: list[str] = ["a", "a", "a", "b", "b", "c", "c", "c"]
    values: NDArray[np.float64] = np.array([1.0, 2.0, 1.5, 4.0, 5.0, 8.0, 7.0, 9.0])
    metric: ce.CorrelationRatio = ce.CorrelationRatio()
    assert metric.evaluate_with_scaling(
        labels, values, FeatureScaling.STANDARD
    ).statistic == pytest.approx(metric.evaluate(labels, values).statistic)


def test_scaling_keeps_validation_of_inputs() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _ = ce.Pearson().evaluate_with_scaling(
            [1.0, np.nan, 3.0], [1.0, 2.0, 3.0], FeatureScaling.STANDARD
        )
    with pytest.raises(ValueError, match="same length"):
        _ = ce.Pearson().evaluate_with_scaling(
            [1.0, 2.0, 3.0], [1.0, 2.0], FeatureScaling.STANDARD
        )


def test_scaling_constant_variable_is_undefined() -> None:
    x: NDArray[np.float64]
    x, _ = related_columns()
    result: CorrelationResult = ce.Pearson().evaluate_with_scaling(
        x[:, 0], np.ones(50), FeatureScaling.STANDARD
    )
    assert result.statistic is None
