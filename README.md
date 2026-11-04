# CorrelationEvaluation

## Overview

CorrelationEvaluation (`correlation_evaluation`) is a Python package of metrics that measure the association between two variables, behind one API. Each metric is an object holding its options, with `evaluate` for one pair of variables and `evaluate_matrix` for every pair of columns. It wraps SciPy and scikit-learn, and computes from the published formula only what they lack.

For module-level design notes, see [src/correlation_evaluation/README.md](src/correlation_evaluation/README.md).

## Metrics

"Directional" metrics answer how much x tells about y, so swapping the variables changes the value.

| Group | Metric | Takes (x, y) | Range | P-value | Backed by |
| --- | --- | --- | --- | --- | --- |
| Linear | `Pearson` | numeric, numeric | -1 to 1 | Yes, and confidence interval | SciPy |
| | `Cosine` | numeric, numeric | -1 to 1 | | Formula |
| | `LaggedPearson` (directional) | numeric, numeric | -1 to 1 | Yes | SciPy |
| | `Partial` | numeric, numeric, covariates | -1 to 1 | Yes | Formula |
| | `Semipartial` (directional) | numeric, numeric, covariates | -1 to 1 | Yes | Formula |
| Robust | `BiweightMidcorrelation` | numeric, numeric | -1 to 1 | | Formula |
| | `PercentageBend` | numeric, numeric | -1 to 1 | Yes | Formula |
| | `Winsorized` | numeric, numeric | -1 to 1 | Yes | Formula |
| Rank | `Spearman` | numeric, numeric | -1 to 1 | Yes | SciPy |
| | `KendallTau` (tau-b, tau-c) | numeric, numeric | -1 to 1 | Yes | SciPy |
| | `WeightedTau` | numeric, numeric | -1 to 1 | | SciPy |
| | `SomersD` (directional) | numeric, numeric | -1 to 1 | Yes | Formula |
| | `GoodmanKruskalGamma` | numeric, numeric | -1 to 1 | | Formula over SciPy |
| | `BlomqvistBeta` | numeric, numeric | -1 to 1 | | Formula |
| | `NormalScores` | numeric, numeric | -1 to 1 | | Formula over SciPy |
| Dependence | `ChatterjeeXi` (directional) | numeric, numeric | about 0 to 1 | Yes | SciPy |
| | `HoeffdingD` | numeric, numeric | -0.5 to 1 without ties; ties in few samples can leave it | | Formula |
| | `DistanceCorrelation` | multivariate, multivariate | 0 to 1; negative values are possible with `is_bias_corrected`, which estimates the square and requires `is_squared` | | Formula |
| | `HilbertSchmidt` | multivariate, multivariate | 0 to 1 | | Formula |
| | `MultiscaleGraph` | multivariate, multivariate | -1 to 1 | Yes | SciPy |
| | `NearestNeighborMutualInformation` | numeric, numeric | 0 and up, nats | | scikit-learn |
| Multivariate | `RVCoefficient` | multivariate, multivariate | 0 to 1 | | Formula |
| | `Canonical` | multivariate, multivariate | 0 to 1 | | Formula |
| Mixed | `PointBiserial` | binary, numeric | -1 to 1 | Yes | SciPy |
| | `RankBiserial` | binary, numeric | -1 to 1 | Yes | Formula over SciPy |
| | `CorrelationRatio` | categorical, numeric | 0 to 1 | Yes | Formula |
| Latent | `Tetrachoric` | binary, binary | -1 to 1 | | Formula over SciPy |
| | `Polychoric` | ordinal, ordinal | -1 to 1 | | Formula over SciPy |
| | `Biserial` | binary, numeric | can exceed 1 in magnitude | | Formula over SciPy |
| | `Polyserial` | ordinal, numeric | can exceed 1 in magnitude | | Formula over SciPy |
| Categorical | `Phi` | binary, binary | -1 to 1 | Yes | SciPy |
| | `Matthews` | classes, classes | -1 to 1 | | scikit-learn |
| | `CramersV` | categorical, categorical | 0 to 1 | Yes | SciPy |
| | `TschuprowT` | categorical, categorical | 0 to 1 | Yes | SciPy |
| | `ContingencyCoefficient` | categorical, categorical | 0 to below 1 | Yes | SciPy |
| | `TheilsU` (directional) | categorical, categorical | 0 to 1 | | Formula over scikit-learn |
| | `GoodmanKruskalLambda` (directional) | categorical, categorical | 0 to 1 | | Formula |
| | `GoodmanKruskalTau` (directional) | categorical, categorical | 0 to 1 | | Formula |
| | `MutualInformation` | categorical, categorical | 0 and up, nats | | scikit-learn |
| | `NormalizedMutualInformation` | categorical, categorical | 0 to 1 | | scikit-learn |
| | `AdjustedMutualInformation` | categorical, categorical | about 0 to 1; few samples can leave it | | scikit-learn |
| Agreement | `LinConcordance` | numeric, numeric | -1 to 1 | | Formula |
| | `CohenKappa` | raters, raters | -1 to 1 | | scikit-learn |
| | `Intraclass` | numeric, numeric | at most 1; below -1 is possible for `TWO_WAY_RANDOM` with few samples | | Formula |

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

Git must be available, because the [DataScaler](https://github.com/ry-yoshida-dev/DataScaler) dependency installs from its repository.

## Examples

Evaluate one pair of variables:

```python
from correlation_evaluation import Alternative, Pearson, Spearman

result = Pearson(confidence_level=0.9).evaluate(x, y)
print(result.statistic, result.p_value, result.confidence_interval)

print(Spearman(alternative=Alternative.GREATER).evaluate(x, y).p_value)
```

Compute a correlation matrix over the columns of `X`:

```python
from correlation_evaluation import KendallTau

matrix = KendallTau().evaluate_matrix(X)
print(matrix.statistics)
print(matrix.p_values)
```

Find dependence that correlation misses:

```python
import numpy as np

from correlation_evaluation import DistanceCorrelation, Pearson

x = np.linspace(-1.0, 1.0, 101)
print(Pearson().evaluate(x, x**2).statistic)  # about 0
print(DistanceCorrelation().evaluate(x, x**2).statistic)  # about 0.49
```

Remove a confounder, and relate labels:

```python
from correlation_evaluation import CramersV, Partial, TheilsU

print(Partial(covariates=age).evaluate(income, spending).statistic)

print(CramersV(is_bias_corrected=True).evaluate(colors, sizes).statistic)
print(TheilsU().evaluate(colors, sizes).statistic)
```

Scale the variables first with a method of [DataScaler](https://github.com/ry-yoshida-dev/DataScaler). A method that would change what the metric measures raises `ValueError`, which lists the supported kinds:

```python
from correlation_evaluation import DistanceCorrelation, LinConcordance
from data_scaler import FeatureScaling, RobustScalingParameters

metric = DistanceCorrelation()
print(metric.evaluate_with_scaling(X, Y, FeatureScaling.STANDARD).statistic)
print(
    metric.evaluate_with_scaling(
        X, Y, RobustScalingParameters(quantile_range=(10.0, 90.0))
    ).statistic
)
print(LinConcordance.supported_scaling_kinds())  # frozenset({ScalingKind.IDENTITY})
```

| Metric | `IDENTITY` | `PROPORTIONAL` | `AFFINE` | `MONOTONIC` | `SAMPLE_WISE` |
| --- | --- | --- | --- | --- | --- |
| `LinConcordance`, `Intraclass` | ✔ | ✗ | ✗ | ✗ | ✗ |
| `Cosine` | ✔ | ✔ | ✗ | ✗ | ✗ |
| Other metrics of one-dimensional numeric variables | ✔ | ✔ | ✔ | ✔ | ✗ |
| Metrics of multivariate variables | ✔ | ✔ | ✔ | ✔ | ✔ |

`FeatureScaling` and every parameter class of DataScaler report their kind as `kind`. Metrics of labels only take no scaling. `SAMPLE_WISE` also raises `ValueError` for a multivariate variable with only one feature. `QuantileScalingParameters` defaults to 1000 quantiles, and scikit-learn warns when that exceeds the samples, so set `n_quantiles` for smaller data.

## Notes

- A statistic is `None` when the metric is undefined for the data, rather than a misleading number: for a constant variable, except for `Cosine`, `MutualInformation`, and `CohenKappa`, where a constant has a meaningful value, and for too few samples. Its p-value and confidence interval are then `None` too. In matrices it is NaN.
- `evaluate` never scales its input. `DistanceCorrelation`, `HilbertSchmidt`, `MultiscaleGraph`, and `RVCoefficient` compare columns on their raw scale, so a column in large units can dominate the value; use `evaluate_with_scaling` when such differences are only a matter of units. It scales x, y, and the covariates of `Partial` and `Semipartial` separately, and leaves labels as they are.
- `evaluate_with_scaling` rejects what would change the meaning of a metric: separate scaling removes what `LinConcordance` and `Intraclass` detect, centering makes `Cosine` a Pearson correlation, and a sample-wise method such as `FeatureScaling.UNIT_NORM` leaves only the sign of a one-dimensional variable. To compare two raters on a common scale, fit one scaler on both yourself. Which accepted method suits the data remains your decision; for example, no method fully corrects an extreme outlier, so deal with outliers first.
- `p_value` and `confidence_interval` are `None` when the metric has no established closed-form test. A permutation test is the general alternative.
- Values beyond 1e50 or below 1e-50 in magnitude are rescaled exactly by a power of two before computing, so they neither overflow nor underflow. `HilbertSchmidt` with a fixed `bandwidth` keeps its units instead.
- NaN or infinite input raises `ValueError`; remove missing values first, because no single policy suits all metrics.
- Labels may be integers, floats, booleans, or strings, including NumPy variable-width strings and object arrays such as pandas string columns. Object arrays must hold one type, so missing values raise an error. Ordinal metrics order them by sort order, so name them so that it matches their meaning; for example, the strings `"10"` and `"9"` sort in the wrong order.
- `Matthews` and `CohenKappa` compare labels across x and y, so both must use the same label set and type, and mixing numbers with strings raises `TypeError`; other categorical metrics treat them as unrelated.
- `HoeffdingD`, `DistanceCorrelation`, `HilbertSchmidt`, and `MultiscaleGraph` take O(n²) memory, and `MultiscaleGraph` additionally runs a permutation test.
- `SomersD` takes O(n²) time, so for very large data `KendallTau`, at O(n log n), scales far better.
- `NearestNeighborMutualInformation` assumes no ties and breaks them with small random noise, so with ties the value depends on `random_state`, though not on the order of the samples or variables; use `MutualInformation` for discrete data.
- `ChatterjeeXi` breaks ties in x at random, as the estimator defines, using `random_state`. With such ties the value depends on the seed, but not on the order of the samples.
- `Semipartial` reports the p-value of `Partial`, since both test whether x and y are related given the covariates. Both are undefined when no degrees of freedom remain, that is, for at most two samples more than independent covariates; collinear covariates count once.
- `Partial`, `Semipartial`, `CorrelationMatrix`, and `ContingencyTable` hold arrays, so they compare by identity.
- `evaluate_matrix` computes all pairs at once for `Pearson` and `Spearman`, and calls other metrics once per pair of columns, which is slower for very wide data. It raises `TypeError` for metrics that relate different kinds of variables, and its validation errors name the column.
- `Tetrachoric` and `Polychoric` have no interior likelihood maximum for tables with empty cells that imply a perfect relationship. The estimate then approaches ±1, such as 0.999, and its exact value is not meaningful.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on Python 3.12 to 3.14, on every push to `main` and every pull request, along with the tests on the minimum supported dependency versions.

## License

Released under the [MIT License](LICENSE).
