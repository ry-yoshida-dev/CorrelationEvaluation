# CorrelationEvaluation

## Overview

CorrelationEvaluation (`correlation_evaluation`) is a Python package of metrics that measure the association between two variables, behind one API. Each metric is an object holding its options, with `evaluate` for one pair of variables, `evaluate_matrix` for every pair of columns, and `evaluate_with_scaling` to scale the variables first. It wraps SciPy and scikit-learn, and computes from the published formula only what they lack.

| Group | Measures | Metrics |
| --- | --- | --- |
| [Linear](src/correlation_evaluation/metrics/linear/README.md) | Linear correlation of numeric variables, also lagged or adjusted for covariates | `Pearson`, `Cosine`, `LaggedPearson`, `Partial`, `Semipartial` |
| [Robust](src/correlation_evaluation/metrics/robust/README.md) | Linear correlation that resists outliers | `BiweightMidcorrelation`, `PercentageBend`, `Winsorized` |
| [Rank](src/correlation_evaluation/metrics/rank/README.md) | Monotonic association from ranks or orderings | `Spearman`, `KendallTau`, `WeightedTau`, `SomersD`, `GoodmanKruskalGamma`, `BlomqvistBeta`, `NormalScores` |
| [Dependence](src/correlation_evaluation/metrics/dependence/README.md) | General dependence, including non-monotonic relationships | `ChatterjeeXi`, `HoeffdingD`, `DistanceCorrelation`, `HilbertSchmidt`, `MultiscaleGraph`, `NearestNeighborMutualInformation` |
| [Multivariate](src/correlation_evaluation/metrics/multivariate/README.md) | Association between sets of variables | `RVCoefficient`, `Canonical` |
| [Mixed](src/correlation_evaluation/metrics/mixed/README.md) | Association between a categorical and a numeric variable | `PointBiserial`, `RankBiserial`, `CorrelationRatio` |
| [Latent](src/correlation_evaluation/metrics/latent/README.md) | Correlation of normal variables assumed to underlie ordinal observations | `Tetrachoric`, `Polychoric`, `Biserial`, `Polyserial` |
| [Categorical](src/correlation_evaluation/metrics/categorical/README.md) | Association between categorical variables | `Phi`, `Matthews`, `CramersV`, `TschuprowT`, `ContingencyCoefficient`, `TheilsU`, `GoodmanKruskalLambda`, `GoodmanKruskalTau`, `MutualInformation`, `NormalizedMutualInformation`, `AdjustedMutualInformation` |
| [Agreement](src/correlation_evaluation/metrics/agreement/README.md) | Agreement between two measurements or raters | `LinConcordance`, `CohenKappa`, `Intraclass` |

## Installation

```bash
pip install .
```

Git must be available, because the [DataScaler](https://github.com/ry-yoshida-dev/DataScaler) dependency installs from its repository.

## Usage

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

Scale the variables first with a method of DataScaler:

```python
from correlation_evaluation import DistanceCorrelation
from data_scaler import FeatureScaling

print(DistanceCorrelation().evaluate_with_scaling(X, Y, FeatureScaling.STANDARD).statistic)
```
