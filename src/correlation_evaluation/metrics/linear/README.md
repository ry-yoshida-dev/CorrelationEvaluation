# linear

## Overview

Linear correlation of numeric variables.

## Components

| Component | Description |
|-----------|-------------|
| [pearson.py](./pearson.py) | `Pearson`, with p-value and confidence interval. |
| [pearson_outcome.py](./pearson_outcome.py) | `PearsonOutcome`, the result shape of the backing Pearson test. |
| [cosine.py](./cosine.py) | `Cosine`, without centering. |
| [lagged_pearson.py](./lagged_pearson.py) | `LaggedPearson`, at a time offset. |
| [covariate_adjusted.py](./covariate_adjusted.py) | `CovariateAdjusted`, the abstract correlation after removing covariates. |
| [partial.py](./partial.py) | `Partial`, with covariates removed from both variables. |
| [semipartial.py](./semipartial.py) | `Semipartial`, with covariates removed from y. |
