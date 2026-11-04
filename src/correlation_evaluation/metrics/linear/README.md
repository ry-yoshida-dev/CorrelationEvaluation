# linear

## Overview

Linear correlation of numeric variables.

## Components

| Component | Description |
|-----------|-------------|
| [pearson.py](./pearson.py) | `Pearson`, with p-value and confidence interval. |
| [cosine.py](./cosine.py) | `Cosine`, without centering. |
| [lagged_pearson.py](./lagged_pearson.py) | `LaggedPearson`, at a time offset. |
| [covariate_adjusted.py](./covariate_adjusted.py) | `CovariateAdjusted`, the abstract correlation after removing covariates. |
| [covariate_adjustments/](./covariate_adjustments/README.md) | `Partial` and `Semipartial`, the concrete adjustments. |
