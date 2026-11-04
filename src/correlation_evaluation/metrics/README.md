# metrics

## Overview

Concrete metrics. Each is a frozen dataclass of its options, grouped by what it measures and which data it takes.

## Components

| Component | Description |
|-----------|-------------|
| [linear/](./linear/README.md) | Linear correlation of numeric variables, also lagged or adjusted for covariates. |
| [robust/](./robust/README.md) | Linear correlation that resists outliers. |
| [rank/](./rank/README.md) | Monotonic association from ranks or orderings. |
| [dependence/](./dependence/README.md) | General dependence, including non-monotonic relationships. |
| [multivariate/](./multivariate/README.md) | Association between sets of variables. |
| [mixed/](./mixed/README.md) | Association between a categorical and a numeric variable. |
| [latent/](./latent/README.md) | Correlation of normal variables assumed to underlie ordinal observations. |
| [categorical/](./categorical/README.md) | Association between categorical variables. |
| [agreement/](./agreement/README.md) | Agreement between two measurements or raters. |
