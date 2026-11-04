# computation

## Overview

Numerical building blocks that metrics share or rely on, with no knowledge of the metric API.

## Components

| Component | Description |
|-----------|-------------|
| [bivariate_normal.py](./bivariate_normal.py) | `BivariateNormal`, the distribution function behind latent likelihoods. |
| [column_correlation.py](./column_correlation.py) | `ColumnCorrelation`, the Pearson correlation and test of all column pairs at once. |
| [column_space.py](./column_space.py) | `ColumnSpace`, the unit-independent span of data columns. |
| [pairwise_distance.py](./pairwise_distance.py) | `PairwiseDistance`, shared distance and centering operations. |
