# rank

## Overview

Monotonic association from ranks or orderings, unaffected by monotonic transformations.

## Components

| Component | Description |
|-----------|-------------|
| [spearman.py](./spearman.py) | `Spearman`. |
| [kendall_tau.py](./kendall_tau.py) | `KendallTau`. |
| [kendall_variant.py](./kendall_variant.py) | `KendallVariant`, tau-b or tau-c. |
| [weighted_tau.py](./weighted_tau.py) | `WeightedTau`, emphasizing top ranks. |
| [somers_d.py](./somers_d.py) | `SomersD`, directional. |
| [goodman_kruskal_gamma.py](./goodman_kruskal_gamma.py) | `GoodmanKruskalGamma`, ignoring ties. |
| [blomqvist_beta.py](./blomqvist_beta.py) | `BlomqvistBeta`, from signs around medians. |
| [normal_scores.py](./normal_scores.py) | `NormalScores`, from normal quantiles of ranks. |
