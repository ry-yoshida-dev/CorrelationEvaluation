# Tests

## Overview

Unit tests for the `correlation_evaluation` package: hand-computed values, agreement with mathematical identities, undefined cases, and input validation.

## Components

| Component | Description |
| --- | --- |
| [`test_metric.py`](test_metric.py) | Tests for validation, label encoding, undefined cases, and matrices, across all metrics. |
| [`test_scaling.py`](test_scaling.py) | Tests for `evaluate_with_scaling`. |
| [`test_invariance.py`](test_invariance.py) | Tests that values do not depend on variable order, sample order, or units, and that undefined results stay consistent. |
| [`test_linear.py`](test_linear.py) | Tests for `metrics/linear`. |
| [`test_robust.py`](test_robust.py) | Tests for `metrics/robust`. |
| [`test_rank.py`](test_rank.py) | Tests for `metrics/rank`. |
| [`test_dependence.py`](test_dependence.py) | Tests for `metrics/dependence`. |
| [`test_multivariate.py`](test_multivariate.py) | Tests for `metrics/multivariate`. |
| [`test_mixed.py`](test_mixed.py) | Tests for `metrics/mixed`. |
| [`test_latent.py`](test_latent.py) | Tests for `metrics/latent`. |
| [`test_categorical.py`](test_categorical.py) | Tests for `metrics/categorical`. |
| [`test_agreement.py`](test_agreement.py) | Tests for `metrics/agreement`. |

## Examples

```bash
uv run pytest
```
