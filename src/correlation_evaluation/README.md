# correlation_evaluation

## Overview

Metrics of association between two variables behind one API. Layers depend only on those listed before them: shared values and validators; `metric.py`, the base class; and `metrics/`, the concrete metrics.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases. |
| [parameter_validator.py](./parameter_validator.py) | `ParameterValidator`, runtime checks for NaN, non-integer, and non-member options. |
| [alternative.py](./alternative.py) | `Alternative`, the hypothesis of a significance test. |
| [confidence_interval.py](./confidence_interval.py) | `ConfidenceInterval`. |
| [hypothesis_outcome.py](./hypothesis_outcome.py) | `HypothesisOutcome`, the result shape of backing significance tests. |
| [result.py](./result.py) | `CorrelationResult`, the outcome for one pair of variables, also built from a test. |
| [matrix.py](./matrix.py) | `CorrelationMatrix`, the outcomes for all pairs of columns. |
| [variable_kind.py](./variable_kind.py) | `VariableKind`, the validation and encoding of each measurement scale. |
| [contingency_table.py](./contingency_table.py) | `ContingencyTable`, joint counts of categorical variables. |
| [metric.py](./metric.py) | `CorrelationMetric`, the abstract metric that handles validation and undefined cases. |
| [metrics/](./metrics/README.md) | Concrete metrics, grouped by what they measure. |
