---
status: accepted
date: 2026-06-29
decision-makers: ["@augustebaum"]
---

# Show `neg_*` metrics from sklearn if they are used

## Context and Problem Statement

We allow users to pass metrics directly from sklearn, some of which are prefixed with `neg_`
for technical reasons.
In `0002-never-present-neg-metrics-from-sklearn` the decision was made to accept `neg_` metric
but silently convert them to their non-neg equivalent.

Following this, a number of bug reports slowly changed the behaviour to let the user use either
the `neg_` or non-`neg_` name, but never in a principled manner.

## Decision

Show `neg_*` metrics from sklearn in skore, when they exist; accept non-`neg_`
metrics when their `neg_` counterpart exists.

For example, see the following snippet:
```py
report1.metrics.add("neg_mean_absolute_error")
report1.metrics.get("neg_mean_absolute_error")  # passes (no conversion is done)
report1.metrics.get("mean_absolute_error")  # fails (no conversion is done)

report2.metrics.add("mean_absolute_error")  # passes (we recognize that neg_mean_absolute_error exists in sklearn, and register the non-neg version)
report2.metrics.get("mean_absolute_error")  # passes
report2.metrics.get("neg_mean_absolute_error")  # fails (no conversion is done)
```

### Consequences

* We minimize user surprise
* Avoid special treatment of `neg_` in the code

## Notes

The following script shows a perhaps surprising outcome:
```py
from skore import evaluate
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import get_scorer

X, y = make_regression(n_samples=500, n_features=50, n_targets=1, random_state=0)
report = evaluate(HistGradientBoostingRegressor(random_state=0), X, y, splitter=5)
report.metrics.add(get_scorer("neg_mean_absolute_percentage_error"))
report.metrics.available()
# ['mean_absolute_percentage_error',
#  'score',
#  'r2',
#  'rmse',
#  'mae',
#  'mape',
#  'fit_time',
#  'predict_time']
```
