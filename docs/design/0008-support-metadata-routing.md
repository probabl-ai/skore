---
status: Design finalized, implementation deferred
date: 2026-03-06
authors:
  - "@auguste-probabl"
  - "@thomass-dev"
---

# Support Metadata Routing for Custom Metrics

## Context

Users need to pass additional data arrays to metrics during computation:
- **Sample weights**: Weight individual samples differently in metric calculations
- **Segment identifiers**: Compute metrics per segment (e.g., per demographic group)
- **Cost matrices**: Business-specific cost parameters
- **Other metadata**: Any additional per-sample information needed by custom metrics

### Current Limitations

Currently, passing metadata to metrics is not supported:
```python
# This doesn't work today
report.metrics.summarize(
    metric="weighted_accuracy",
    sample_weight=weights  # ❌ Not supported
)
```

Users must work around by:
1. Hard-coding weights into custom metric functions (not flexible)
2. Using global variables (not thread-safe, not serializable)
3. Re-computing metrics manually outside of skore (loses caching, integration)

### Related Issues

**Issue #2120**: Metadata routing support
- Users need to pass sample weights, segment data to metrics
- Should work for both EstimatorReport and CrossValidationReport
- Should integrate with sklearn's metadata routing API

---

## Decision

Implement metadata routing using **sklearn's metadata routing API** with **call-time parameter passing**.

### Design Principles

1. **Use sklearn's existing API**: Leverage `set_score_request()` and `get_metadata_routing()`
2. **Call-time, not registration-time**: Metadata is passed when calling `summarize()`, not when registering
3. **Automatic slicing for CV**: CrossValidationReport automatically slices metadata per fold
4. **Backward compatible**: Existing code continues to work without changes

---

## Architecture

### Key Insight

**CrossValidationReport delegates all metric computation to EstimatorReport instances** (one per fold). Each EstimatorReport already has pre-sliced X_test, y_test for its fold. We extend this pattern to slice metadata arrays too.

### Flow Diagram

```
User calls:
  cv_report.metrics.summarize(metric="weighted_accuracy", sample_weight=weights)
  └─> weights.shape = (n_samples,)  # Full dataset length

CrossValidationReport:
  For each fold (train_indices, test_indices):
    1. Slice metadata: fold_weights = weights[test_indices]
    2. Call fold_report.metrics.summarize(..., sample_weight=fold_weights)
       └─> fold_report is an EstimatorReport

EstimatorReport:
  1. Check scorer.get_metadata_routing() to see what it requests
  2. Route sample_weight to scorer if requested
  3. Compute: scorer(y_true, y_pred, sample_weight=fold_weights)

Aggregate results across folds
```

---

## Implementation Design

### For EstimatorReport (Simple Case)

Metadata arrays match test set length:

```python
from sklearn.metrics import make_scorer, accuracy_score

def weighted_accuracy(y_true, y_pred, sample_weight):
    return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

scorer = make_scorer(weighted_accuracy, response_method="predict")
scorer.set_score_request(sample_weight=True)  # sklearn API

# Register (if using registry) or use directly
report.metrics.register(scorer)

# Pass metadata at call-time
weights = np.array([...])  # len(weights) == len(X_test)
report.metrics.summarize(
    metric="weighted_accuracy",
    sample_weight=weights  # Routed to metrics that request it
)
```

**Implementation in EstimatorReport._compute_metric_scores()**:

```python
def _compute_metric_scores(self, metric, **kwargs):
    # Get scorer's metadata routing
    if hasattr(scorer, 'get_metadata_routing'):
        routing = scorer.get_metadata_routing()

        # Extract requested metadata from kwargs
        routed_params = {}
        for param_name in routing.score.requests:
            if param_name in kwargs:
                routed_params[param_name] = kwargs.pop(param_name)

        # Call scorer with routed metadata
        y_pred = self.get_predictions(...)
        score = scorer._score_func(y_true, y_pred, **routed_params)
    else:
        # No routing - existing behavior
        score = scorer(estimator, X, y_true)

    return score
```

### For CrossValidationReport (Complex Case)

Metadata arrays match full dataset length, automatically sliced per fold:

```python
# User passes metadata arrays matching full dataset length
weights = np.array([...])  # len(weights) == len(X)

cv_report.metrics.summarize(
    metric="weighted_accuracy",
    sample_weight=weights  # Automatically sliced per fold
)
```

**Step 1: Extend `_generate_estimator_report()` to accept and slice metadata**

Currently in `skore/_sklearn/_cross_validation/report.py`:
```python
def _generate_estimator_report(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None,
    pos_label: PositiveLabel | None,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    metadata: dict[str, ArrayLike] | None = None,  # NEW
) -> EstimatorReport:
    if y is None:
        y_train = None
        y_test = None
    else:
        y_train = _safe_indexing(y, train_indices)
        y_test = _safe_indexing(y, test_indices)

    report = EstimatorReport(
        estimator,
        fit=True,
        X_train=_safe_indexing(X, train_indices),
        y_train=y_train,
        X_test=_safe_indexing(X, test_indices),
        y_test=y_test,
        pos_label=pos_label,
    )

    # NEW: Store sliced metadata on the report
    if metadata:
        report._test_metadata = {
            key: _safe_indexing(val, test_indices)
            for key, val in metadata.items()
        }
        report._train_metadata = {
            key: _safe_indexing(val, train_indices)
            for key, val in metadata.items()
        }

    return report
```

**Step 2: Update `CrossValidationReport._compute_metric_scores()` to propagate metadata**

In `skore/_sklearn/_cross_validation/metrics_accessor.py`:

```python
def _compute_metric_scores(
    self,
    report_metric_name: str,
    *,
    data_source: DataSource = "test",
    aggregate: Aggregate | None = None,
    **metric_kwargs: Any,
) -> pd.DataFrame:
    # Separate metadata params from regular kwargs
    metadata_params = self._extract_metadata_params(metric_kwargs)

    cache_key = deep_key_sanitize(
        (
            self._parent._hash,
            report_metric_name,
            data_source,
            aggregate,
            metric_kwargs,
            # Include metadata in cache key
            tuple(metadata_params.keys()) if metadata_params else None,
        )
    )

    results = self._parent._cache.get(cache_key)
    if results is None:
        parallel = Parallel(...)

        results = [
            result.frame(**frame_kwargs)
            for result in track(
                parallel(
                    delayed(self._call_fold_with_metadata)(
                        report, report_metric_name, data_source,
                        metadata_params, metric_kwargs
                    )
                    for report in self._parent.estimator_reports_
                ),
                description="Compute metric for each split",
                total=len(self._parent.estimator_reports_),
            )
        ]
        # ... rest of aggregation logic

    return results

def _extract_metadata_params(self, kwargs: dict) -> dict[str, ArrayLike]:
    """Extract metadata arrays from kwargs.

    Metadata arrays are numpy arrays matching full dataset length.
    """
    metadata = {}
    keys_to_remove = []

    for key, value in kwargs.items():
        if isinstance(value, np.ndarray) and len(value) == len(self._parent._X):
            metadata[key] = value
            keys_to_remove.append(key)

    for key in keys_to_remove:
        kwargs.pop(key)

    return metadata

def _call_fold_with_metadata(
    self, report, method_name, data_source, metadata_params, kwargs
):
    """Call fold's method with sliced metadata."""
    fold_kwargs = kwargs.copy()

    # Add sliced metadata if available
    if metadata_params and hasattr(report, f'_{data_source}_metadata'):
        fold_metadata = getattr(report, f'_{data_source}_metadata')
        fold_kwargs.update(fold_metadata)

    # Call fold's method
    return getattr(report.metrics, method_name)(
        data_source=data_source,
        **fold_kwargs
    )
```

**Step 3: Update `CrossValidationReport._fit_estimator_reports()` to pass metadata**

```python
def _fit_estimator_reports(self, metadata=None) -> list[EstimatorReport]:
    """Fit the estimator reports.

    Parameters
    ----------
    metadata : dict[str, ArrayLike] | None
        Metadata arrays to slice per fold. Keys are param names, values are
        arrays with len == len(X).

    Returns
    -------
    estimator_reports : list of EstimatorReport
        The estimator reports.
    """
    parallel = Parallel(...)

    return list(
        track(
            parallel(
                delayed(_generate_estimator_report)(
                    clone(self._estimator),
                    self._X,
                    self._y,
                    self._pos_label,
                    train_indices,
                    test_indices,
                    metadata=metadata,  # NEW
                )
                for (train_indices, test_indices) in self.split_indices
            ),
            description=f"Processing cross-validation\nfor {self.estimator_name_}",
            total=len(self.split_indices),
        )
    )
```

**Challenge**: Metadata needs to be known at initialization time for slicing, but it's passed at `summarize()` call-time.

**Solution**: Lazy slicing - don't slice at initialization, slice when needed:
- Store full metadata arrays on CrossValidationReport
- Slice dynamically in `_call_fold_with_metadata()`
- Use fold's `test_indices` from `split_indices`

Revised approach:

```python
# In CrossValidationReport._compute_metric_scores()
def _call_fold_with_metadata(
    self, report, fold_idx, method_name, data_source, metadata_params, kwargs
):
    """Call fold's method with sliced metadata."""
    fold_kwargs = kwargs.copy()

    # Slice metadata for this fold
    if metadata_params:
        train_indices, test_indices = self._parent.split_indices[fold_idx]
        indices = test_indices if data_source == "test" else train_indices

        fold_metadata = {
            key: _safe_indexing(val, indices)
            for key, val in metadata_params.items()
        }
        fold_kwargs.update(fold_metadata)

    # Call fold's method
    return getattr(report.metrics, method_name)(
        data_source=data_source,
        **fold_kwargs
    )
```

---

## User-Facing API

### Registration (using sklearn's API)

```python
from sklearn.metrics import make_scorer

def business_loss(y_true, y_pred, cost_matrix, segment):
    """Custom metric using metadata."""
    # Compute loss using cost_matrix and segment
    ...
    return loss

scorer = make_scorer(business_loss, response_method="predict")
scorer.set_score_request(cost_matrix=True, segment=True)

# Register (optional, if using registry)
report.metrics.register(scorer)
```

### Usage - EstimatorReport

Metadata arrays match test set length:

```python
cost_matrix = np.array([[0, 10], [5, 0]])
segments = np.array([...])  # len(segments) == len(X_test)

report.metrics.summarize(
    metric="business_loss",
    cost_matrix=cost_matrix,  # Routed to business_loss
    segment=segments
)
```

### Usage - CrossValidationReport

Metadata arrays match full dataset length (auto-sliced per fold):

```python
cost_matrix = np.array([[0, 10], [5, 0]])
segments = np.array([...])  # len(segments) == len(X) - full dataset!

cv_report.metrics.summarize(
    metric="business_loss",
    cost_matrix=cost_matrix,
    segment=segments  # Automatically sliced per fold
)
```

### Multiple Metrics with Different Routing

```python
scorer1 = make_scorer(weighted_acc, response_method="predict")
scorer1.set_score_request(sample_weight=True)

scorer2 = make_scorer(segmented_f1, response_method="predict")
scorer2.set_score_request(segment=True)

scorer3 = make_scorer(plain_acc, response_method="predict")
# No routing configured

report.metrics.summarize(
    metric=["weighted_acc", "segmented_f1", "plain_acc"],
    sample_weight=weights,  # Only routed to weighted_acc
    segment=segments        # Only routed to segmented_f1
)
```

---

## Consequences

### Positive

1. **Sklearn compatibility**: Uses standard sklearn API
2. **Flexible**: Supports arbitrary metadata parameters
3. **Automatic CV handling**: CrossValidationReport slices metadata transparently
4. **Cacheable**: Metadata routing doesn't break caching
5. **Backward compatible**: Existing code works without changes

### Negative

1. **More complex**: Additional logic for slicing and routing
2. **Validation needed**: Must validate array lengths
3. **Documentation burden**: Users need to understand sklearn's routing API

### Neutral

1. **Call-time vs registration-time**: We chose call-time for flexibility
2. **Automatic detection**: We detect metadata arrays by length (could be explicit instead)

---

## Alternatives Considered

### Alternative 1: Registration-time metadata declaration

```python
# Register with metadata
report.metrics.register(
    scorer,
    metadata={"sample_weight": weights}  # ❌ Fixed at registration
)
```

**Rejected**: Too inflexible - metadata often changes between calls (e.g., different weight schemes for experimentation).

### Alternative 2: Separate method for metadata

```python
# Separate API
report.metrics.summarize_with_metadata(
    metric="weighted_acc",
    metadata={"sample_weight": weights}
)
```

**Rejected**: Duplicates API surface, makes metrics routing a special case rather than normal behavior.

### Alternative 3: Explicit metadata parameter

```python
# Explicit metadata dict
report.metrics.summarize(
    metric="weighted_acc",
    metadata={"sample_weight": weights, "segment": segments}
)
```

**Rejected**: Extra nesting, less ergonomic. sklearn's routing handles this at the scorer level already.

---

## Testing Strategy

### Unit Tests

1. **EstimatorReport metadata routing**
   - Metric receives metadata when requested
   - Metric doesn't receive metadata when not requested
   - Multiple metrics with different routing in one call
   - Array length validation

2. **CrossValidationReport metadata slicing**
   - Metadata correctly sliced per fold
   - Slicing uses correct indices (test vs train)
   - Fold reports receive correct subset
   - Aggregation works correctly

3. **Edge cases**
   - Empty metadata dict
   - Metadata array wrong length (should error)
   - Scorer without routing configured
   - Mix of scorers with/without routing

### Integration Tests

1. **End-to-end with sklearn scorers**
   - Use sklearn's actual `set_score_request()` API
   - Verify routing through full stack

2. **With metrics registry**
   - Registered metrics with routing
   - Serialization of routing config

3. **Caching behavior**
   - Metadata changes invalidate cache
   - Same metadata hits cache

---

## Open Questions

### Q1: Train vs Test metadata?

Should metadata routing work for `data_source="train"` too?

**Answer**: Yes, use train_indices when data_source="train".

### Q2: Array length validation?

When should we validate metadata array lengths?

**Answer**: At call-time in `summarize()`. Raise clear error:
```python
if len(sample_weight) != len(X_test):
    raise ValueError(
        f"sample_weight has length {len(sample_weight)} but X_test has "
        f"length {len(X_test)}. For CrossValidationReport, pass the full "
        f"dataset length {len(X)}."
    )
```

### Q3: Metadata array detection?

How to distinguish metadata arrays from regular kwargs?

**Answer**: For CrossValidationReport, detect by length:
```python
if isinstance(value, np.ndarray) and len(value) == len(self._X):
    # It's a metadata array
```

For EstimatorReport, no special detection needed - just pass kwargs through and let sklearn's routing handle it.

### Q4: Documentation discoverability?

How to make this feature discoverable?

**Answer**:
1. Add examples to user guide
2. Show in error messages when array length is wrong
3. Mention in `help()` output
4. Add docstring examples

---

## Implementation Timeline

**Status**: Design finalized, implementation deferred

**Why defer?**
1. Orthogonal to metrics registry (can be implemented independently)
2. Requires touching multiple files (both report types)
3. Needs extensive testing across CV scenarios
4. Lower user priority than registry itself

**When to implement?**
- After metrics registry ships
- When user demand increases
- When sklearn's routing API stabilizes (it's relatively new)

**Breaking changes risk**: None - the design is extensible without breaking changes. We can implement this later and existing code will continue working (backward compatible).

---

## References

- [sklearn metadata routing docs](https://scikit-learn.org/stable/metadata_routing.html)
- `sklearn.metrics.make_scorer` API
- `sklearn.model_selection._validation._score` (sklearn's internal routing implementation)
- [Issue #2120][#2120]
- [Issue #2180][#2180]

[#2120]: https://github.com/probabl-ai/skore/issues/2120
[#2180]: https://github.com/probabl-ai/skore/issues/2180
