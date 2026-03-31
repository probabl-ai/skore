---
status: Accepted
date: 2026-03-06
authors:
  - "@auguste-probabl"
  - "@thomass-dev"
---

# Introduce Metrics Registry for EstimatorReport

## Context

Users want to register custom metrics on reports so that:
1. Metrics persist with the report and survive serialization (Project storage)
2. Custom metrics appear automatically in `report.metrics.summarize()`
3. Metrics are discoverable without knowing their names explicitly
4. Each metric has metadata (name, verbose_name, greater_is_better)
5. Built-in metric names cannot be accidentally overridden

### Current Limitations

Currently, users must pass custom metrics to `summarize()` every time:
```python
report.metrics.summarize(metric=[my_custom_metric])
```

Problems:
- Custom metrics don't appear in default `summarize()` output
- When sharing reports, collaborators need the original function definition
- No way to list available custom metrics
- Cache invalidation issues when editing function definitions (see [issue #2061][#2061])
- Behaviour is inconsistent between `summarize()` and `<metric>()` (see [issue #2001][[#2001])

### Related Issues

[Issue #2061][#2061]: Registry for custom metrics
- User wants metrics to "travel" with the report when shared
- Need automatic inclusion in `summarize()`
- Need discoverability

[Issue #2120][#2120]: Metadata routing support
- Users need to pass sample weights, segment data to metrics
- Out of scope for this iteration, but design should accommodate future support

[Issue #2203][#2203]: First-class sklearn Scorer support
- `make_scorer()` objects should work seamlessly
- Extract `_score_func`, `_response_method`, `_sign`, `_kwargs` automatically
- Handle `neg_*` scorers correctly

## Decision

Implement a metrics registry as a per-instance attribute on `EstimatorReport.metrics` accessor.

### API Design

#### Registration
```python
from sklearn.metrics import make_scorer

def business_loss(y_true, y_pred, cost_fp=10, cost_fn=5):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * cost_fp + fn * cost_fn

scorer = make_scorer(business_loss, greater_is_better=False, response_method="predict")
report.metrics.register(scorer)

# Also supports string names, will run `sklearn.metrics.get_scorer` same as now
report.metrics.register("f1")
```

#### Discovery
```python
# Access registry
report.metrics.registry  # Dict-like: {name: Metric}

# View metadata
metric = report.metrics.registry["business_loss"]
print(metric.name)                # "business_loss"
print(metric.verbose_name)        # "Business Loss"
print(metric.greater_is_better)   # False
```

#### Usage
```python
# Auto-included in summarize
report.metrics.summarize()  # Shows built-ins + registered metrics

# Explicit call
report.metrics.summarize(metric="business_loss")

# Mixed
report.metrics.summarize(metric=["accuracy", "business_loss"])
```

#### Serialization
```python
# Save to Project
project.put("my_report", report)

# Later / by collaborator
report2 = project.get("my_report")
report2.metrics.summarize()  # Previously registered custom metrics are present

# Review source for security
print(report2.metrics.registry["business_loss"].source_code)
```

### Core Behaviors

#### `register` supports sklearn _BaseScorers and sklearn metric strings

Skore should provide a Metric object/`make_metric` factory, but for convenience and
compatibility with existing workflow, it should also accept other things.

```python
# sklearn Scorer
scorer = make_scorer(my_func, greater_is_better=False, response_method="predict")
report.metrics.register(scorer)

# sklearn metric strings
report.metrics.register("f1")
report.metrics.register("neg_mean_squared_error")
```

The case could also be made for MLFlow [EvaluationMetrics](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html?highlight=metric#mlflow.metrics.EvaluationMetric).

Plain callables (without `make_scorer`) do not have first-class support, to incentivize providing more structured data.

#### Default metrics cannot be overwritten

Registration raises `ValueError` if metric name conflicts with built-ins.

**Protected names** (technical and verbose):
- accuracy, precision, recall, roc_auc, brier_score, log_loss
- r2, rmse, fit_time, predict_time

**Matching logic**: Check both:
- Technical name (normalized: strip `_score`, lowercase)
- Verbose name (case-insensitive)

Example:
```python
fake_accuracy = make_scorer(lambda y_true, y_pred: 1.0, response_method="predict")
fake_accuracy._score_func.__name__ = "accuracy_score"

report.metrics.register(fake_accuracy)  # ValueError: conflicts with built-in
```

#### Duplicate custom names are allowed

Re-registering with same name:
- Clears cache for that metric
- Replaces previous registration
- No error, no warning

This enables iterative development in notebooks.

However, this may not be sufficient considering the team collaboration setting: Bob might mistakenly defined a metric with the same name as one that was defined by Alice.
Future enhancements to mitigate this might include session tracking (was this metric registered in the current session?), a `force=True` flag, or warnings.

#### Lambda and Closures are allowed but raise a warning

These callables are typically not picklable.

```python
scorer = make_scorer(lambda y_true, y_pred: accuracy_score(y_true, y_pred),
                     response_method="predict")

report.metrics.register(scorer)  # UserWarning: may not survive pickling
```

#### Checks are done at registration time

- Check `response_method` exists on estimator
- Check name doesn't conflict with built-ins
- Warn if lambda/closure

**Not validated**:
- Whether metric is compatible with ML task (allow flexibility)
- Whether estimator is fitted (allow registration before fit)

#### sklearn scorers have first class support

Special handling for `neg_*` scorers:
```python
if name.startswith("neg_"):
    name = name[4:]  # Strip "neg_"
    verbose_name = verbose_name.replace("Neg ", "")
    # Sign already flipped by sklearn, favorability adjusted
```

Result: `neg_mean_squared_error` displays as "Mean Squared Error" with `(↘︎)` icon.

Note: Multimetric scorers are not supported. A single scorer that returns multiple *different* metrics is ambiguous:

```python
def multi_metric_scorer(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
```

The existing code would treat dict keys as "labels" (per-class scores) rather than different metrics, which is incorrect.

**Solution**: Users should register metrics separately:
```python
# ✅ Do this
report.metrics.register(accuracy_scorer)
report.metrics.register(precision_scorer)
report.metrics.register(recall_scorer)

# ❌ Not this
report.metrics.register(multi_metric_scorer)  # Ambiguous
```

**Rationale**:
- Multimetric scorers are rare in practice
- Registering separately is clearer and more explicit
- Avoids ambiguity with per-label dict returns
- Can add explicit support later if demand emerges (e.g., `register_multimetric()`)

#### Using the registry should not break report serialization

**Best-effort approach**:

✅ **Functions that pickle well**:
- Named module-level functions
- sklearn built-in scorers
- Functions with no captured variables

⚠️ **Functions that may fail**:
- Lambda functions
- Closures (functions capturing outer scope)
- Interactive/notebook-defined functions (without `dill`)

**On pickle failure**:
- Store `score_func = None`
- Preserve all metadata
- Preserve `source_code`
- Warn on unpickle

**After unpickle with failed function**:
```python
metric = report.metrics.registry["my_lambda"]
metric.is_callable()  # False
metric.source_code    # Still available for review
```

**Security consideration**: Captured source code allows review before trusting pickled functions.

#### Re-registering a metric leads to cache invalidation

Cache keys use the metric name. When a metric is re-registered, its cache entries are explicitly cleared:

**Benefits**:
- ✅ Simple - no extra state needed on `Metric`
- ✅ No fragile bytecode/source hashing

**Trade-off accepted**: Even trivial changes (whitespace) invalidate cache. This is intentional for correctness and simplicity.

## Consequences

### Positive

1. **Persistence**: Custom metrics travel with reports through Project storage
2. **Auto-inclusion**: Registered metrics appear in `summarize()` by default
3. **Discoverability**: `report.metrics.registry` lists all registered metrics
4. **Security**: Source code captured for review before trusting pickled functions
5. **Cache correctness**: Automatic invalidation on re-register prevents stale results
7. **Flexibility**: Supports sklearn scorers and string names
8. **Validation**: Fail-fast at registration time
9. **No breaking changes**: Entirely additive to existing API

### Negative

1. **Silent replacement**: Could accidentally overwrite in shared reports (deferred fix)
2. **Pickle limitations**: Lambdas/closures may not survive serialization
3. **Cache invalidation**: Even trivial changes (whitespace) clear cache (accepted trade-off)
4. **Scope limited**: EstimatorReport only; CrossValidation/Comparison deferred
5. **No metadata routing**: Deferred to future iteration

### Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| Silent replace | Iterative dev convenience | Risk of accidental overwrite (mitigated: future enhancement) |
| Best-effort pickle | Graceful degradation | Lambdas don't survive (mitigated: warning + source code) |
| Validation at registration | Fail fast | Can't register before estimator is created (minor) |
| Per-instance registry | Isolated per report | No global/class-level defaults (future enhancement) |

## Implementation Notes

### Metric Return Types

The registry supports metrics with various return types, leveraging skore's existing infrastructure:

Scalar returns (most common), dict returns for per-class/per-label scores, list returns for multi-output regression.

The registry doesn't need special handling for these cases: the existing `summarize()` logic in `_MetricsAccessor` already handles all return types correctly.

## Alternatives Considered

### Cache Invalidation

**Alternative 1: Hash source code**
```python
import inspect, hashlib
source = inspect.getsource(metric_fn)
func_hash = hashlib.md5(source.encode()).hexdigest()
```
❌ Rejected: Fragile, fails for lambdas/built-ins, whitespace-dependent

**Alternative 2: Hash bytecode**
```python
func_hash = hashlib.md5(metric_fn.__code__.co_code).hexdigest()
```
❌ Rejected: Python version dependent, optimization flag dependent, very fragile

**Alternative 3: Manual versioning**
```python
report.metrics.register(scorer, version="v2")
```
❌ Rejected: User burden, easy to forget, more API surface

**Alternative 4: Function `id()`**
```python
func_id = id(metric_fn)
```
❌ Rejected: Doesn't survive pickle, changes across sessions

**Alternative 5: UUID per registration**
- Simple, no fragile hashing
- Survives pickle
- But unnecessary: since we explicitly clear cache on re-register, there's no need to encode identity in the key itself

**✅ Selected: Explicit cache clear on re-registration**
- Metric name in cache key (same as built-ins)
- `_clear_cache_for_metric(name)` called before replacing a registration
- Simpler - no extra state on `Metric`

### Duplicate Name Handling

**Alternative 1: Raise error**
```python
if name in self._registry:
    raise ValueError(f"Metric '{name}' already registered")
```
❌ Rejected: Annoying for iterative development

**Alternative 2: Require `force=True`**
```python
report.metrics.register(scorer, force=True)
```
❌ Rejected: Extra API burden, still annoying

**Alternative 3: Warn on replace**
```python
if name in self._registry:
    warnings.warn(f"Replacing existing metric '{name}'")
```
❌ Rejected: Warning fatigue in notebooks

**✅ Selected: Silent replace**
- Best for iterative development
- Add collaboration protection later

### Registry API

**Alternative 1: Method call**
```python
report.metrics.list_registered()
```
❌ Rejected: Less intuitive than property

**Alternative 2: Named differently**
```python
report.metrics.custom_metrics
```
❌ Rejected: More verbose, no clearer

**✅ Selected: `report.metrics.registry`**
- Concise, clear
- Dict-like access pattern

## Future Enhancements

### Collaboration Protection

When reports are shared, prevent accidental overwriting:
- **Option A**: Track session ownership (`registered_in_session` flag)
- **Option B**: Require `force=True` to replace
- **Option C**: Warn if replacing metric with source code

Defer until collaboration features are actively used.

### Metadata Routing Support

Support passing additional data arrays (sample_weight, segment IDs, etc.) to metrics.

**See**: [ADR-metadata-routing.md](./ADR-metadata-routing.md) for complete design.

Metadata routing is orthogonal to the registry - it affects all metrics (built-in and registered). The registry design is compatible with future metadata routing support.

### Plain Callable Support

Register functions without `make_scorer()`:
```python
report.metrics.register(my_function, response_method="predict", greater_is_better=False)
```

### Version History

Track multiple versions of a metric, allow rollback:
```python
report.metrics.registry["my_metric"].history  # List of versions
report.metrics.revert("my_metric", version=2)
```

### Cache Statistics

```python
report.metrics.cache_stats()  # Show hits/misses per metric
```

### 7. Security Scanning

Automated scanning of pickled function source code for malicious patterns.

## Testing Strategy

### Test Coverage (12 test classes, ~55 tests)

1. **Basic Registration**: Register, discover, metadata access
2. **Summarize Integration**: Auto-include, explicit call, mixed usage
3. **Built-in Protection**: Name/verbose conflicts, all protected names
4. **Scorer Extraction**: Extract `_score_func`, `_response_method`, `_kwargs`, `_sign`
5. **neg_* Handling**: Sign flip, favorability, name cleaning
6. **Cache Behavior**: Caching, re-register invalidation, isolated cache entries
7. **Edge Cases**: Missing data, incompatible methods, duplicates
8. **Different ML Tasks**: Binary, multiclass, regression, multioutput
9. **Return Value**: Returns `None`
10. **String Scorers**: Register by name, metadata extraction
11. **Validation Timing**: Check response_method at registration
12. **Serialization**: Pickle/unpickle, lambdas, source code, metadata preservation

### Key Test Cases

- Re-registering clears only that metric's cache
- Lambda functions warn and degrade gracefully
- Source code captured and preserved
- Metadata always survives serialization
- Whitespace-only changes invalidate cache (via re-register)

## Appendix: User Stories

### Story 1: Register and use custom business metric
```python
from sklearn.metrics import make_scorer

def business_loss(y_true, y_pred, cost_fp=10, cost_fn=5):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * cost_fp + fn * cost_fn

scorer = make_scorer(business_loss, greater_is_better=False,
                     response_method="predict", cost_fp=10, cost_fn=5)

report.metrics.register(scorer)

# Auto-included
display = report.metrics.summarize()
assert "Business Loss" in display.data["metric"].values

# Explicit call
display = report.metrics.summarize(metric="business_loss")
```

### Story 2: Share report with custom metrics
```python
# User A creates report
report = EstimatorReport(estimator, **data)
report.metrics.register(my_custom_scorer)
project.put("team_report", report)

# User B retrieves report
report2 = project.get("team_report")

# Custom metric still works
report2.metrics.summarize()  # Includes custom metric

# Review source for security
print(report2.metrics.registry["my_custom_metric"].source_code)
```

### Story 3: Iterative development
```python
# Initial version
def my_metric(y_true, y_pred):
    return np.mean(y_true == y_pred)

scorer = make_scorer(my_metric, response_method="predict")
report.metrics.register(scorer)
report.metrics.summarize()  # Computes, caches

# Edit function
def my_metric(y_true, y_pred):
    return np.mean((y_true == y_pred) * 2)  # Changed!

# Re-register (clears cache automatically)
scorer = make_scorer(my_metric, response_method="predict")
report.metrics.register(scorer)
report.metrics.summarize()  # Computes fresh, correct result
```

### Story 4: Cannot override built-ins
```python
fake_accuracy = make_scorer(lambda y_true, y_pred: 1.0, response_method="predict")
fake_accuracy._score_func.__name__ = "accuracy_score"

# Raises ValueError
report.metrics.register(fake_accuracy)  # Error: conflicts with built-in
```

[#2061]: https://github.com/probabl-ai/skore/issues/2061
[#2120]: https://github.com/probabl-ai/skore/issues/2120
[#2203]: https://github.com/probabl-ai/skore/issues/2203
[#2277]: https://github.com/probabl-ai/skore/pull/2277
[#2001]: https://github.com/probabl-ai/skore/issues/2001
