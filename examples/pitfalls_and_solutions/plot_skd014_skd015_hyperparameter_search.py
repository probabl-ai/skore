"""
.. _example_skd014_hyperparams_at_search_edge_skd015_hyperparameters_worth_tuning:

SKD014 & SKD015 - Hyperparameter search pitfalls
================================================

This example walks through mitigations when checks
:ref:`SKD014 <skd014-hyperparams-at-search-edge>` and
:ref:`SKD015 <skd015-hyperparameters-worth-tuning>` fire on a fitted search
object. SKD014 is an issue: numeric ``best_params_`` sit on the minimum or
maximum value tried, so the true optimum may lie outside the searched range.
SKD015 is a tip: high-impact axes are missing from the search space, which
is incomplete rather than necessarily wrong.

These findings often appear together on the same
:class:`~sklearn.model_selection.BaseSearchCV` report. The user guide
recommends addressing them together when both fire.

Mitigations from the :ref:`automated_checks` user guide:

**SKD014 - hyperparameters at search edge** (issue)

- extend ``param_grid`` or ``param_distributions`` beyond the flagged bounds,
- for :class:`~sklearn.model_selection.RandomizedSearchCV`, increase ``n_iter``
  and sample from a wider range,
- if SKD015 also fires, widen the search on every recommended axis.

**SKD015 - hyperparameters worth tuning** (tip)

- add the suggested parameters to ``param_grid`` or ``param_distributions``.

We tune a :class:`~sklearn.ensemble.HistGradientBoostingClassifier` inside
:func:`~skrub.tabular_pipeline` on the employee salaries dataset (above-median
salary as the positive class). The walkthrough has three beats: missing axes
(SKD015), edge hits (SKD014), then one joint fix that clears both.
"""

# %%
# Load the employee salaries dataset
# ==================================
#
# We predict whether salary exceeds the median. Mixed HR features suit
# :func:`~skrub.tabular_pipeline`.

import pandas as pd
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X = dataset.X
y_salary = dataset.y.squeeze()
y = pd.Series(
    (y_salary > y_salary.median()).astype(int),
    name="high_earner",
    index=X.index,
)

# %%
# Inspect predictors and the binary target with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
TableReport(y)

# %%
# Shared search setup
# ===================
#
# The base object is ``tabular_pipeline`` around HGB. Outer hold-out and inner
# ``cv=3`` folds are stratified (no time ordering required for these checks).

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skore import TrainTestSplit
from skrub import tabular_pipeline

splitter = TrainTestSplit(test_size=0.2, random_state=42, stratify=y)
cv = 3
base_pipeline = tabular_pipeline(HistGradientBoostingClassifier(random_state=42))

# %%
# Trigger SKD015 - grid with only ``max_iter``
# ============================================
#
# ``max_iter`` is a budget parameter, not a complexity knob in the SKD015 table.
# Searching only iteration count leaves learning rate, depth, and leaf size at
# defaults - the cleanest way to see SKD015 without an edge-of-grid effect.

from skore import evaluate

max_iter_only_search = GridSearchCV(
    base_pipeline,
    param_grid={
        "histgradientboostingclassifier__max_iter": [100, 200, 300],
    },
    cv=cv,
    n_jobs=4,
    refit=True,
)

report_axes = evaluate(
    max_iter_only_search,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# SKD015 should tip that learning rate, depth, and leaf size were not searched.
# ``best_params_`` only contains ``max_iter`` - that incompleteness is the point.

report_axes.checks.summarize(fast_mode=True)

# %%
report_axes.estimator_.best_params_

# %%
# A search that only tweaks training budget ignores the axes that usually move
# generalization for tree ensembles.

# %%
# Trigger SKD014 - narrow ``RandomizedSearchCV``
# ==============================================
#
# With only eight random draws and tight bounds, ``best_params_`` often lands on
# the minimum or maximum tried on at least one axis. Compare the issue to
# ``best_params_`` below: edge hits mean the optimum may lie outside the box you
# searched.

narrow_search = RandomizedSearchCV(
    base_pipeline,
    param_distributions={
        "histgradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1],
        "histgradientboostingclassifier__max_iter": [100, 200, 300],
        "histgradientboostingclassifier__max_depth": [3, 5, 8],
        "histgradientboostingclassifier__min_samples_leaf": [10, 20, 50],
    },
    n_iter=8,
    cv=cv,
    random_state=42,
    n_jobs=4,
    refit=True,
)

report_edge = evaluate(
    narrow_search,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# SKD014 should list numeric parameters at search edges as an issue; SKD015 may
# tip as well if a recommended axis is still missing from the distributions.

report_edge.checks.summarize(fast_mode=True)

# %%
report_edge.estimator_.best_params_

# %%
report_edge.metrics.summarize(data_source="both").frame()

# %%
# SKD014 & SKD015 - widen bounds and add every recommended axis
# =============================================================
#
# When both checks fire (or you want one fix for both stories), extend values
# beyond the previous edges and include the high-impact axes SKD015 asks
# for. Prior boundary bests become interior grid points when the box grows.

full_search = GridSearchCV(
    base_pipeline,
    param_grid={
        "histgradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1],
        "histgradientboostingclassifier__max_iter": [50, 100, 200, 300],
        "histgradientboostingclassifier__max_depth": [2, 3, 5, 8, None],
        "histgradientboostingclassifier__min_samples_leaf": [10, 20, 50, 100],
    },
    cv=cv,
    n_jobs=4,
    refit=True,
)

report_full = evaluate(
    full_search,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# SKD014 and SKD015 often clear together once the grid is complete and wide
# enough for interior bests. Check ``best_params_``: values should sit strictly
# inside the ranges you tried when SKD014 is gone.

report_full.checks.summarize(fast_mode=True)

# %%
report_full.estimator_.best_params_

# %%
# Compare search strategies
# =========================
#
# Hold-out metrics for the incomplete grid, the narrow random search, and the
# joint fix. Clearing the checks means the search design improved; still judge
# models on validation metrics and cost, not check status alone.

from skore import compare

compare(
    {
        "max_iter_only": report_axes,
        "narrow_random_search": report_edge,
        "full_recommended_axes": report_full,
    }
).metrics.summarize(data_source="both").frame()

# %%
# Conclusion
# ==========
#
# SKD015 tips about incomplete search spaces; SKD014 issues when optima stick to
# the boundary of the box you tried. In this walkthrough, a ``max_iter``-only grid
# missed key axes, a narrow random search hit edges, and one wider complete
# grid addressed both. Expand the search before deploying ``best_params_`` -
# green checks are about search hygiene, not a guarantee of the best model.
