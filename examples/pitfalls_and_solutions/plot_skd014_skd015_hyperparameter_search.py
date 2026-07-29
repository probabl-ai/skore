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
recommends addressing them jointly when both fire.

Mitigations from the :ref:`automated_checks` user guide:

**SKD014 - hyperparameters at search edge** (issue)

- extend ``param_grid`` or ``param_distributions`` beyond the flagged bounds,
- for :class:`~sklearn.model_selection.RandomizedSearchCV`, increase ``n_iter``
  and sample from a wider range,
- if SKD015 also fires, widen the search on every recommended axis.

**SKD015 - hyperparameters worth tuning** (tip)

- add the suggested parameters to ``param_grid`` or ``param_distributions``.

We tune a :class:`~sklearn.ensemble.HistGradientBoostingClassifier` inside
:func:`~skrub.tabular_pipeline` on a stratified subsample of the employee
salaries dataset (above-median salary as the positive class). The walkthrough
has three beats: missing axes (SKD015), edge hits (SKD014), then one joint fix
that clears both.
"""

# %%
# Load the employee salaries dataset
# ==================================
#
# We predict whether salary exceeds the median. Mixed HR features suit
# :func:`~skrub.tabular_pipeline`. A 3,000-row stratified subsample keeps the
# gallery grids short while preserving class balance.

import pandas as pd
from sklearn.model_selection import train_test_split
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X_full = dataset.X
y_salary = dataset.y.squeeze()
y_full = pd.Series(
    (y_salary > y_salary.median()).astype(int),
    name="high_earner",
    index=X_full.index,
)

X, _, y, _ = train_test_split(
    X_full,
    y_full,
    train_size=3_000,
    stratify=y_full,
    random_state=42,
)
y = pd.Series(y, name="high_earner")

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
from sklearn.model_selection import GridSearchCV
from skore import TrainTestSplit
from skrub import tabular_pipeline

splitter = TrainTestSplit(test_size=0.2, random_state=42, stratify=y)
cv = 3
base_pipeline = tabular_pipeline(
    HistGradientBoostingClassifier(max_iter=200, random_state=42)
)

# %%
# Trigger SKD015 - grid with only ``max_iter``
# ============================================
#
# ``max_iter`` is a budget parameter, not a complexity knob in the SKD015 table.
# A single-value grid still runs a search, but SKD014 skips axes with fewer than
# two distinct tried values, so this beat isolates the SKD015 tip.

from skore import evaluate

max_iter_only_search = GridSearchCV(
    base_pipeline,
    param_grid={
        "histgradientboostingclassifier__max_iter": [200],
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
# A search that only tweaks training budget ignores the axes that usually move
# generalization for tree ensembles.

report_axes.checks.summarize(fast_mode=True)

# %%
report_axes.estimator_.best_params_

# %%
# Trigger SKD014 - two-point ``GridSearchCV``
# ==========================================
#
# With exactly two values on each searched axis, whichever value wins is always
# the tried minimum or maximum, so SKD014 fires deterministically. We still omit
# depth / leaf axes so SKD015 tips as well. Prefer a small grid over
# :class:`~sklearn.model_selection.RandomizedSearchCV` here: every candidate is
# evaluated, and the edge story does not depend on which draws were sampled.

edge_search = GridSearchCV(
    base_pipeline,
    param_grid={
        "histgradientboostingclassifier__learning_rate": [0.05, 0.1],
        "histgradientboostingclassifier__max_iter": [100, 200],
    },
    cv=cv,
    n_jobs=4,
    refit=True,
)

report_edge = evaluate(
    edge_search,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# SKD014 should list numeric parameters at search edges as an issue; SKD015
# should tip because depth / leaf axes are still missing.

report_edge.checks.summarize(fast_mode=True)

# %%
report_edge.estimator_.best_params_

# %%
report_edge.metrics.summarize(data_source="both").frame()

# %%
# SKD014 & SKD015 - widen bounds and add recommended axes
# =======================================================
#
# Pad beyond the previous two-point edges so those learning-rate values become
# interior grid points, and add ``max_depth`` so SKD015 clears (``max_depth``
# covers the tree-complexity family). ``None`` in ``max_depth`` is non-numeric,
# so SKD014 ignores that axis and only watches learning rate - fewer ways for
# the gallery to flake.

full_search = GridSearchCV(
    base_pipeline,
    param_grid={
        "histgradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "histgradientboostingclassifier__max_depth": [3, 5, None],
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
# SKD015 should clear once a recommended complexity axis is present with
# learning rate. SKD014 clears when the best learning rate sits strictly inside
# ``[0.01, 0.05, 0.1, 0.2]`` (not at the padded ends). Check ``best_params_``
# against the grid above.

report_full.checks.summarize(fast_mode=True)

# %%
report_full.estimator_.best_params_

# %%
# Compare search strategies
# =========================
#
# Hold-out metrics for the incomplete grid, the two-point edge grid, and the
# joint fix. Clearing the findings means the *search design* improved; still
# judge models on validation metrics and cost, not check status alone.

from skore import compare

compare(
    {
        "max_iter_only": report_axes,
        "two_point_edge_grid": report_edge,
        "padded_recommended_axes": report_full,
    }
).metrics.summarize(data_source="both").frame()

# %%
# Conclusion
# ==========
#
# SKD015 tips about incomplete search spaces; SKD014 issues when optima stick to
# the boundary of the box you tried. In this walkthrough, a ``max_iter``-only
# search missed key axes, a two-point grid forced edge hits, and one padded
# complete grid addressed both. Expand the search before deploying
# ``best_params_`` - green checks are about search hygiene, not a guarantee of
# the best model.
