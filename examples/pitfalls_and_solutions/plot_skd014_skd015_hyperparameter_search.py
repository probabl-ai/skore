"""
.. _example_skd014_hyperparams_at_search_edge_skd015_hyperparameters_worth_tuning:

SKD014 & SKD015: Hyperparameter search pitfalls
===============================================

This example walks through mitigations when checks
:ref:`SKD014 <skd014-hyperparams-at-search-edge>` and
:ref:`SKD015 <skd015-hyperparameters-worth-tuning>` fire on a fitted search
object. SKD014 is an issue: numeric ``best_params_`` sit on the minimum or
maximum value tried, so the true optimum may lie outside the searched range.
SKD015 is a tip: a hyperparameter is missing from the search space, which
is incomplete rather than necessarily wrong.

Usually, those tests are related to the search CV object and we advocate to
address them jointly when both fire.

Mitigations from the :ref:`automated_checks` user guide:

**SKD014: hyperparameters at search edge** (issue)

- extend ``param_grid`` or ``param_distributions`` beyond the flagged bounds,
- for :class:`~sklearn.model_selection.RandomizedSearchCV`, increase ``n_iter``
  and sample from a wider range,
- if SKD015 also fires, widen the search on every recommended hyperparameter.

**SKD015: hyperparameters worth tuning** (tip)

- add the suggested parameters to ``param_grid`` or ``param_distributions``.

In this example, we tune a
:class:`~sklearn.ensemble.HistGradientBoostingClassifier` inside
:func:`~skrub.tabular_pipeline` on a stratified subsample of the employee
salaries dataset (above-median salary as the positive class). The walkthrough
has three parts: missing hyperparameters (SKD015), edge hits (SKD014), then
one joint fix that clears both.
"""

# %%
# Load the employee salaries dataset
# ==================================
#
# The raw target is continuous salary. We turn the regression problem into a
# binary classification task: predict whether an employee earns more than the
# median salary among employees in the dataset. Mixed HR features suit
# :func:`~skrub.tabular_pipeline`. A 3,000-row stratified subsample keeps the
# gallery grids short while preserving class balance.

from sklearn.model_selection import train_test_split
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X_full = dataset.X
y_full = (dataset.y > dataset.y.median()).astype(int).rename("high_earner")

X, _, y, _ = train_test_split(
    X_full,
    y_full,
    train_size=3_000,
    stratify=y_full,
    random_state=42,
)

# %%
# Let us inspect predictors and the binary target with
# :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
TableReport(y)

# %%
# Shared search setup
# ===================
#
# Let us wrap HGB in :func:`~skrub.tabular_pipeline`. Early stopping lets a
# wide ``max_iter`` grid pick an interior budget in the first beat below. The
# outer hold-out uses :class:`~skore.TrainTestSplit` when we call
# :func:`~skore.evaluate`; each
# :class:`~sklearn.model_selection.GridSearchCV` below sets its own inner
# ``cv``.

from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import tabular_pipeline

base_pipeline = tabular_pipeline(
    HistGradientBoostingClassifier(
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
)
base_pipeline

# %%
# Trigger SKD015: grid with only ``max_iter``
# ===========================================
#
# ``max_iter`` is a budget parameter, not a complexity knob in the SKD015 table.
# We use :class:`~sklearn.model_selection.GridSearchCV` (not randomized search)
# so every candidate is evaluated and the run is reproducible. The grid is wide
# on purpose: with early stopping, the best ``max_iter`` lands strictly inside
# the list, so SKD014 stays quiet and this beat isolates the SKD015 tip about
# missing recommended hyperparameters.

from sklearn.model_selection import GridSearchCV
from skore import TrainTestSplit, evaluate

splitter = TrainTestSplit(test_size=0.2, random_state=42, stratify=y)

max_iter_only_search = GridSearchCV(
    base_pipeline,
    param_grid={
        "histgradientboostingclassifier__max_iter": [
            10,
            25,
            50,
            100,
            200,
            500,
            1000,
        ],
    },
    cv=3,
    scoring="neg_log_loss",
    n_jobs=4,
    refit=True,
)

report = evaluate(
    max_iter_only_search,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# In the Tips tab, SKD015 should tip that learning rate, depth, and leaf size
# were not searched. SKD014 should not fire here: the best ``max_iter`` is not
# the minimum or maximum of the grid above. ``best_params_`` only contains
# ``max_iter``: that incompleteness is the point. A search that only tweaks
# training budget ignores the hyperparameters that usually move generalization
# for tree ensembles.

report.checks.summarize(fast_mode=True)

# %%
report.estimator_.best_params_

# %%
# Trigger SKD014: two-point ``GridSearchCV``
# ==========================================
#
# With exactly two values on each searched hyperparameter, whichever value wins
# is always the tried minimum or maximum, so SKD014 fires deterministically. We
# still omit depth / leaf hyperparameters so SKD015 tips as well. Prefer a small
# grid over :class:`~sklearn.model_selection.RandomizedSearchCV` here: every
# candidate is evaluated, and the edge story does not depend on which draws were
# sampled.

edge_search = GridSearchCV(
    base_pipeline,
    param_grid={
        "histgradientboostingclassifier__learning_rate": [0.05, 0.1],
        "histgradientboostingclassifier__max_iter": [100, 200],
    },
    cv=3,
    scoring="neg_log_loss",
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
# SKD014 should list numeric parameters at search edges as an issue; in the Tips
# tab, SKD015 should tip because depth / leaf hyperparameters are still missing.

report_edge.checks.summarize(fast_mode=True)

# %%
report_edge.estimator_.best_params_

# %%
report_edge.metrics.summarize(data_source="both").frame()

# %%
# SKD014 & SKD015: widen bounds and add recommended hyperparameters
# =================================================================
#
# Let us pad beyond the previous two-point edges so those learning-rate values
# become interior grid points, and add ``max_depth`` so SKD015 clears
# (``max_depth`` covers the tree-complexity family). ``None`` in ``max_depth``
# is non-numeric, so SKD014 ignores that hyperparameter and only watches
# learning rate: fewer ways for the gallery to flake.

full_search = GridSearchCV(
    base_pipeline,
    param_grid={
        "histgradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "histgradientboostingclassifier__max_depth": [3, 5, None],
    },
    cv=3,
    scoring="neg_log_loss",
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
# SKD015 should clear once a recommended complexity hyperparameter is present
# with learning rate. SKD014 clears when the best learning rate sits strictly
# inside ``[0.01, 0.05, 0.1, 0.2]`` (not at the padded ends). Check
# ``best_params_`` against the grid above.

report_full.checks.summarize(fast_mode=True)

# %%
report_full.estimator_.best_params_

# %%
# Compare search strategies
# =========================
#
# Hold-out log-loss and ROC AUC for the incomplete grid, the two-point edge
# grid, and the joint fix. Clearing the findings means the *search design*
# improved; still judge models on validation metrics and cost, not check status
# alone.

from skore import compare

metrics = (
    compare(
        {
            "max_iter_only": report,
            "two_point_edge_grid": report_edge,
            "padded_recommended_params": report_full,
        }
    )
    .metrics.summarize(data_source="both", metric=["log_loss", "roc_auc"])
    .frame()
)
metrics.transpose()

# %%
# Conclusion
# ==========
#
# SKD015 tips about incomplete search spaces; SKD014 issues when optima stick to
# the boundary of the box you tried. In this walkthrough, a ``max_iter``-only
# search missed key hyperparameters, a two-point grid forced edge hits, and one
# padded complete grid addressed both. Expand the search before deploying
# ``best_params_``: passing checks are about search hygiene, not a guarantee of
# the best model.
