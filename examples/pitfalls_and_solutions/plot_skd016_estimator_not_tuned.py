"""
.. _example_skd016_estimator_not_tuned:

SKD016 - Estimator not tuned
============================

This example walks through mitigations when check
:ref:`SKD016 <skd016-estimator-not-tuned>` tips on a plain estimator left at
scikit-learn defaults. The check compares initialization parameters against a
curated table of high-impact hyperparameters and suggests axes worth tuning.

Mitigations from the :ref:`automated_checks` user guide:

- wrap the estimator in :class:`~sklearn.model_selection.GridSearchCV` or
  :class:`~sklearn.model_selection.RandomizedSearchCV` over the suggested
  parameters,
- or set sensible non-default values manually.

We use the employee salaries dataset (above-median salary as the positive
class) with a default :func:`~skrub.tabular_pipeline` classifier. The goal is
to move off factory defaults either through search or hand-picked values.
"""

# %%
# Load the employee salaries dataset
# ==================================
#
# Mixed HR features suit ``tabular_pipeline``. A default
# ``tabular_pipeline("classifier")`` leaves
# :class:`~sklearn.ensemble.HistGradientBoostingClassifier` at sklearn
# defaults — the setup SKD016 is designed to flag.

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
# Inspect inputs and the binary target with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
TableReport(y)

# %%
# We use the same stratified split for every comparison.

from skore import TrainTestSplit

splitter = TrainTestSplit(test_size=0.2, random_state=42, stratify=y)

# %%
# Trigger SKD016 - untuned default pipeline
# =========================================
#
# Defaults are fine for a first look at the table, but they are not a production
# configuration. SKD016 names the high-impact axes that usually matter first for
# this estimator family.

import skore
from skrub import tabular_pipeline

report = skore.evaluate(
    tabular_pipeline("classifier"),
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# SKD016 should tip that HistGradientBoostingClassifier remains at defaults.
# Read which parameters it lists — the next sections search or set those knobs.

report.checks.summarize()

# %%
report.metrics.summarize(data_source="both").frame()

# %%
# Wrap the estimator in RandomizedSearchCV
# ========================================
#
# Search the axes SKD016 typically flags for HGB (learning rate, iteration
# budget, depth, leaf size) instead of accepting sklearn defaults. Once the
# report wraps a fitted search object, SKD016 clears.
#
# A tuned search can still raise :ref:`SKD014 <skd014-hyperparams-at-search-edge>`
# or :ref:`SKD015 <skd015-hyperparameters-worth-tuning>` if the box is too narrow
# or incomplete — see that combined example.

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

base_pipeline = tabular_pipeline(HistGradientBoostingClassifier(random_state=42))

param_distributions = {
    "histgradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "histgradientboostingclassifier__max_iter": [100, 200, 300, 400],
    "histgradientboostingclassifier__max_depth": [3, 5, 8, None],
    "histgradientboostingclassifier__min_samples_leaf": [10, 20, 50],
}

tuned_search = RandomizedSearchCV(
    base_pipeline,
    param_distributions=param_distributions,
    n_iter=8,
    cv=3,
    random_state=42,
    n_jobs=4,
    refit=True,
)

report_tuned = skore.evaluate(
    tuned_search,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# SKD016 should be absent; the report wraps a fitted search object.

report_tuned.checks.summarize(fast_mode=True)

# %%
report_tuned.estimator_.best_params_

# %%
# Set sensible non-default values manually
# ========================================
#
# When a full search is impractical, hand-pick hyperparameters that differ from
# defaults. SKD016 clears as soon as impactful knobs are no longer factory
# settings — that is an intentional configuration signal, not proof that the
# values are optimal. Prefer validated search when you can afford it.

model_manual = tabular_pipeline(
    HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42,
    )
)

report_manual = skore.evaluate(
    model_manual,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# SKD016 should be absent once hyperparameters differ from defaults.

report_manual.checks.summarize(fast_mode=True)

# %%
report_manual.metrics.summarize(data_source="both").frame()

# %%
# Compare mitigations
# ===================
#
# Hold-out metrics for the default pipeline and the hand-tuned one. The
# RandomizedSearchCV report wraps a search estimator, so its metric table can
# look different in :func:`~skore.compare`; we show it in its own cell below.

skore.compare(
    {
        "default_pipeline": report,
        "hand_tuned_hgb": report_manual,
    }
).metrics.summarize(data_source="both").frame()

# %%
report_tuned.metrics.summarize(data_source="both").frame()

# %%
# Conclusion
# ==========
#
# SKD016 nudges you off scikit-learn defaults for high-impact estimators.
# Randomized search and hand-tuned HGB parameters both clear the tip; clearing
# the check means you left factory settings, not that the model is finished.
# Pair manual choices with periodic search, and watch SKD014/SKD015 once you
# wrap a ``BaseSearchCV``.
