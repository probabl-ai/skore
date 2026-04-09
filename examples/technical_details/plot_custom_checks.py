"""
.. _example_custom_checks:

============================
Adding custom diagnostic checks
============================

`skore` lets you extend the built-in diagnostic checks with your own.
This example shows how to write a custom check function and register it
with a report via :meth:`~skore.EstimatorReport.add_checks`.

A check is any callable that accepts a single report argument and returns a
dictionary, mapping a check code to an issue dictionary with keys ``"title"``,
``"explanation"``, and optionally ``"docs_url"``.
When the check finds no issue, it should return an empty dictionary.
"""

# %%
# Writing a custom check for a single estimator
# ==============================================
#
# We start by defining a simple check that flags models with a very large
# number of features. The check inspects the test data attached to the
# report.

import numpy as np
from skore import Check, DiagnosticNotApplicable


def check_high_feature_count(report):
    """Flag when the number of features exceeds a threshold."""
    if report.X_test is None:
        raise DiagnosticNotApplicable()

    n_features = np.asarray(report.X_test).shape[1]
    if n_features > 50:
        return (
            f"The dataset has {n_features} features which may hurt model performance. "
            "Consider feature selection or dimensionality reduction."
        )

    return None


custom_check_1 = Check(
    check_high_feature_count,
    "CSTM001",
    "High feature count",
    "https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection",
    "estimatore",
)

# %%
# Registering the check
# =====================
#
# ``add_checks`` accepts a list of ``Check`` instances, and registers them.
from sklearn.linear_model import LinearRegression
from skore import evaluate

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 80))
y = X[:, 0] + rng.normal(size=200)

report = evaluate(LinearRegression(), X, y)
report.add_checks([custom_check_1])
report.diagnose()

# %%
# The ``docs_url`` key is optional. When provided as a full URL (starting
# with ``"https"``), it is used as-is. When it is a plain anchor string
# it points to the skore diagnostic user guide. When omitted entirely,
# no documentation link is shown.

# %%
# Cross-validation level checks
# ==============================
#
# :class:`~skore.CrossValidationReport` and :class:`~skore.ComparisonReport` can also
# receive custom checks, either ran on the full report or on the component estimator
# reports.
#
# By default, checks are run on the full report. To run checks on the component
# estimator reports and aggregate the results across splits, pass ``level="estimator"``
# to the ``add_checks`` method.
#
# In this example, we will corrupt the first fold of the target in a cross validation
# scheme to create high score variance across splits, then write a check that detects it.

import pandas as pd

y_noisy = y.copy()
y_noisy[: len(y_noisy) // 5] = rng.normal(size=len(y_noisy) // 5)
cv_report = evaluate(LinearRegression(), X, y_noisy, splitter=5)


def check_cv_score_variance(report):
    """Flag high score variance across CV splits."""
    frames = [
        sub_report.metrics.summarize(data_source="test").data
        for sub_report in report.estimator_reports_
    ]
    scores = pd.concat(frames, ignore_index=True)

    high_var_metrics = [
        metric_name
        for metric_name, group in scores.groupby("metric")
        if group["score"].std() > 0.1
    ]

    if high_var_metrics:
        return f"Metrics with high variance: {', '.join(high_var_metrics)}."
    return None


custom_check_2 = Check(
    check_cv_score_variance,
    "CSTM002",
    "High score variance across folds",
    "",
    "cross-validation",
)


cv_report.add_checks([custom_check_2, custom_check_1])
cv_report.diagnose()
