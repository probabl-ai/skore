"""
.. _example_custom_checks:

===============================
Adding custom diagnostic checks
===============================

`skore` lets you extend the built-in diagnostic checks with your own.
This example shows how to write a custom check function and register it
with a report via :meth:`~skore.EstimatorReport.add_checks`.
"""

# %%
# Writing a custom check for a single estimator
# =============================================
#
# We start by defining a simple check that flags models with a very large
# number of features. The check inspects the test data attached to the
# report. We throw an exception when the test data is not available to avoid
# running the check when it is not applicable. The check function is wrapped in a
# :class:`~skore.Check` instance and registered with the report via
# :meth:`~skore.EstimatorReport.add_checks`.
#
# The `docs_url` argument is optional. When provided as a full URL (starting
# with ``"https"``), it is used as-is. When it is a plain anchor string
# it points to the skore diagnostic user guide. When omitted entirely,
# no documentation link is shown.

import numpy as np
from skore import Check, DiagnosticNotApplicable


class CustomCheck1(Check):
    code = "CSTM001"
    title = "High feature count"
    report_type = "estimator"
    docs_url = "https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection"

    def check_function(self, report):
        """Flag when the number of features exceeds a threshold."""
        if report.X_test is None:
            raise DiagnosticNotApplicable()

        n_features = X.shape[1]
        if n_features > 50:
            return (
                f"The dataset has {n_features} features which may hurt model performance. "
                "Consider feature selection or dimensionality reduction."
            )
        return None


# %%
# Registering the check
# =====================
#
# :meth:`~skore.EstimatorReport.add_checks` accepts a list of ``Check`` instances,
# and registers them. The next call to :meth:`~skore.EstimatorReport.diagnose` runs
# any newly added checks on top of the built-in checks.
from sklearn.linear_model import LinearRegression
from skore import evaluate

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 80))
y = X[:, 0] + rng.normal(size=200)

report = evaluate(LinearRegression(), X, y)
report.add_checks([CustomCheck1()])
report.diagnose()

# %%
# Cross-validation level checks
# =============================
#
# :class:`~skore.CrossValidationReport` and :class:`~skore.ComparisonReport` can also
# receive custom checks, either ran on the full report or on the component estimator
# reports.
#
# The `report_type` argument of :class:`~skore.Check` controls the scope of the check.
# Let's write a check that is specific to cross-validation reports: it flags metrics
# with high variance across splits.
#
# We will corrupt the first fold of the target to illustrate the check.
import pandas as pd

y_noisy = y.copy()
y_noisy[: len(y_noisy) // 5] = rng.normal(size=len(y_noisy) // 5)
cv_report = evaluate(LinearRegression(), X, y_noisy, splitter=5)


class CustomCheck2(Check):
    code = "CSTM002"
    title = "High score variance across CV splits"
    report_type = "cross-validation"
    docs_url = None

    def check_function(self, report):
        """Flag high score variance across CV splits."""
        frames = [
            sub_report.metrics.summarize(data_source="test").data
            for sub_report in report.estimator_reports_
        ]
        scores = pd.concat(frames, ignore_index=True)

        high_var_metrics = [
            metric_name
            for metric_name, group in scores.groupby("metric_verbose_name")
            if group["score"].std() > 0.1
        ]

        if high_var_metrics:
            return f"Metrics with high variance: {', '.join(high_var_metrics)}."
        return None


cv_report.add_checks([CustomCheck2()])
cv_report.diagnose()

# %%
# Aggregating checks across estimator reports
# ===========================================
#
# We can also reuse our first check to run it on the component estimator reports
# and aggregate the results across splits.

cv_report.add_checks([CustomCheck1()])
cv_report.diagnose()

# %%
# Similarly, :class:`~skore.ComparisonReport` aggregates checks across its
# component reports.
from sklearn.ensemble import RandomForestRegressor

comparison_report = evaluate(
    [LinearRegression(), RandomForestRegressor()], X, y, splitter=5
)
comparison_report.add_checks([CustomCheck1(), CustomCheck2()])
comparison_report.diagnose()
