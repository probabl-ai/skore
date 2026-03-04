"""
.. _example_use_case_regression:

=======================
Regression Use Case
=======================

This example shows how to use skore to evaluate a regression model,
inspect prediction errors, and compare multiple regression models.
"""

# %%
# Setup
# =====
#
# We use a synthetic regression dataset to simulate a real project.

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from skore import ComparisonReport, CrossValidationReport

X, y = make_regression(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    noise=0.1,
    random_state=42,
)

# %%
# Evaluate a single model
# =======================
#
# :class:`~skore.CrossValidationReport` gives us a structured view of model
# performance. We start by evaluating a Ridge regression.

reg = Ridge()
cv_report = CrossValidationReport(reg, X, y, splitter=5)
cv_report.help()

# %%
# View the metrics summary
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The metrics summary provides an aggregated view of the cross-validated
# performance across all folds.

metrics_summary = cv_report.metrics.summarize().frame()
print(metrics_summary)

# %%
# Plot the prediction error
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The prediction error plot shows how well the model's predictions match
# the actual target values.

pred_error = cv_report.metrics.prediction_error()
pred_error.plot()

# %%
# Compare multiple models
# =======================
#
# :class:`~skore.ComparisonReport` helps us benchmark different regressors
# side by side. Here we compare Ridge with a GradientBoostingRegressor.

reg1 = Ridge()
reg2 = GradientBoostingRegressor(n_estimators=100, random_state=42)

report1 = CrossValidationReport(reg1, X, y, splitter=5)
report2 = CrossValidationReport(reg2, X, y, splitter=5)

comparison = ComparisonReport([report1, report2])
comparison.metrics.summarize().frame()

# %%
# Conclusion
# ==========
#
# This example demonstrated a typical regression workflow using skore:
#
# - Using :class:`~skore.CrossValidationReport` to evaluate a single regressor
# - Inspecting performance with metrics and prediction error plots
# - Using :class:`~skore.ComparisonReport` to compare multiple regressors
