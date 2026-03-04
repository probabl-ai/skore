"""
.. _example_use_case_classification:

==========================
Classification Use Case
==========================

This example shows how to use skore to evaluate a classification model
on a real-world-style problem, get methodological guidance, and compare
multiple models efficiently.
"""

# %%
# Setup
# =====
#
# We use a synthetic classification dataset to simulate a real project.

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skore import ComparisonReport, CrossValidationReport

X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_classes=2,
    random_state=42,
)

# %%
# Evaluate a single model
# =======================
#
# :class:`~skore.CrossValidationReport` gives us a structured view of model
# performance. We start by evaluating a simple logistic regression.

clf = LogisticRegression(max_iter=1000)
cv_report = CrossValidationReport(clf, X, y, splitter=5)
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
# Plot the ROC curve across folds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ROC curve is a useful visualization for understanding the trade-off
# between the true positive rate and the false positive rate.

roc_display = cv_report.metrics.roc()
roc_display.plot()

# %%
# Compare multiple models
# =======================
#
# :class:`~skore.ComparisonReport` helps us benchmark different classifiers
# side by side. Here we compare LogisticRegression with a RandomForest.

clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)

report1 = CrossValidationReport(clf1, X, y, splitter=5)
report2 = CrossValidationReport(clf2, X, y, splitter=5)

comparison = ComparisonReport([report1, report2])
comparison.metrics.summarize().frame()

# %%
# Conclusion
# ==========
#
# This example demonstrated a typical classification workflow using skore:
#
# - Using :class:`~skore.CrossValidationReport` to evaluate a single classifier
# - Inspecting performance with metrics and ROC curves
# - Using :class:`~skore.ComparisonReport` to compare multiple classifiers
