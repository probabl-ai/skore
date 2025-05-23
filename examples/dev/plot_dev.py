"""
.. _example_dev:

=========================
Example driven production
=========================

This example demonstrates the DataFrame output functionality for various metric displays.
"""

# %%
# prediction_error
# ================

# %%
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from skore import train_test_split
from skore import EstimatorReport

X, y = load_diabetes(return_X_y=True)
split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
regressor = Ridge()
report = EstimatorReport(regressor, **split_data)

# Get prediction error display and its DataFrame
display = report.metrics.prediction_error()
display.plot(kind="actual_vs_predicted")
df_pred_error = display.frame()
df_pred_error.head()

# %%
# roc_curve
# =========

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(
    n_classes=2, n_clusters_per_class=2, n_informative=3, random_state=42
)
split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
classifier = LogisticRegression(max_iter=10_000)
report = EstimatorReport(classifier, **split_data)

# Get ROC curve display and its DataFrame
display = report.metrics.roc()
display.plot()
df_roc = display.frame()
df_roc.head()

# %%
# precision_recall_curve
# ======================

# Get Precision-Recall curve display and its DataFrame
display = report.metrics.precision_recall()
display.plot()
df_pr = display.frame()
df_pr.head()

# %%
