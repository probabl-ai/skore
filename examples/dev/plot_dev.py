"""
.. _example_dev:

===
Dev
===
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
classifier = Ridge()
report = EstimatorReport(classifier, **split_data)
display = report.metrics.prediction_error()
display.plot(kind="actual_vs_predicted")

# %%
display.frame()

# %%
# display.y_true

# %%
# display.y_pred

# %%
display.frame()

# %%
# roc
# ===

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = X, y = make_classification(
    n_classes=3, n_clusters_per_class=2, n_informative=3, random_state=42
)
split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
classifier = LogisticRegression(max_iter=10_000)
report = EstimatorReport(classifier, **split_data)
display = report.metrics.roc()
display.plot()

# %%
display.ml_task

# %%
display.tpr

# %%
display.fpr

# %%
display.frame()
# %%
