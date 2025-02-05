"""
.. _example_compare_estimators:

============================================
Example driven development - Compare feature
============================================
"""

# %%
# From your Python code, create and load a skore :class:`~skore.Project`:

# %%
import skore

# sphinx_gallery_start_ignore
import os
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = Path(temp_dir.name)
os.chdir(temp_dir_path)
# sphinx_gallery_end_ignore
my_project = skore.Project("my_project")

# %%
# This will create a skore project directory named ``my_project.skore`` in your
# current working directory.

# %%
# Evaluate your model using skore's :class:`~skore.EstimatorReport`:

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from skore import EstimatorReport, train_test_split

X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
logistic_regression = LogisticRegression()
estimator_report_logistic_regression = EstimatorReport(
    logistic_regression,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# %%
# Now let us create a new model, hopefully better than the previous one, with the same data
random_forest = RandomForestClassifier(max_depth=2, random_state=0)
estimator_report_random_forest = EstimatorReport(
    random_forest,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)

# %%
comp = skore.ComparisonReport(reports=[estimator_report_logistic_regression, estimator_report_random_forest,])

# %%
# Compute the accuracy for each estimator
comp.metrics.accuracy()

# %%
# Plot accuracy using pandas API
comp.metrics.accuracy().plot.bar()

# %%
import pandas as pd


def highlight_max(s):
    # thanks chatgpt
    return ["font-weight: bold" if v == s.max() else "" for v in s]


df_metrics = pd.concat(
    [estimator_report_logistic_regression.metrics.report_metrics(), estimator_report_random_forest.metrics.report_metrics()],
    ignore_index=False,
)

# %%
#
#   .. code-block:: python
#
#       comp.metrics.metric()
#       comp.metrics.custom_metric(scoring=my_func)
#

df_metrics.style.apply(highlight_max, axis=0)

# %%
# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore
# 
