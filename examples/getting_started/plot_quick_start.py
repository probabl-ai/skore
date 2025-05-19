"""
.. _example_quick_start:

===========
Quick start
===========
"""

# %%
# Machine learning evaluation and diagnostics
# ===========================================
#
# Evaluate your model using skore's :class:`~skore.EstimatorReport`:

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skore import EstimatorReport

X, y = make_classification(n_classes=2, n_samples=20_000, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

log_report = EstimatorReport(
    LogisticRegression(), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
# Display the help tree to see all the insights that are available to you (skore
# detected that you are doing binary classification):

# %%
log_report.help()

# %%
# Display the report metrics that was computed for you:

# %%
df_report_metrics = log_report.metrics.report_metrics(pos_label=1)
df_report_metrics

# %%
# Display the ROC curve that was generated for you:

# %%
roc_plot = log_report.metrics.roc()
roc_plot.plot()

# %%
# Skore project: storing some items
# =================================

# %%
# From your Python code, create and load a skore :class:`~skore.Project`:

# %%
import skore

# sphinx_gallery_start_ignore
import os
import tempfile

temp_dir = tempfile.TemporaryDirectory()
os.environ["SKORE_WORKSPACE"] = temp_dir.name
# sphinx_gallery_end_ignore
my_project = skore.Project("my_project")

# %%
# This will create a skore project directory named ``my_project.skore`` in your
# current working directory.

# %%
# Store some previous results in the skore project for safe-keeping:

# %%
my_project.put("log_report", log_report)

# %%
# Let us store a second model:

# %%
from sklearn.ensemble import RandomForestClassifier

rf_report = EstimatorReport(
    RandomForestClassifier(),
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
my_project.put("rf_report", rf_report)

# %%
# Now, let us retrieve the data that we previously stored:

# %%
metadata = my_project.reports.metadata()
print(type(metadata))

# %%
# We can perform some queries on our stored data:

# %%
from pprint import pprint

report_get = metadata.query("ml_task.str.contains('classification')").reports()
pprint(report_get)

# %%
# For example, we can retrieve the report metrics from the first estimator report:

# %%

# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore

report_get[0].metrics.report_metrics(pos_label=1)

# %%
# .. note::
#   If rendered in a Jupyter notebook, ``metadata`` would render an interactive
#   parallel coordinate plot to search for your preferred model based on some metrics.
#   Here is a screenshot:
#
#   .. image:: https://raw.githubusercontent.com/sylvaincom/sylvaincom.github.io/master/files/probabl/skore/screenshot_quick_start.png
#       :alt: Screenshot

# %%
# .. seealso::
#
#    For a more in-depth guide, see our :ref:`example_skore_getting_started` page!
