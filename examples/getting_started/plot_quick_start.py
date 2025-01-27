"""
.. _example_quick_start:

===========
Quick start
===========
"""

# %%
# From your Python code, create and load a skore :class:`~skore.Project`:

# %%
import skore

my_project = skore.open("my_project", create=True, serve=False)

# %%
# This will create a skore project directory named ``quick_start.skore`` in your
# current working directory.

# %%
# Evaluate your model using skore's :class:`~skore.CrossValidationReport`:

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skore import CrossValidationReport

X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
clf = LogisticRegression()

cv_report = CrossValidationReport(clf, X, y)

# %%
# Display the help tree to see all the insights that are available to you (given that
# you are doing binary classification):

# %%
cv_report.help()

# %%
# Display the report metrics that was computed for you:

# %%
df_cv_report_metrics = cv_report.metrics.report_metrics()
df_cv_report_metrics

# %%
# Display the ROC curve that was generated for you:

# %%
import matplotlib.pyplot as plt

roc_plot = cv_report.metrics.plot.roc()
roc_plot
plt.tight_layout()

# %%
# Store the results in the skore project for safe-keeping:

# %%
my_project.put("df_cv_report_metrics", df_cv_report_metrics)
my_project.put("roc_plot", roc_plot)

# %%
# Retrieve what was stored:

# %%
df_get = my_project.get("df_cv_report_metrics")
df_get

# %%
# .. admonition:: What's next?
#
#    For a more in-depth guide, see our :ref:`example_skore_getting_started` page!

# %%
# Cleanup the project
# -------------------
#
# Let's clear the skore project (to avoid any conflict with other documentation
# examples).

# %%
my_project.clear()
