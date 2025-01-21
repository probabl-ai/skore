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

my_project = skore.open("quick_start", overwrite=True)

# %%
# This will create a skore project directory named ``quick_start.skore`` in your
# current working directory and overwrite any pre-existing project with the
# same path (which you might not want to do that depending on your use case).

# %%
# Evaluate your model using skore's :class:`~skore.CrossValidationReport`:

# %%
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from skore import CrossValidationReport

X, y = load_iris(return_X_y=True)
clf_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

cv_report = CrossValidationReport(clf_pipeline, X, y)

# %%
# Display the help tree to see all the insights that are available to you:

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
# .. admonition:: What's next?
#
#    For a more in-depth guide, see our :ref:`example_skore_getting_started` page!
