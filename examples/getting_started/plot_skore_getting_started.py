"""
.. _example_skore_getting_started:

======================
Skore: getting started
======================
"""

# %%
# This getting started guide illustrates how to use skore and why:
#
# #.    Get assistance when developing your ML/DS projects to avoid common pitfalls
#       and follow recommended practices.
#
#       *   :class:`skore.EstimatorReport`: get an insightful report on your estimator
#
#       *   :class:`skore.CrossValidationReport`: get an insightful report on your
#           cross-validation results
#
#       *   :func:`skore.train_test_split`: get diagnostics when splitting your data
#
# #.    Track your ML/DS results using skore's :class:`~skore.Project`
#       (for storage).

# %%
# Machine learning evaluation and diagnostics
# ===========================================
#
# Skore implements new tools or wraps some key scikit-learn class / functions to
# automatically provide insights and diagnostics when using them, as a way to
# facilitate good practices and avoid common pitfalls.

# %%
# Model evaluation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In order to assist its users when programming, skore has implemented a
# :class:`skore.EstimatorReport` class.
#
# Let us load some synthetic data and get the estimator report for a
# :class:`~sklearn.linear_model.LogisticRegression`:

# %%
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from skore import EstimatorReport

X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = LogisticRegression(random_state=0)

est_report = EstimatorReport(
    clf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

# %%
# Now, we can display the help tree to see all the insights that are available to us
# (skore detected that we are doing binary classification):

# %%
est_report.help()

# %%
# We can get the report metrics that was computed for us:

# %%
df_est_report_metrics = est_report.metrics.report_metrics()
df_est_report_metrics

# %%
# We can also plot the ROC curve that was generated for us:

# %%
import matplotlib.pyplot as plt

roc_plot = est_report.metrics.plot.roc()
roc_plot
plt.tight_layout()

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.EstimatorReport`, see :ref:`example_estimator_report`.


# %%
# Cross-validation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# skore has also (re-)implemented a :class:`skore.CrossValidationReport` class that
# contains several :class:`skore.EstimatorReport` for each fold.

# %%
from skore import CrossValidationReport

cv_report = CrossValidationReport(clf, X, y, cv_splitter=5)

# %%
# We display the cross-validation report helper:

# %%
cv_report.help()

# %%
# We display the metrics for each fold:

# %%
df_cv_report_metrics = cv_report.metrics.report_metrics()
df_cv_report_metrics

# %%
# We display the ROC curves for each fold:

# %%
roc_plot_cv = cv_report.metrics.plot.roc()
roc_plot_cv
plt.tight_layout()

# %%
# We can retrieve the estimator report of a specific fold to investigate further,
# for example the first fold:

# %%
est_report_fold = cv_report.estimator_reports_[0]
df_report_metrics_fold = est_report_fold.metrics.report_metrics()
df_report_metrics_fold

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :class:`skore.CrossValidationReport`, see :ref:`example_use_case_employee_salaries`.

# %%
# Train-test split with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Skore has implemented a :func:`skore.train_test_split` function that wraps
# scikit-learn's :func:`sklearn.model_selection.train_test_split`.
#
# Let us load a dataset containing some time series data:

# %%
import pandas as pd
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X, y = dataset.X, dataset.y
X["date_first_hired"] = pd.to_datetime(X["date_first_hired"])
X.head(2)

# %%
# We can observe that there is a ``date_first_hired`` which is time-based.
# Now, let us apply :func:`skore.train_test_split` on this data:

# %%
import skore

X_train, X_test, y_train, y_test = skore.train_test_split(
    X, y, random_state=0, shuffle=False
)

# %%
# We get a ``TimeBasedColumnWarning`` advising us to use
# :class:`sklearn.model_selection.TimeSeriesSplit` instead!
# Indeed, we should not shuffle time-ordered data!

# %%
# .. seealso::
#
#   More methodological advice is available.
#   For more information about the motivation and usage of
#   :func:`skore.train_test_split`, see :ref:`example_train_test_split`.

# %%
# Tracking: skore project
# =======================
#
# A key feature of skore is its :class:`~skore.Project` that allows to store
# items of many types.

# %%
# Setup: creating and loading a skore project
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
#   Let's start by creating a skore project directory named ``my_project.skore`` in our
#   current directory:

# %%

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
# Skore project: storing and retrieving some items
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now that the project exists, we can store some useful items in it (in the same
# directory) using :func:`~skore.Project.put`), with a "universal" key-value convention:

# %%
my_project.put("my_int", 3)
my_project.put("df_cv_report_metrics", df_cv_report_metrics)
my_project.put("roc_plot", roc_plot)

# %%
# .. note ::
#   With the skore :func:`~skore.Project.put`, there is no need to remember the API for
#   each type of object: ``df.to_csv(...)``, ``plt.savefig(...)``, ``np.save(...)``,
#   etc.

# %%
# We can retrieve the value of an item:

# %%
my_project.get("my_int")

# %%
my_project.get("df_cv_report_metrics")

# %%
# .. seealso::
#
#   For more information about the functionalities and the different types
#   of items that we can store in a skore :class:`~skore.Project`,
#   see :ref:`example_working_with_projects`.

# %%
# Tracking the history of items
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Suppose we store several values for a same item called ``my_key_metric``:

# %%
my_project.put("my_key_metric", 4)

my_project.put("my_key_metric", 9)

my_project.put("my_key_metric", 16)

# %%
# Skore does not overwrite items with the same name (key): instead, it stores
# their history so that nothing is lost:

# %%
history = my_project.get("my_key_metric", version="all")
# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore
history

# %%
# These tracking functionalities are very useful to:
#
# * never lose some key machine learning metrics,
# * and observe the evolution over time / runs.

# %%
# .. seealso::
#
#   For more functionalities about the tracking of items using their history,
#   see :ref:`example_tracking_items`.

# %%
# .. admonition:: Stay tuned!
#
#   These are only the initial features: skore is a work in progress and aims to be
#   an end-to-end library for data scientists.
#
#   Feedbacks are welcome: please feel free to join our
#   `Discord <http://discord.probabl.ai>`_ or
#   `create an issue <https://github.com/probabl-ai/skore/issues>`_.
